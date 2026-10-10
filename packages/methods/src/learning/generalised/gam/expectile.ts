/**
 * Expectile GAMs by asymmetric least squares (Newey and Powell, 1987; Schnabel and Eilers, 2009, "Optimal expectile
 * smoothing"): the $\tau$-expectile curve minimises $\sum_i \lvert \tau - \indicator(y_i < f_i) \rvert (y_i - f_i)^2$,
 * fitted by iterating weighted Gaussian GAM fits with weight $\tau$ above the curve and $1 - \tau$ below it until no
 * point changes side (least asymmetrically weighted squares, LAWS). The first fit has every weight $\tfrac12$, so it
 * is the mean curve; each fit chooses its own smoothing parameters by the model's method. At $\tau = \tfrac12$ the
 * expectile is the mean.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { gaussianFamily } from 'aifn-compute/probability/likelihoods'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import type { Estimator } from 'aifn-compute/learning/estimators'
import { gam, type GamData, type GamModel, type GamParams } from './model'
import { defineModel } from 'aifn-compute/learning/estimators'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A LAWS state. */
export type ExpectileState = Status & {
  /** LAWS iterations done. */
  t: number
  /**
   * The asymmetric weights this iteration's fit used ($n$): $\tfrac12$ everywhere at iteration 0, then $\tau$ above
   * the previous curve and $1 - \tau$ on or below it (before the data's own weights multiply them).
   */
  weights: Tensor
  /** The curve at the data ($n$). */
  fitted: Tensor
  /** This iteration's coefficients $\betavec$ ($P$). */
  coefficients: Tensor
  /** Points whose side of the curve (and so weight) differs from the weights this fit used (0 at iteration 0). */
  switched: number
  /** Share of points strictly below the curve. */
  below: number
  /** Whether no point switched side (never at iteration 0); LAWS then stops. */
  converged: boolean
  /** The smoothing parameters of this iteration's fit. */
  lambdas: number[]
  /** The total EDF of this iteration's fit. */
  edf: number
  /** The smoothing criterion of this iteration's fit (NaN when $\lambda$ is fixed). */
  criterion: number
  /** The curve on `options.grid` ($m$), when one is given. */
  curve?: Tensor
}

/** Options of `expectileLaws` and `expectileFan`. */
export type ExpectileLawsOptions = {
  /** Inputs ($m \times d$) at which every state also records its curve. */
  grid?: Tensor
}

/** Hyperparameters of `expectileGam`. */
export type ExpectileGamParams = Omit<GamParams, 'family' | 'link'> & {
  /** The asymmetry $\tau \in (0, 1)$; $\tfrac12$ gives the mean. */
  tau: number
  /** Most LAWS iterations (default 50). */
  maxLawsSteps?: number
}

/**
 * LAWS as a traceable algorithm: `init` fits the mean curve, and each step refits a Gaussian GAM with the asymmetric
 * weights the last curve implies (times the data's own weights). It is converged when no point changes side. Throws
 * `DomainError` when $\tau \notin (0, 1)$.
 *
 * @param params The terms, the smoothing-parameter choice and $\tau$ (the family and link are Gaussian and identity);
 *   `maxLawsSteps` is not read here.
 * @param data The features, responses and optional prior weights.
 * @param options `grid`, inputs at which every state also records its curve.
 * @returns The algorithm.
 *
 * @example At tau = 0.9 the share of points below the curve rises from about half to about three quarters
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.3))
 * const laws = expectileLaws({ terms: [s(0)], tau: 0.9 }, { x, y })
 * for (const st of trace(laws, undefined, 10).steps) {
 *   print('iteration', st.t, 'switched', st.switched, 'below', st.below)
 * }
 */
export function expectileLaws(
  params: ExpectileGamParams,
  data: GamData,
  options: ExpectileLawsOptions = {},
): Algorithm<void, ExpectileState> {
  const { tau } = params
  if (!(tau > 0 && tau < 1)) throw new DomainError('expectileGam', 'expectileGam: τ must be in (0, 1)')
  const y = Float64Array.from(toFlat(data.y))
  const n = y.length
  const prior = data.weights ? Float64Array.from(toFlat(data.weights)) : new Float64Array(n).fill(1)
  const estimator = gam({ ...params, family: gaussianFamily() })
  const fitWith = (w: Float64Array) =>
    estimator.fit({
      ...data,
      weights: fromData(
        Float64Array.from(w, (v, i) => v * prior[i]),
        [n],
      ),
    })
  /** The asymmetric weights the curve f implies: τ above it, 1 − τ below. */
  const sides = (f: Float64Array) => Float64Array.from(y, (v, i) => (v > f[i] ? tau : 1 - tau))
  const stateOf = (w: Float64Array, model: GamModel, t: number): ExpectileState => {
    const f = Float64Array.from(toFlat(model.fitted))
    const next = sides(f)
    let switched = 0
    let below = 0
    for (let i = 0; i < n; i++) {
      if (t > 0 && next[i] !== w[i]) switched++
      if (y[i] < f[i]) below++
    }
    return {
      t,
      weights: fromData(w, [n]),
      fitted: model.fitted,
      coefficients: model.coefficients,
      switched,
      below: below / n,
      converged: t > 0 && switched === 0,
      lambdas: model.lambdas,
      edf: model.edf,
      criterion: model.smoothingScore.value,
      ...(options.grid ? { curve: model.decide(options.grid) } : {}),
    }
  }
  return {
    name: 'expectile-laws',
    init: () => {
      const w = new Float64Array(n).fill(0.5)
      return stateOf(w, fitWith(w), 0)
    },
    step: (state) => {
      const w = sides(Float64Array.from(toFlat(state.fitted)))
      return stateOf(w, fitWith(w), state.t + 1)
    },
  }
}

/**
 * A $\tau$-expectile GAM: the model of the last LAWS iteration (a Gaussian GAM fitted with the final asymmetric
 * weights), with $\tau$ in `tau` and the LAWS run, of at most `maxLawsSteps` iterations, in `laws`. Fitting throws
 * `DomainError` when $\tau \notin (0, 1)$.
 *
 * @param params The terms, the smoothing-parameter choice, $\tau$ and the most LAWS iterations (see
 *   `ExpectileGamParams`).
 * @returns The estimator; its `fit` takes `GamData`.
 *
 * @example The expectile at tau = 0.5 is the mean; at 0.9 it lies above it
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.3))
 * const grid = tensor([[1], [3], [5]])
 * const mean = gam({ terms: [s(0)] }).fit({ x, y })
 * const half = expectileGam({ terms: [s(0)], tau: 0.5 }).fit({ x, y })
 * const high = expectileGam({ terms: [s(0)], tau: 0.9 }).fit({ x, y })
 * print('mean =', mean.decide(grid))
 * print('tau = 0.5:', half.decide(grid))
 * print('tau = 0.9:', high.decide(grid), 'after', high.laws.final.t, 'LAWS steps')
 */
export function expectileGam(
  params: ExpectileGamParams,
): Estimator<GamData, GamModel & { tau: number; laws: Trace<ExpectileState> }> {
  const { maxLawsSteps = 50 } = params
  return {
    name: 'expectile-gam',
    params,
    fit(data) {
      const laws = trace(expectileLaws(params, data), undefined, maxLawsSteps, {
        record: { switched: (s) => s.switched, below: (s) => s.below },
      })
      const final = laws.final
      const n = final.weights.shape[0]
      const prior = data.weights ? toFlat(data.weights) : null
      const w = Float64Array.from(toFlat(final.weights), (v, i) => v * (prior ? prior[i] : 1))
      const model = gam({ ...params, family: gaussianFamily() }).fit({ ...data, weights: fromData(w, [n]) })
      return { ...model, tau: params.tau, laws }
    },
  }
}

/** Expectile curves at several levels, as plain data (for a worker): every LAWS state of each $\tau$. */
export type ExpectileFan = {
  /** The levels $\tau$, as given. */
  taus: number[]
  /** Per $\tau$, every LAWS state from iteration 0 until no point changes side (or `maxLawsSteps`). */
  runs: ExpectileState[][]
}

/**
 * LAWS at each $\tau$ of `taus` on the same terms and data, keeping every state (with its curve on `options.grid`):
 * the fan of expectile curves a figure draws and plays.
 *
 * @param params The terms, the smoothing-parameter choice and the most LAWS iterations per level (default 50).
 * @param data The features, responses and optional prior weights.
 * @param taus The levels, each in $(0, 1)$.
 * @param options `grid`, inputs at which every state records its curve.
 * @returns The levels and, per level, every LAWS state.
 *
 * @example Three expectile curves at x = 3
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.3))
 * const fan = expectileFan({ terms: [s(0)] }, { x, y }, [0.1, 0.5, 0.9], { grid: tensor([[3]]) })
 * fan.runs.forEach((states, i) => {
 *   const last = states.at(-1)
 *   print('tau', fan.taus[i], 'f(3) =', last.curve, 'below', last.below, 'LAWS steps', last.t)
 * })
 */
export function expectileFan(
  params: Omit<ExpectileGamParams, 'tau'>,
  data: GamData,
  taus: readonly number[],
  options: ExpectileLawsOptions = {},
): ExpectileFan {
  const { maxLawsSteps = 50 } = params
  return {
    taus: [...taus],
    runs: taus.map((tau) => [
      ...trace(expectileLaws({ ...params, tau }, data, options), undefined, maxLawsSteps).steps,
    ]),
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

defineModel(
  {
    key: 'expectileGam',
    module: 'learning/generalised/gam',
    name: 'Expectile GAM',
    summary: 'An additive model of the τ-expectile, fitted by iteratively reweighted least asymmetric squares.',
    task: 'regression',
    capabilities: ['forward', 'decide', 'predictive', 'expect', 'sample'],
    hyper: space({
      tau: real(0.01, 0.99, { default: 0.5, label: 'τ' }),
      method: oneOf(['reml', 'gcv', 'fixed']),
      maxLawsSteps: int(1, 200, { default: 50 }),
    }),
    notes: ['expectile-generalised-additive-models'],
  },
  expectileGam,
)
