/**
 * Expectile GAMs by asymmetric least squares (Newey and Powell, 1987; Schnabel and Eilers, 2009, "Optimal expectile
 * smoothing"): the τ-expectile curve minimises Σ |τ − 1(yᵢ < fᵢ)| (yᵢ − fᵢ)², fitted by iterating weighted Gaussian
 * GAM fits with weight τ above the curve and 1 − τ below it until no point changes side (LAWS).
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
  /** Asymmetric weights τ or 1 − τ per point, [n]. */
  weights: Tensor
  /** The curve at the data, [n]. */
  fitted: Tensor
  /** This iteration's coefficients β [P]. */
  coefficients: Tensor
  /** Points whose side of the curve (and so weight) differs from the weights this fit used. */
  switched: number
  /** Share of points below the curve. */
  below: number
  converged: boolean
  /** The smoothing parameters, total EDF and smoothing criterion (NaN when λ is fixed) of this iteration's fit. */
  lambdas: number[]
  edf: number
  criterion: number
  /** The curve on `options.grid`, when one is given, [m]. */
  curve?: Tensor
}

/** Options of `expectileLaws` and `expectileFan`. */
export type ExpectileLawsOptions = {
  /** Inputs [m, d] at which every state also records its curve. */
  grid?: Tensor
}

/** Hyperparameters of `expectileGam`. */
export type ExpectileGamParams = Omit<GamParams, 'family' | 'link'> & {
  /** The asymmetry τ ∈ (0, 1); ½ gives the mean. */
  tau: number
  /** Most LAWS iterations (default 50). */
  maxLawsSteps?: number
}

/** LAWS as a traceable algorithm; each step is one weighted GAM fit. */
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
 * A τ-expectile GAM: the model of the last LAWS iteration (a Gaussian GAM fitted with the final asymmetric weights),
 * with the LAWS run in `laws`.
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

/** Expectile curves at several levels, as plain data (for a worker): every LAWS state of each τ. */
export type ExpectileFan = {
  taus: number[]
  /** Per τ, every LAWS state from iteration 0 until no point changes side (or `maxLawsSteps`). */
  runs: ExpectileState[][]
}

/**
 * LAWS at each τ of `taus` on the same terms and data, keeping every state (with its curve on `options.grid`): the
 * fan of expectile curves a figure draws and plays.
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
