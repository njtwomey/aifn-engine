/**
 * Expectile GAMs trained by any fitter. The $\tau$-expectile curve at fixed smoothing parameters minimises the
 * penalised asymmetric squared loss
 *
 * $$
 * J_\tau(\betavec) = \frac{\sum_i w_i \rho_\tau(y_i - \xvec_i^\top\betavec) + \betavec^\top\Smat_\lambda\betavec}{2n},
 * \qquad \rho_\tau(u) = \lvert \tau - \indicator(u < 0) \rvert \, u^2,
 * $$
 *
 * convex and continuously differentiable (Newey and Powell, 1987; Schnabel and Eilers, 2009). `expectileProblem` writes
 * $J_\tau$ as a `GamProblem` (objective by primitives through `expectileLoss`, so autodiff gives $\nabla J_\tau$;
 * the closed form $\nabla J_\tau = (\Smat_\lambda\betavec - \Xmat^\top(\wvec \odot \avec \odot \rvec))/n$
 * with residuals $\rvec$ and $a_i = \tau$ above the curve and $1 - \tau$ below), so gradient descent, SGD, Adam and
 * L-BFGS of `fitters.ts` minimise it end to end. LAWS (`expectileLaws`) is the P-IRLS of this problem: a Newton step
 * on each region of fixed residual signs.
 */

import { add, div, fromData, matmul, mul, sum, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { stream } from 'aifn-compute/foundation/random'
import { expectileLoss } from 'aifn-compute/learning/losses'
import { expectile } from 'aifn-compute/probability/stats'
import { gaussianFamily } from 'aifn-compute/probability/likelihoods'
import { penalisedFit } from '../smoothing'
import { expectileLaws, type ExpectileGamParams, type ExpectileState } from './expectile'
import { gamFitter, type GamFitMethod, type GamFitState, type GamFitterOptions } from './fitters'
import { gamProblem, type GamData, type GamEvaluation, type GamProblem } from './problem'
import { DomainError } from 'aifn-compute/foundation/errors'

type F64 = Float64Array
/**
 * A vector tensor holding a copy of some numbers.
 *
 * @param a The values; not modified.
 * @returns A float64 tensor of shape $[n]$ for $n$ values.
 */
const vec = (a: ArrayLike<number>) => fromData(Float64Array.from(a), [a.length])

/**
 * The $\tau$-expectile problem on a Gaussian identity-link GAM problem's design, data and penalty: $J_\tau$ (see the
 * module comment). `evaluate` reports $\sum_i w_i\rho_\tau$ as the deviance, `working` the asymmetric weights at
 * $\betavec$ (so `gamModel` gives the EDF and bands of the weighted fit there), `optimum` is LAWS at these $\lambda$
 * (at most 100 reweightings), and `nullDeviance` is the loss of the best constant, the weighted $\tau$-expectile of
 * $\yvec$. The base's $\lambda$, start and curvature (a bound on $J_\tau$'s) are kept. As $\rho_{1/2}(u) = u^2/2$
 * halves the loss but not the penalty, the minimiser at $\tau = \tfrac12$ is the base's fit at $2\lambda$. Throws
 * `DomainError` when $\tau \notin (0, 1)$ or the base is not Gaussian with the identity link.
 *
 * @param base The Gaussian identity-link problem whose design, data and penalty $J_\tau$ uses.
 * @param tau The level $\tau \in (0, 1)$.
 * @returns $J_\tau$ as a `GamProblem`, for any fitter of `fitters.ts`.
 *
 * @example L-BFGS on the 0.9-expectile objective reaches the LAWS optimum
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.3))
 * const base = gamProblem({ terms: [s(0)] }, { x, y })
 * const problem = expectileProblem(base, 0.9)
 * print('L-BFGS: J =', run(gamLbfgs(problem), undefined, 50).objective)
 * print('LAWS: J =', problem.evaluate(problem.optimum.beta).objective)
 * print('best constant: J =', problem.nullDeviance / (2 * 40))
 */
export function expectileProblem(base: GamProblem, tau: number): GamProblem {
  if (!(tau > 0 && tau < 1)) throw new DomainError('expectileProblem', `expectileProblem: τ = ${tau} is not in (0, 1)`)
  if (base.family.name !== 'gaussian' || base.link.name !== 'identity')
    throw new DomainError(
      'expectileProblem',
      'expectileProblem: the base problem must be Gaussian with the identity link',
    )
  const A = base.design
  const { n, P, X } = A
  const { y, w, o, penalty } = base
  const sideWeight = (r: number) => (r < 0 ? 1 - tau : tau)
  const linear = (beta: ArrayLike<number>) => {
    const f = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let v = o[i]
      for (let a = 0; a < P; a++) v += X[i * P + a] * beta[a]
      f[i] = v
    }
    return f
  }
  const quadratic = (beta: ArrayLike<number>, from = 0, to = P) => {
    let s = 0
    for (let a = from; a < to; a++) for (let b = from; b < to; b++) s += beta[a] * penalty[a * P + b] * beta[b]
    return s
  }

  // J_τ by primitives (β differentiated; the data enter as constants).
  const St = fromData(penalty, [P, P])
  const full = { X: fromData(X, [n, P]), y: fromData(y, [n]), w: fromData(w, [n]), o: fromData(o, [n]), scale: 1 }
  const batchOf = (rows: ArrayLike<number>) => {
    const b = rows.length
    const Xb = new Float64Array(b * P)
    const pick = (a: F64) =>
      fromData(
        Float64Array.from(rows, (i) => a[i]),
        [b],
      )
    for (let r = 0; r < b; r++) Xb.set(X.subarray(rows[r] * P, (rows[r] + 1) * P), r * P)
    return { X: fromData(Xb, [b, P]), y: pick(y), w: pick(w), o: pick(o), scale: n / b }
  }
  const objective = (beta: Value, rows?: ArrayLike<number>): Value => {
    const B = rows ? batchOf(rows) : full
    const f = add(matmul(B.X, beta), B.o)
    const loss = sum(mul(B.w, expectileLoss(f, B.y, { expectile: tau, reduction: 'none' })))
    const pen = sum(mul(beta, matmul(St, beta)))
    return div(add(mul(B.scale, loss), pen), 2 * n)
  }
  const gradient = (beta: ArrayLike<number>) => {
    const f = linear(beta)
    const g = new Float64Array(P)
    for (let i = 0; i < n; i++) {
      const r = y[i] - f[i]
      const c = w[i] * sideWeight(r) * r
      for (let a = 0; a < P; a++) g[a] -= X[i * P + a] * c
    }
    for (let a = 0; a < P; a++) {
      let s = 0
      for (let b = 0; b < P; b++) s += penalty[a * P + b] * beta[b]
      g[a] = (g[a] + s) / n
    }
    return g
  }
  const evaluate = (beta: ArrayLike<number>): GamEvaluation => {
    const f = linear(beta)
    let dev = 0
    for (let i = 0; i < n; i++) {
      const r = y[i] - f[i]
      dev += w[i] * sideWeight(r) * r * r
    }
    const pen = quadratic(beta)
    return {
      eta: f,
      mu: f,
      valid: true,
      deviance: dev,
      penalty: pen,
      termPenalties: A.terms.map((t, j) => quadratic(beta, A.offsets[j], A.offsets[j] + t.size)),
      penalisedDeviance: dev + pen,
      objective: (dev + pen) / (2 * n),
    }
  }
  const working = (beta: ArrayLike<number>) => {
    const f = linear(beta)
    return {
      z: Float64Array.from(y, (v, i) => v - o[i]),
      W: Float64Array.from(f, (v, i) => w[i] * sideWeight(y[i] - v)),
    }
  }
  // LAWS at these λ: weighted fits until no point changes side.
  let optimum: ReturnType<typeof penalisedFit> | undefined
  const solve = () => {
    let a = new Float64Array(n).fill(0.5)
    let fit = penalisedFit(
      A,
      { y: base.data.y, weights: vec(a.map((v, i) => v * w[i])) },
      base.family,
      base.link,
      penalty,
      50,
    )
    for (let it = 0; it < 100; it++) {
      const f = linear(fit.beta)
      const next = Float64Array.from(f, (v, i) => sideWeight(y[i] - v))
      if (next.every((v, i) => v === a[i])) break
      a = next
      fit = penalisedFit(
        A,
        { y: base.data.y, weights: vec(a.map((v, i) => v * w[i])) },
        base.family,
        base.link,
        penalty,
        50,
        fit.beta,
      )
    }
    return fit
  }
  // The loss of the best constant: the weighted τ-expectile of y.
  const e0 = expectile(y, tau, { weights: w })
  let nullDeviance = 0
  for (let i = 0; i < n; i++) nullDeviance += w[i] * sideWeight(y[i] - e0) * (y[i] - e0) ** 2

  return {
    kind: 'gam-problem',
    spec: base.spec,
    family: gaussianFamily(),
    link: base.link,
    design: A,
    data: base.data,
    y,
    w,
    o,
    lambdas: base.lambdas,
    penalty,
    active: base.active,
    smoothing: base.smoothing,
    get optimum() {
      return (optimum ??= solve())
    },
    start: base.start,
    nullDeviance,
    // ∇²J_τ = (XᵀW_τX + S_λ)/n with W_τ ≤ w, so the Gaussian problem's curvature bounds it.
    get curvature() {
      return base.curvature
    },
    objective,
    gradient,
    evaluate,
    working,
  }
}

/** The fitters of an expectile: `p-irls` means LAWS; the others run on `expectileProblem`. */
export type ExpectileFitMethod = Exclude<GamFitMethod, 'backfitting'>

/** One fitter's run in an `expectileTrainingRun`. */
export type ExpectileRunRequest = {
  /** The fitter (`p-irls` means the LAWS states themselves). */
  method: ExpectileFitMethod
  /** The fitter's options. */
  options?: GamFitterOptions
  /** The most steps to take (not read for `p-irls`). */
  steps: number
  /** The seed of the fitter's stream (default `'expectile-training'`). */
  seed?: string | number
}

/** A fitter's state on $J_\tau$, with the share of points below the curve. */
export type ExpectileFitState = GamFitState & { below: number }

/** Expectile fits at several levels by one or more fitters, as plain data (for a worker). */
export type ExpectileTrainingRun = {
  /** The levels $\tau$, as given. */
  taus: number[]
  /** One entry per level. */
  runs: {
    /** The level. */
    tau: number
    /** $\lambda$ where LAWS settled (by the model's method): the $\lambda$ every fitter then uses. */
    lambdas: number[]
    /** The smoothing criterion of LAWS's last fit. */
    criterion: number
    /** The total EDF of LAWS's last fit. */
    edf: number
    /** LAWS iterations to settle. */
    lawsSteps: number
    /** The minimiser of $J_\tau$ at those $\lambda$ (LAWS's last coefficients) and $J_\tau$ there. */
    optimum: { coefficients: Tensor; objective: number }
    /** Per request: every state from step 0. */
    fits: { method: ExpectileFitMethod; states: ExpectileFitState[] }[]
  }[]
}

/**
 * For each $\tau$: LAWS chooses $\lambda$ (the model's method, re-selected at each iteration) and gives the minimiser
 * of $J_\tau$; each requested fitter then minimises $J_\tau$ at that $\lambda$ from the common start, keeping every
 * state. A `p-irls` request reports the LAWS iterations themselves, scored on $J_\tau$ at the final $\lambda$.
 *
 * @param params The terms, the smoothing-parameter choice and the most LAWS iterations (default 50).
 * @param data The features, responses and optional prior weights.
 * @param taus The levels, each in $(0, 1)$.
 * @param requests The fitters to run at each level.
 * @returns Per level, LAWS's choices and optimum and every state of each fitter.
 *
 * @example LAWS and L-BFGS agree on the 0.9-expectile
 * const r = stream(0)
 * const x = uniform(r, 0, 6, { shape: [40, 1] })
 * const y = add(sin(reshape(x, [40])), normals(r, 40, 0, 0.3))
 * const out = expectileTrainingRun({ terms: [s(0)] }, { x, y }, [0.9], [
 *   { method: 'p-irls', steps: 0 },
 *   { method: 'lbfgs', steps: 30 },
 * ])
 * const level = out.runs[0]
 * print('LAWS steps =', level.lawsSteps, 'optimum J =', level.optimum.objective)
 * for (const fit of level.fits) {
 *   const last = fit.states.at(-1)
 *   print(fit.method, 'J =', last.objective, 'below =', last.below)
 * }
 */
export function expectileTrainingRun(
  params: Omit<ExpectileGamParams, 'tau'>,
  data: GamData,
  taus: readonly number[],
  requests: readonly ExpectileRunRequest[],
): ExpectileTrainingRun {
  const { maxLawsSteps = 50 } = params
  return {
    taus: [...taus],
    runs: taus.map((tau) => {
      const laws = [...trace(expectileLaws({ ...params, tau }, data), undefined, maxLawsSteps).steps]
      const final = laws[laws.length - 1]
      const base = gamProblem(
        {
          terms: params.terms,
          family: 'gaussian',
          lambdas: final.lambdas,
          gamma: params.gamma,
          maxSteps: params.maxSteps,
        },
        data,
      )
      const problem = expectileProblem(base, tau)
      const below = (beta: ArrayLike<number>) => {
        const f = problem.evaluate(beta).mu
        let c = 0
        for (let i = 0; i < f.length; i++) if (problem.y[i] < f[i]) c++
        return c / f.length
      }
      const lawsState = (s: ExpectileState): ExpectileFitState => {
        const beta = toFlat(s.coefficients)
        const e = problem.evaluate(beta)
        return {
          t: s.t,
          coefficients: s.coefficients,
          objective: e.objective,
          deviance: e.deviance,
          penalisedDeviance: e.penalisedDeviance,
          gradNorm: Math.hypot(...problem.gradient(beta)),
          stepSize: NaN,
          working: null,
          workingWeights: null,
          batch: null,
          epochs: s.t,
          converged: s.converged,
          diverged: false,
          below: s.below,
        }
      }
      const fits = requests.map(({ method, options, steps, seed }) => {
        if (method === 'p-irls') return { method, states: laws.map(lawsState) }
        const tr = trace(gamFitter(problem, method, options), undefined, steps, {
          stream: stream(seed ?? 'expectile-training'),
        })
        const states = tr.steps.map((st) => {
          const { inner: _inner, ...plain } = st as GamFitState & { inner?: unknown }
          return { ...(plain as GamFitState), below: below(toFlat(plain.coefficients)) }
        })
        return { method, states }
      })
      const beta = toFlat(final.coefficients)
      return {
        tau,
        lambdas: final.lambdas,
        criterion: final.criterion,
        edf: final.edf,
        lawsSteps: laws.length - 1,
        optimum: { coefficients: final.coefficients, objective: problem.evaluate(beta).objective },
        fits,
      }
    }),
  }
}
