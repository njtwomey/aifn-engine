/**
 * Iteratively reweighted least squares (IRLS; Nelder and Wedderburn, 1972; McCullagh and Nelder, 1989, §2.5) for any
 * family and link, with an optional quadratic penalty $\betavec^\top\Pmat\betavec$ (penalised IRLS, as
 * `aifn-methods/learning/generalised/gam` uses it; Wood, 2017, "Generalized Additive Models", 2nd ed., §6.1.1). Each
 * step solves the weighted least-squares problem $(\Xmat^\top\Wmat\Xmat + \Pmat)\betavec = \Xmat^\top\Wmat\zvec$
 * for the working response $z_i = \eta_i - o_i + (y_i - \mu_i)/\mu'(\eta_i)$ and weights
 * $W_{ii} = w_i \mu'(\eta_i)^2 / V(\mu_i)$; a step that leaves the mean space or raises the penalised deviance is
 * halved towards the previous coefficients. The deviance is computed from $\eta$ (stable where $\mu$ rounds to the
 * edge of the mean space), and a singular system takes the minimum-norm solution and is reported, not thrown.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { cholesky, choleskySolve, lstsq } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { likelihood, type Family, type Link } from 'aifn-compute/probability/likelihoods'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The problem an IRLS run solves. */
export type IrlsProblem = {
  /** Design matrix $\Xmat$, $n \times p$. */
  design: Tensor
  /** Responses $\yvec$, $n$ (binomial: proportions). */
  y: Tensor
  /** The response family: variance function, deviance, starting mean and mean space. */
  family: Family
  /** The link $g$, one the family takes. */
  link: Link
  /** Prior weights $\wvec$, $n$ (binomial: trials); default 1. */
  weights?: Tensor
  /** Offset $\ovec$, $n$, added to the linear predictor; default 0. */
  offset?: Tensor
  /**
   * Penalty matrix $\Pmat$, $p \times p$, added to $\Xmat^\top\Wmat\Xmat$ (e.g. ridge $\lambda\Imat$ or a smoothing
   * penalty $\sum_j \lambda_j \Smat_j$); default none.
   */
  penalty?: Tensor
  /**
   * Converged when $\lvert D - D_{\text{old}}\rvert / (\lvert D\rvert + 0.1)$ is below this, for the penalised
   * deviance $D$ (default 1e-8, as R's `glm`).
   */
  tolerance?: number
}

/** One IRLS state. */
export type IrlsState = Status & {
  /** Steps taken. */
  t: number
  /**
   * $\betavec$, $p$; null at step 0 when it starts from the family's initial mean rather than from coefficients.
   */
  coefficients: Tensor | null
  /** Linear predictor $\etavec = \Xmat\betavec + \ovec$, $n$. */
  eta: Tensor
  /** Fitted means $\muvec = g^{-1}(\etavec)$, $n$. */
  mu: Tensor
  /** Deviance $\sum_i w_i d(y_i, \mu_i)$. */
  deviance: number
  /** Deviance $+ \betavec^\top\Pmat\betavec$. */
  penalisedDeviance: number
  /** Working response $\zvec$ at $\etavec$ (offset removed), $n$, for the next step. */
  working: Tensor
  /** Working weights $W_{ii}$ at $\etavec$, $n$, for the next step (0 where $\mu$ is at the edge of the mean space). */
  workingWeights: Tensor
  /** Step halvings taken to reach this state. */
  halvings: number
  /**
   * $\Xmat^\top\Wmat\Xmat + \Pmat$ was singular in the step that reached this state; the minimum-norm solution was
   * taken.
   */
  singular: boolean
  /** The penalised deviance changed by less than the tolerance, or no step could lower it below rounding. */
  converged: boolean
  /** No valid step could be found ($\muvec$ left the mean space or the deviance was not finite). */
  diverged: boolean
}

/**
 * The problem as plain arrays: the design `X` (row-major, $n p$ values), its shape `n` and `p`, responses `y`, prior
 * weights `w` and offset `o` (defaults filled in), and the penalty `P` ($p^2$ values, or null).
 */
type Dense = {
  X: Float64Array
  n: number
  p: number
  y: Float64Array
  w: Float64Array
  o: Float64Array
  P: Float64Array | null
}

/**
 * Copy a problem into plain arrays, filling in unit weights and a zero offset. Throws `ShapeError` when the responses
 * and the design's rows differ in number.
 *
 * @param problem The problem.
 * @returns Its arrays.
 */
function dense(problem: IrlsProblem): Dense {
  const [n, p] = problem.design.shape
  const X = Float64Array.from(toFlat(problem.design))
  const y = Float64Array.from(toFlat(problem.y))
  if (y.length !== n) throw new ShapeError('irls', `irls: ${n} rows of X but ${y.length} responses`)
  const w = problem.weights ? Float64Array.from(toFlat(problem.weights)) : new Float64Array(n).fill(1)
  const o = problem.offset ? Float64Array.from(toFlat(problem.offset)) : new Float64Array(n)
  const P = problem.penalty ? Float64Array.from(toFlat(problem.penalty)) : null
  return { X, n, p, y, w, o, P }
}

/**
 * A vector tensor over an array, without copying.
 *
 * @param a The values.
 * @returns A tensor of shape $[\text{length}]$.
 */
const vec = (a: Float64Array) => fromData(a, [a.length])
/**
 * A tensor's values (or a number) as a fresh `Float64Array`.
 *
 * @param t The tensor or number.
 * @returns A copy of its values (one value for a number).
 */
const flat = (t: Tensor | number) => (typeof t === 'number' ? Float64Array.of(t) : Float64Array.from(toFlat(t)))

/**
 * The deviance $D = \sum_i w_i d(y_i, \mu_i)$ with the family's unit deviance $d$ (twice the log-likelihood ratio of
 * the saturated model, per observation), from the means; R's `deviance` of a `glm`.
 *
 * @param family The response family.
 * @param y The responses, $n$.
 * @param mu The fitted means, $n$.
 * @param weights Prior weights $w_i$, $n$ (default 1).
 * @returns The deviance.
 *
 * @example Poisson deviance of three counts, and the saturated model's
 * import { poissonFamily } from 'aifn-compute/probability/likelihoods'
 * const y = tensor([0, 1, 5])
 * print('D at mu = 1, 2, 3:', deviance(poissonFamily(), y, tensor([1, 2, 3])))
 * print('D at mu = y:', deviance(poissonFamily(), y, tensor([0, 1, 5])))
 */
export function deviance(family: Family, y: Tensor, mu: Tensor, weights?: Tensor): number {
  const d = flat(family.unitDeviance(y, mu) as Tensor)
  const w = weights ? flat(weights) : null
  let s = 0
  for (let i = 0; i < d.length; i++) s += (w ? w[i] : 1) * d[i]
  return s
}

/**
 * The means, working response and working weights at a linear predictor. Where $V(\mu) = 0$ or $\mu' = 0$ (the mean
 * rounded to the edge of its space) the weight is 0 and $z = \eta - o$.
 *
 * @param problem The problem (family and link).
 * @param D Its arrays (prior weights, responses, offset).
 * @param eta The linear predictor $\etavec$, $n$; read only.
 * @returns `mu`, `z` and `W`, each $n$ values.
 */
function atEta(problem: IrlsProblem, D: Dense, eta: Float64Array) {
  const etaT = vec(eta)
  const mu = flat(problem.link.inverse(etaT) as Tensor)
  const d = flat(problem.link.derivative(etaT) as Tensor)
  const V = flat(problem.family.variance(vec(mu)) as Tensor)
  // Where μ has rounded to the edge of the mean space (V = 0 or μ′ = 0) the observation carries no weight in this
  // step; its z is then η itself, so it pulls on nothing.
  const W = Float64Array.from(d, (di, i) => {
    const v = (D.w[i] * di * di) / V[i]
    return Number.isFinite(v) ? v : 0
  })
  const z = Float64Array.from(eta, (e, i) => (W[i] > 0 ? e - D.o[i] + (D.y[i] - mu[i]) / d[i] : e - D.o[i]))
  return { mu, z, W }
}

/**
 * The penalty $\betavec^\top\Pmat\betavec$ (0 without a penalty or coefficients).
 *
 * @param D The problem's arrays (the penalty $\Pmat$).
 * @param beta The coefficients, $p$, or null at the start.
 * @returns The penalty.
 */
function penaltyOf(D: Dense, beta: Float64Array | null): number {
  if (!D.P || !beta) return 0
  let s = 0
  for (let a = 0; a < D.p; a++) for (let b = 0; b < D.p; b++) s += beta[a] * D.P[a * D.p + b] * beta[b]
  return s
}

/**
 * Solve $(\Xmat^\top\Wmat\Xmat + \Pmat)\betavec = \Xmat^\top\Wmat\zvec$ by Cholesky (no jitter), or by
 * minimum-norm least squares (reported) when it is singular.
 *
 * @param X The design, row-major, $n p$ values.
 * @param n The number of rows.
 * @param p The number of columns.
 * @param W The working weights, $n$.
 * @param z The working response, $n$.
 * @param P The penalty, $p^2$ values, or null.
 * @returns `beta`, the solution; `singular`, whether the Cholesky factorisation failed; `A`, the matrix
 *   $\Xmat^\top\Wmat\Xmat + \Pmat$ ($p^2$ values, row-major).
 */
function weightedSolve(
  X: Float64Array,
  n: number,
  p: number,
  W: Float64Array,
  z: Float64Array,
  P: Float64Array | null,
): { beta: Float64Array; singular: boolean; A: Float64Array } {
  const A = new Float64Array(p * p)
  const b = new Float64Array(p)
  for (let i = 0; i < n; i++) {
    const wi = W[i]
    if (wi === 0) continue
    for (let a = 0; a < p; a++) {
      const xa = X[i * p + a] * wi
      if (xa === 0) continue
      b[a] += xa * z[i]
      for (let c = 0; c <= a; c++) A[a * p + c] += xa * X[i * p + c]
    }
  }
  for (let a = 0; a < p; a++) for (let c = a + 1; c < p; c++) A[a * p + c] = A[c * p + a]
  if (P) for (let k = 0; k < p * p; k++) A[k] += P[k]
  const f = cholesky(fromData(A, [p, p]), { jitter: false })
  if (!f.failed) return { beta: flat(choleskySolve(f.L, vec(b)) as Tensor), singular: false, A }
  // A singular XᵀWX (e.g. collinear columns): the minimum-norm solution, as for any least-squares problem.
  return { beta: flat(lstsq(fromData(A, [p, p]), vec(b)).x), singular: true, A }
}

/**
 * The linear predictor $\Xmat\betavec + \ovec$.
 *
 * @param X The design, row-major, $n p$ values.
 * @param n The number of rows.
 * @param p The number of columns.
 * @param beta The coefficients, $p$.
 * @param o The offset, $n$.
 * @returns $\etavec$, $n$ values.
 */
function matVec(X: Float64Array, n: number, p: number, beta: Float64Array, o: Float64Array): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = o[i]
    for (let a = 0; a < p; a++) s += X[i * p + a] * beta[a]
    out[i] = s
  }
  return out
}

/**
 * IRLS as a traceable algorithm. `init` starts from the family's initial mean $\muvec_0$ (so
 * $\etavec_0 = g(\muvec_0)$; its deviance is that of $\muvec_0$, near the saturated model's) or from given
 * coefficients. For a canonical link IRLS is Newton's method on the (penalised) log-likelihood; otherwise it is Fisher
 * scoring. A step halves towards the previous coefficients up to 30 times; when none lowers the penalised deviance,
 * the run has converged if the trials were within rounding of it and has diverged otherwise. The first step from the
 * initial mean is accepted whatever its deviance. Throws `ShapeError` when the responses and design rows differ in
 * number.
 *
 * @param problem The design, responses, family and link, and the optional weights, offset, penalty and tolerance.
 * @returns The algorithm, for `run` or `trace`; its start is `{}` or `{ coefficients }`.
 *
 * @example A Poisson GLM in two groups converges in a few steps to the logs of the group means
 * import { link, poissonFamily } from 'aifn-compute/probability/likelihoods'
 * // Columns: group indicator, intercept. Group means 2 and 7.
 * const design = tensor([[0, 1], [0, 1], [0, 1], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
 * const y = tensor([1, 3, 0, 4, 6, 9, 2, 11])
 * const record = { deviance: (state) => state.deviance }
 * const t = trace(irls({ design, y, family: poissonFamily(), link: link('log') }), {}, 25, { record })
 * print('deviance by step =', t.series.deviance)
 * print('steps =', t.final.t, ' converged =', t.final.converged)
 * print('exp(coefficients) =', exp(t.final.coefficients))
 */
export function irls(problem: IrlsProblem): Algorithm<{ coefficients?: Tensor }, IrlsState> {
  const D = dense(problem)
  const tol = problem.tolerance ?? 1e-8
  // The deviance from η (stable where μ rounds to the edge of the mean space, e.g. σ(40) = 1), and validity: any
  // finite η for a link onto the mean space, else μ inside it.
  const lik = likelihood(problem.family, problem.link.name)
  const devOf = (eta: Float64Array) => {
    const d = flat(lik.unitDeviance(problem.y, vec(eta)) as Tensor)
    let s = 0
    for (let i = 0; i < d.length; i++) s += D.w[i] * d[i]
    return s
  }
  const validAt = (eta: Float64Array, mu: Float64Array) =>
    problem.link.total ? eta.every(Number.isFinite) : problem.family.validMean(vec(mu))
  const stateAt = (
    beta: Float64Array | null,
    eta: Float64Array,
    t: number,
    halvings: number,
    singular: boolean,
    previous: number | null,
    diverged = false,
  ): IrlsState => {
    const q = atEta(problem, D, eta)
    const dev = devOf(eta)
    const pdev = dev + penaltyOf(D, beta)
    return {
      t,
      coefficients: beta ? vec(beta) : null,
      eta: vec(eta),
      mu: vec(q.mu),
      deviance: dev,
      penalisedDeviance: pdev,
      working: vec(q.z),
      workingWeights: vec(q.W),
      halvings,
      singular,
      converged: previous !== null && Math.abs(pdev - previous) / (Math.abs(pdev) + 0.1) < tol,
      diverged: diverged || !Number.isFinite(pdev),
    }
  }
  return {
    name: 'glm-irls',
    init: ({ coefficients } = {}) => {
      if (coefficients) {
        const beta = flat(coefficients)
        return stateAt(beta, matVec(D.X, D.n, D.p, beta, D.o), 0, 0, false, null)
      }
      const mu0 = problem.family.initialMean(problem.y, problem.weights ?? vec(D.w))
      const eta0 = flat(problem.link.link(mu0) as Tensor)
      return stateAt(null, eta0, 0, 0, false, null)
    },
    step: (state) => {
      const z = flat(state.working)
      const W = flat(state.workingWeights)
      const { beta: proposed, singular } = weightedSolve(D.X, D.n, D.p, W, z, D.P)
      const first = !state.coefficients
      // From the starting mean there are no previous coefficients to halve towards; use the least-squares projection
      // of the starting η onto the columns of X instead (as a fallback only, when the first step is invalid).
      const old = state.coefficients
        ? flat(state.coefficients)
        : weightedSolve(
            D.X,
            D.n,
            D.p,
            new Float64Array(D.n).fill(1),
            Float64Array.from(flat(state.eta), (e, i) => e - D.o[i]),
            null,
          ).beta
      let beta = proposed
      let halvings = 0
      let closest = Infinity
      for (; halvings <= 30; halvings++) {
        const eta = matVec(D.X, D.n, D.p, beta, D.o)
        const mu = flat(problem.link.inverse(vec(eta)) as Tensor)
        const pdev = validAt(eta, mu) ? devOf(eta) + penaltyOf(D, beta) : NaN
        if (Number.isFinite(pdev)) closest = Math.min(closest, Math.abs(pdev - state.penalisedDeviance))
        const ok = Number.isFinite(pdev) && (first || pdev <= state.penalisedDeviance * (1 + 1e-10) + 1e-12)
        if (ok) return stateAt(beta, eta, state.t + 1, halvings, singular, state.penalisedDeviance)
        beta = Float64Array.from(beta, (b, j) => (b + old[j]) / 2)
      }
      // No step lowers the penalised deviance: at its minimum to rounding (the trials differ from it by no more than
      // rounding), the run has converged; otherwise no valid step exists and it has diverged.
      const atPrecision = closest <= 1e-9 * (Math.abs(state.penalisedDeviance) + 1)
      return { ...state, t: state.t + 1, halvings, converged: atPrecision, diverged: !atPrecision }
    },
  }
}
