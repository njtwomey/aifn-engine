/**
 * Iteratively reweighted least squares (IRLS; Nelder and Wedderburn, 1972; McCullagh and Nelder, 1989, §2.5) for any
 * family and link, with an optional quadratic penalty βᵀPβ (penalised IRLS, as `aifn-methods/learning/gam` uses it;
 * Wood, 2017, "Generalized Additive Models", 2nd ed., §6.1.1). Each step solves the weighted least-squares problem
 * (XᵀWX + P)β = XᵀWz for the working response z = η − o + (y − μ)/μ′(η) and weights W = w μ′(η)²/V(μ); a step that
 * leaves the mean space or raises the penalised deviance is halved towards the previous coefficients.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { cholesky, choleskySolve, lstsq } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { likelihood, type Family, type Link } from 'aifn-compute/probability/likelihoods'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The problem an IRLS run solves. */
export type IrlsProblem = {
  /** Design matrix X [n, p]. */
  design: Tensor
  /** Responses y [n] (binomial: proportions). */
  y: Tensor
  family: Family
  link: Link
  /** Prior weights w [n] (binomial: trials); default 1. */
  weights?: Tensor
  /** Offset o [n] added to the linear predictor; default 0. */
  offset?: Tensor
  /** Penalty matrix P [p, p] added to XᵀWX (e.g. ridge λI or a smoothing penalty Σλⱼ Sⱼ); default none. */
  penalty?: Tensor
  /** Converged when |D − D_old| / (|D| + 0.1) < tolerance for the penalised deviance D (default 1e-8, as R's `glm`). */
  tolerance?: number
}

/** One IRLS state. */
export type IrlsState = Status & {
  /** Steps taken. */
  t: number
  /** β [p]; null at step 0, which starts from the family's initial mean rather than from coefficients. */
  coefficients: Tensor | null
  /** Linear predictor η = Xβ + o [n]. */
  eta: Tensor
  /** Fitted means μ = g⁻¹(η) [n]. */
  mu: Tensor
  /** Deviance Σ wᵢ d(yᵢ, μᵢ). */
  deviance: number
  /** Deviance + βᵀPβ. */
  penalisedDeviance: number
  /** Working response z and working weights W at η, for the next step. */
  working: Tensor
  workingWeights: Tensor
  /** Step halvings taken to reach this state. */
  halvings: number
  /** XᵀWX + P was singular in the step that reached this state; the minimum-norm solution was taken. */
  singular: boolean
  converged: boolean
  /** No valid step could be found (μ left the mean space or the deviance was not finite). */
  diverged: boolean
}

type Dense = {
  X: Float64Array
  n: number
  p: number
  y: Float64Array
  w: Float64Array
  o: Float64Array
  P: Float64Array | null
}

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

const vec = (a: Float64Array) => fromData(a, [a.length])
const flat = (t: Tensor | number) => (typeof t === 'number' ? Float64Array.of(t) : Float64Array.from(toFlat(t)))

/** Deviance Σ wᵢ d(yᵢ, μᵢ). */
export function deviance(family: Family, y: Tensor, mu: Tensor, weights?: Tensor): number {
  const d = flat(family.unitDeviance(y, mu) as Tensor)
  const w = weights ? flat(weights) : null
  let s = 0
  for (let i = 0; i < d.length; i++) s += (w ? w[i] : 1) * d[i]
  return s
}

/** μ, z and W at η. */
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

function penaltyOf(D: Dense, beta: Float64Array | null): number {
  if (!D.P || !beta) return 0
  let s = 0
  for (let a = 0; a < D.p; a++) for (let b = 0; b < D.p; b++) s += beta[a] * D.P[a * D.p + b] * beta[b]
  return s
}

/** Solve (XᵀWX + P)β = XᵀWz by Cholesky, or by minimum-norm least squares (reported) when it is singular. */
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
 * IRLS as a traceable algorithm. `init` starts from the family's initial mean μ₀ (so η₀ = g(μ₀)) or from given
 * coefficients. For a canonical link IRLS is Newton's method on the (penalised) log-likelihood; otherwise it is Fisher
 * scoring.
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
