/**
 * Binary Gaussian-process classification by expectation propagation with the probit likelihood Φ(yf): compute's
 * multivariate EP (`aifn-compute/inference/expectation-propagation`) with the prior N(0, K) and one Gaussian site per training
 * point; the EP log marginal likelihood; and its gradient in K at the EP fixed point.
 *
 * Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", Algorithms 3.5 (EP), 3.6 (predictions),
 * eq. 3.65 (the evidence, in the stable form of the GPML toolbox's infEP) and eq. 5.27 (its gradient).
 */

import { customVjp } from 'aifn-compute/foundation/autodiff'
import {
  multivariateExpectationPropagation,
  probitTilted,
  type MvEpState,
} from 'aifn-compute/inference/expectation-propagation'
import { cholesky, solveTriangular } from 'aifn-compute/numerics/linalg'
import {
  fromData,
  mul,
  toFlat,
  type Matrix,
  type Tensor,
  type Value,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The factor of B = I + S K S with S = diag(s), and R = S B⁻¹ S. Laplace uses s = √W, EP s = √τ̃. Returns L (lower,
 * flat), ½ log|B| and a solver for B⁻¹.
 */
export function stableFactor(K: Float64Array, s: Float64Array) {
  const n = s.length
  const B = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i * n + j] = (i === j ? 1 : 0) + s[i] * K[i * n + j] * s[j]
  const { L } = cholesky(fromData(B, [n, n]))
  const Lf = new Float64Array(toFlat(L))
  let logDetHalf = 0
  for (let i = 0; i < n; i++) logDetHalf += Math.log(Lf[i * n + i])
  // Triangular solves by `aifn-compute/numerics/linalg`'s `solveTriangular`; each returns a fresh flat [n, m] array.
  const flatSolve = (M: Float64Array, m: number, transpose: boolean) =>
    new Float64Array(toFlat(solveTriangular(L, fromData(Float64Array.from(M), [n, m]), { transpose })))
  /** L⁻¹ M for M [n, m] (flat). */
  const forward = (M: Float64Array, m: number) => flatSolve(M, m, false)
  /** L⁻ᵀ M for M [n, m] (flat). */
  const backward = (M: Float64Array, m: number) => flatSolve(M, m, true)
  /** B⁻¹ M for M [n, m] (flat). */
  const solve = (M: Float64Array, m: number) => backward(forward(M, m), m)
  /** R = S B⁻¹ S, [n, n] flat. */
  const R = () => {
    const D = new Float64Array(n * n)
    for (let i = 0; i < n; i++) D[i * n + i] = s[i]
    const X = solve(D, n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) X[i * n + j] *= s[i]
    return X
  }
  return { L: Lf, logDetHalf, solve, forward, R }
}

/** Matrix–vector product of a flat [n, n] matrix. */
function mv(A: Float64Array, v: Float64Array): Float64Array {
  const n = v.length
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += A[i * n + j] * v[j]
    out[i] = s
  }
  return out
}

/** The problem an EP run for GP classification solves. */
export type GpEpProblem = {
  K: Tensor
  /** Labels 0 or 1, [n]. */
  labels: Tensor
  /** Weight of the old site in each update, in [0, 1) (default 0). */
  damping?: number
  /** A sweep in which no site parameter moves by more than this has converged (default 1e-8). */
  tolerance?: number
}

/**
 * Expectation propagation for GP classification with the probit likelihood (Rasmussen and Williams, 2006,
 * Algorithm 3.5), one site update per step: compute's `multivariateExpectationPropagation` with the prior N(0, K), one
 * site per latent value fᵢ (identity projections) and the tilted moments of q₋ᵢ(fᵢ) Φ(yᵢfᵢ) (`probitTilted`, yᵢ = ±1).
 * The state's `logEvidence` is EP's log marginal likelihood (eq. 3.65) as of the last sweep's end. `init` takes
 * optional starting sites (a warm start); by default every site is 1 (τ̃ = ν̃ = 0).
 */
export function gpEp(
  problem: GpEpProblem,
): Algorithm<{ sitePrecision?: Tensor; siteShift?: Tensor } | void, MvEpState> {
  const n = problem.K.shape[0]
  const y = toFlat(problem.labels)
  if (!y.every((v) => v === 0 || v === 1)) throw new DomainError('gpEp', 'gpEp: labels must be 0 or 1')
  const sign = Float64Array.from(y, (v) => 2 * v - 1)
  const alg = multivariateExpectationPropagation({
    prior: { mean: fromData(new Float64Array(n), [n]) as Vector, covariance: problem.K as Matrix },
    tilted: (i, cavity) => probitTilted(cavity.mean, cavity.variance, sign[i]),
    damping: problem.damping,
    tolerance: problem.tolerance,
  })
  return { ...alg, name: 'gp-expectation-propagation' }
}

/**
 * The predictive weights of a converged EP run: the latent mean at x* is k*ᵀα with
 * α = ν̃ − S̃^½ B⁻¹ S̃^½ K ν̃ (R&W Algorithm 3.6), and s = √τ̃ for the variance.
 */
export function epWeights(K: Float64Array, tau: Float64Array, nu: Float64Array) {
  const n = tau.length
  const s = tau.map((v) => Math.sqrt(Math.max(v, 0)))
  const factor = stableFactor(K, s)
  const Knu = mv(K, nu)
  const inner = factor.solve(
    Float64Array.from(Knu, (v, i) => s[i] * v),
    1,
  )
  const alpha = Float64Array.from(nu, (v, i) => v - s[i] * inner[i])
  return { alpha, s, factor, n }
}

/** Options of `gpEpEvidence` and the EP fits. */
export type GpEpOptions = {
  /** Most EP sweeps (default 100). */
  maxSweeps?: number
  /** Site-change tolerance per sweep (default 1e-8). */
  tolerance?: number
  damping?: number
}

/**
 * The EP log marginal likelihood as a differentiable function of the Gram matrix K (labels 0 and 1, probit link).
 * Its reverse rule is ∂ log Z_EP/∂K = ½ ααᵀ − ½ S̃^½ B⁻¹ S̃^½ at the EP fixed point (R&W eq. 5.27; the sites are
 * stationary there, so no implicit term appears). Successive calls start EP from the previous call's sites.
 */
export function gpEpEvidence(labels: Tensor, options: GpEpOptions = {}): (K: Value) => Value {
  const { maxSweeps = 100, tolerance = 1e-8, damping = 0 } = options
  let warm: { sitePrecision: Tensor; siteShift: Tensor } | null = null
  const solve = (K: Tensor) => {
    const n = K.shape[0]
    const alg = gpEp({ K, labels, tolerance, damping })
    const run1 = (start: typeof warm) => run(alg, start ?? {}, maxSweeps * n)
    let final = run1(warm)
    // A warm start from another K's sites can fail where a cold start does not.
    if (!Number.isFinite(final.logEvidence) && warm) final = run1(null)
    if (Number.isFinite(final.logEvidence)) warm = { sitePrecision: final.sitePrecision, siteShift: final.siteShift }
    return final
  }
  const gradient = (K: Tensor, final: MvEpState): Tensor => {
    const Kf = new Float64Array(toFlat(K))
    const { alpha, factor, n } = epWeights(
      Kf,
      new Float64Array(toFlat(final.sitePrecision)),
      new Float64Array(toFlat(final.siteShift)),
    )
    const R = factor.R()
    const G = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) G[i * n + j] = 0.5 * alpha[i] * alpha[j] - 0.5 * R[i * n + j]
    return fromData(G, [n, n])
  }
  return customVjp(
    (K: Value) => solve(K as Tensor).logEvidence as Value,
    (K: Value) => {
      const final = solve(K as Tensor)
      return { out: final.logEvidence as Value, residuals: gradient(K as Tensor, final) }
    },
    (G: Tensor, cot: Value) => [mul(cot, G)],
  )
}
