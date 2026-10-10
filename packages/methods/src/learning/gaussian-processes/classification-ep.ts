/**
 * Binary Gaussian-process classification by expectation propagation with the probit likelihood $\Phi(yf)$: compute's
 * multivariate EP (`aifn-compute/inference/expectation-propagation`) with the prior $\Gauss(\zeros, \Kmat)$ and one
 * Gaussian site per training point; the EP log marginal likelihood; and its gradient in $\Kmat$ at the EP fixed point.
 * Labels are 0 and 1, with $y = \pm 1$ in the likelihood.
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
 * The factor of $\Bmat = \Imat + \Smat \Kmat \Smat$ with $\Smat = \diag(\svec)$, and $\Rmat = \Smat \Bmat^{-1} \Smat$.
 * Laplace uses $\svec = \sqrt{\diag \Wmat}$, EP $\svec = \sqrt{\tilde\tauvec}$. Jitter is added if $\Bmat$ needs it
 * and is not reported.
 *
 * @param K The Gram matrix $\Kmat$ as a row-major array of $n^2$ values; not modified.
 * @param s The diagonal $\svec$ of $\Smat$, $n$ values.
 * @returns `L`, the lower Cholesky factor of $\Bmat$ (row-major); `logDetHalf`, $\frac{1}{2} \log\lvert \Bmat \rvert$;
 *   `forward(M, m)`, $\Lmat^{-1}\Mmat$, and `solve(M, m)`, $\Bmat^{-1}\Mmat$, for a row-major $n \times m$ array
 *   $\Mmat$ (each a new array); and `R()`, $\Rmat$ as a row-major array.
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

/**
 * The matrix–vector product $\Amat\vvec$.
 *
 * @param A The matrix $\Amat$ as a row-major array of $n^2$ values.
 * @param v The vector $\vvec$, $n$ values.
 * @returns A new array of the $n$ values of $\Amat\vvec$.
 */
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
  /** The prior covariance $\Kmat$ of the latent values, $n \times n$. */
  K: Tensor
  /** Labels 0 or 1, `[n]`. */
  labels: Tensor
  /** Weight of the old site in each update, in $[0, 1)$ (default 0). */
  damping?: number
  /** A sweep in which no site parameter moves by more than this has converged (default 1e-8). */
  tolerance?: number
}

/**
 * Expectation propagation for GP classification with the probit likelihood (Rasmussen and Williams, 2006,
 * Algorithm 3.5), one site update per step: compute's `multivariateExpectationPropagation` with the prior
 * $\Gauss(\zeros, \Kmat)$, one site per latent value $f_i$ (identity projections) and the tilted moments of
 * $q_{-i}(f_i) \Phi(y_i f_i)$ (`probitTilted`, $y_i = \pm 1$). The state's `logEvidence` is EP's log marginal
 * likelihood (eq. 3.65) as of the last sweep's end. `init` takes optional starting sites (a warm start); by default
 * every site is 1 ($\tilde\tau = \tilde\nu = 0$). Throws `DomainError` for a label other than 0 or 1.
 *
 * @param problem The Gram matrix, the labels, the damping and the tolerance.
 * @returns The algorithm; a sweep is $n$ steps.
 *
 * @example Three points: EP converges in a few sweeps
 * // The RBF Gram matrix at 0, 1 and 2 (lengthscale 1), and the labels 0, 0, 1
 * const K = tensor([[1, 0.61, 0.14], [0.61, 1, 0.61], [0.14, 0.61, 1]])
 * const s = run(gpEp({ K, labels: tensor([0, 0, 1]) }), {}, 100)
 * print('sweeps', s.sweep, ' converged', s.converged)
 * print('posterior mean', s.mean, ' log Z_EP', s.logEvidence)
 * print('site precisions', s.sitePrecision)
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
 * The predictive weights of a converged EP run: the latent mean at $\xvec_*$ is $\kvec_*^\top\alphavec$ with
 * $\alphavec = \tilde\nuvec - \tilde\Smat^{1/2} \Bmat^{-1} \tilde\Smat^{1/2} \Kmat \tilde\nuvec$ (R&W
 * Algorithm 3.6), and $\svec = \sqrt{\tilde\tauvec}$ for the variance (negative site precisions taken as 0).
 *
 * @param K The Gram matrix $\Kmat$ as a row-major array of $n^2$ values.
 * @param tau The site precisions $\tilde\tauvec$, $n$ values.
 * @param nu The site shifts $\tilde\nuvec$, $n$ values.
 * @returns `alpha`, $\alphavec$; `s`, $\svec$; `factor`, the `stableFactor` of $\Bmat$; and `n`.
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
  /** Weight of the old site in each update, in $[0, 1)$ (default 0). */
  damping?: number
}

/**
 * The EP log marginal likelihood as a differentiable function of the Gram matrix $\Kmat$ (labels 0 and 1, probit
 * link). Its reverse rule is the gradient
 * $\frac{1}{2} \alphavec\alphavec^\top - \frac{1}{2} \tilde\Smat^{1/2} \Bmat^{-1} \tilde\Smat^{1/2}$ of
 * $\log Z_{\mathrm{EP}}$ in $\Kmat$ at the EP fixed point (R&W eq. 5.27; the sites are stationary there, so no
 * implicit term appears). Successive calls start EP from the previous call's sites, and cold again when that gives a
 * non-finite value.
 *
 * @param labels The labels, 0 or 1, `[n]`.
 * @param options The most sweeps, the site-change tolerance and the damping.
 * @returns The function from $\Kmat$ ($n \times n$, possibly traced) to $\log Z_{\mathrm{EP}}$.
 *
 * @example The evidence and its derivative along $c\Kmat$, against a central difference
 * const K = tensor([[1, 0.61, 0.14], [0.61, 1, 0.61], [0.14, 0.61, 1]])
 * const evidence = gpEpEvidence(tensor([0, 0, 1]))
 * print('log Z_EP', evidence(K))
 * print('d/dc at c = 1', grad((c) => evidence(mul(c, K)))(1))
 * print('central difference', (evidence(mul(1.001, K)) - evidence(mul(0.999, K))) / 0.002)
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
