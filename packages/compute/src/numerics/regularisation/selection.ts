/**
 * Choosing the Tikhonov weight $\lambda$ (standard form, penalty $\lambda \lVert \xvec \rVert^2$): the L-curve and its
 * corner, and Morozov's discrepancy principle. Both read the residual norm $\rho(\lambda)^{1/2}$ and solution norm
 * $\eta(\lambda)^{1/2}$ off one SVD of $\Amat$, with $\beta_i = \uvec_i^\top\bvec$ and
 * $f_i = s_i^2/(s_i^2 + \lambda)$:
 *
 * $$
 * \rho(\lambda) = \lVert \Amat\xvec_\lambda - \bvec \rVert^2
 * = \sum_i ((1 - f_i)\beta_i)^2 + \lVert \bvec_\perp \rVert^2,
 * \qquad \eta(\lambda) = \lVert \xvec_\lambda \rVert^2 = \sum_i (f_i\beta_i/s_i)^2.
 * $$
 *
 * $\rho$ increases and $\eta$ decreases with $\lambda$. Hansen and O'Leary (1993), "The use of the L-curve in the
 * regularization of discrete ill-posed problems", SIAM J. Sci. Comput. 14(6); Morozov (1966), "On the solution of
 * functional equations by the method of regularization", Soviet Math. Dokl. 7; Hansen (1998), §7.2 and §7.5.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { checkLambda, filtered, spectral, type Spectral } from './tikhonov'

/**
 * $\rho$, $\eta$ and the derivative $\dot\eta = d\eta/d\lambda = -(2/\lambda)\sum_i f_i^2(1 - f_i)\beta_i^2/s_i^2$ at
 * one weight.
 *
 * @param sp The decomposition, as `spectral` returns it.
 * @param lambda The weight $\lambda > 0$.
 * @returns The squared residual norm `rho`, squared solution norm `eta` and the derivative `etaDot`.
 */
function norms(sp: Spectral, lambda: number): { rho: number; eta: number; etaDot: number } {
  let rho = sp.outside
  let eta = 0
  let sum = 0
  for (let i = 0; i < sp.k; i++) {
    const s2 = sp.s[i] * sp.s[i]
    const f = s2 / (s2 + lambda)
    const b2 = sp.beta[i] * sp.beta[i]
    rho += (1 - f) * (1 - f) * b2
    if (s2 > 0) {
      eta += (f * f * b2) / s2
      sum += (f * f * (1 - f) * b2) / s2
    }
  }
  return { rho, eta, etaDot: (-2 / lambda) * sum }
}

/** The result of `lCurve`. */
export type LCurve = {
  /** The penalty weights $\lambda_j$, in the order given (or the default grid, descending). */
  lambdas: Tensor
  /** $\lVert \Amat\xvec_{\lambda_j} - \bvec \rVert$, one per weight. */
  residualNorm: Tensor
  /** $\lVert \xvec_{\lambda_j} \rVert$, one per weight. */
  solutionNorm: Tensor
  /** The signed curvature $\kappa_j$ of the log-log curve, positive where it bends into the corner. */
  curvature: Tensor
  /** The corner: the index of the largest curvature and its weight. */
  corner: { index: number; lambda: number }
}

/**
 * The L-curve of standard-form Tikhonov regularisation: the points $(\log \lVert \Amat\xvec_\lambda - \bvec \rVert,
 * \log \lVert \xvec_\lambda \rVert)$ for a grid of weights, the curvature of that curve at each, and its corner, the
 * weight of largest curvature (Hansen and O'Leary, 1993). The curvature is the closed form of Hansen (2000), "The
 * L-curve and its use in the numerical treatment of inverse problems", rewritten for the weight $\lambda$ on the
 * squared norm:
 *
 * $$
 * \kappa = -\frac{2\eta\rho}{\dot\eta}\,
 * \frac{\lambda\dot\eta\rho + \eta\rho + \lambda^2\eta\dot\eta}{(\lambda^2\eta^2 + \rho^2)^{3/2}},
 * $$
 *
 * with $\rho$, $\eta$ the squared norms and $\dot\eta = d\eta/d\lambda$, signed so that the corner is a positive
 * maximum. Throws `DomainError` for a weight that is not positive and finite.
 *
 * @param A The matrix $\Amat$, $m \times n$. Not modified.
 * @param b The right-hand side $\bvec$, $m$ values. Not modified.
 * @param lambdas The weights $\lambda_j > 0$. Left out, 100 weights $\lambda = \mu^2$ with $\mu$ geometric from $s_1$
 *   down to $\max(s_k, 16\varepsilon s_1)$, as Regularization Tools' `l_curve`.
 * @returns The curve, its curvature and its corner, as `LCurve`.
 *
 * @example The corner of a noisy ill-posed problem: the 6 by 6 Hilbert matrix
 * const A = tensor(Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => 1 / (i + j + 1))))
 * // A times ones, perturbed in the fourth decimal.
 * const b = tensor([2.451, 1.5919, 1.2189, 0.9946, 0.8466, 0.7355])
 * const curve = lCurve(A, b, [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
 * print('residual norms =', curve.residualNorm)
 * print('solution norms =', curve.solutionNorm)
 * print('curvature =', curve.curvature)
 * print('corner =', curve.corner)
 * print('x at the corner =', tikhonov(A, b, { lambda: curve.corner.lambda }).x)
 */
export function lCurve(A: MatrixLike, b: VectorLike, lambdas?: VectorLike): LCurve {
  const sp = spectral(A, b, 'lCurve')
  let ls: dense.F64
  if (lambdas === undefined) {
    const hi = sp.s[0]
    const lo = Math.max(sp.s[sp.k - 1], 16 * 2.220446049250313e-16 * hi)
    if (!(hi > 0)) throw new DomainError('lCurve', 'lCurve: A is zero')
    const count = 100
    ls = Float64Array.from({ length: count }, (_, j) => (hi * (lo / hi) ** (j / (count - 1))) ** 2)
  } else {
    ls = dense.toF64(lambdas, 'lCurve')
  }
  for (const l of ls)
    if (!(l > 0 && Number.isFinite(l)))
      throw new DomainError('lCurve', `lCurve: every lambda must be positive and finite, got ${l}`)
  const r = ls.length
  const residualNorm = new Float64Array(r)
  const solutionNorm = new Float64Array(r)
  const curvature = new Float64Array(r)
  let index = -1
  for (let j = 0; j < r; j++) {
    const l = ls[j]
    const { rho, eta, etaDot } = norms(sp, l)
    residualNorm[j] = Math.sqrt(rho)
    solutionNorm[j] = Math.sqrt(eta)
    curvature[j] =
      ((-2 * eta * rho) / etaDot) *
      ((l * etaDot * rho + eta * rho + l * l * eta * etaDot) / (l * l * eta * eta + rho * rho) ** 1.5)
    if (Number.isFinite(curvature[j]) && (index < 0 || curvature[j] > curvature[index])) index = j
  }
  return {
    lambdas: dense.vec(ls),
    residualNorm: dense.vec(residualNorm),
    solutionNorm: dense.vec(solutionNorm),
    curvature: dense.vec(curvature),
    corner: { index, lambda: index < 0 ? NaN : ls[index] },
  }
}

/** Options of `discrepancyPrinciple`: the noise level, as `delta` or as `sigma`, and the safety factor. */
export type DiscrepancyOptions = {
  /** The norm $\delta = \lVert \evec \rVert$ of the noise in $\bvec$. */
  delta?: number
  /**
   * The standard deviation $\sigma$ of each entry of the noise, for $\delta = \sigma\sqrt{m}$ (when `delta` is not
   * given).
   */
  sigma?: number
  /** The safety factor $\tau \ge 1$ (default 1): the residual is matched to $\tau\delta$. */
  tau?: number
}

/** The result of `discrepancyPrinciple`. */
export type Discrepancy = {
  /**
   * The weight $\lambda$ with $\lVert \Amat\xvec_\lambda - \bvec \rVert = \tau\delta$ (0 or Infinity when not
   * attained).
   */
  lambda: number
  /** The solution $\xvec_\lambda$, $n$ values. */
  x: Tensor
  /** $\lVert \Amat\xvec_\lambda - \bvec \rVert$. */
  residualNorm: number
  /** $\lVert \xvec_\lambda \rVert$. */
  solutionNorm: number
  /** The target $\tau\delta$. */
  target: number
  /**
   * False when no weight attains the target: it is at most the least-squares residual (then `lambda` is 0, the
   * minimum-norm least-squares solution) or at least $\lVert \bvec \rVert$ (then `lambda` is Infinity and $\xvec = 0$).
   */
  attained: boolean
}

/**
 * Morozov's discrepancy principle: the Tikhonov weight $\lambda$ at which the residual matches the noise,
 * $\lVert \Amat\xvec_\lambda - \bvec \rVert = \tau\delta$. The residual norm increases with $\lambda$, from the
 * least-squares residual at $\lambda = 0$ to $\lVert \bvec \rVert$ as $\lambda \to \infty$, so the root is unique;
 * it is found by bisection on $\log\lambda$ to the last bit, each residual from the SVD in $O(k)$. A
 * target outside that range is reported in `attained`, not thrown. Throws `DomainError` when neither `delta` nor
 * `sigma` is a non-negative number, or $\tau < 1$.
 *
 * @param A The matrix $\Amat$, $m \times n$. Not modified.
 * @param b The noisy right-hand side $\bvec$, $m$ values. Not modified.
 * @param options The noise level (`delta` or `sigma`) and the safety factor `tau`, as `DiscrepancyOptions`.
 * @returns The weight, its solution and norms, and whether the target was attained, as `Discrepancy`.
 *
 * @example The weight at which the residual equals the noise
 * const A = tensor([[1, 0], [0, 0.01], [0, 0]])
 * const b = tensor([1, 0.02, 0.05])
 * const d = discrepancyPrinciple(A, b, { delta: 0.06 })
 * print('lambda =', d.lambda, ' residual norm =', d.residualNorm)
 * print('x =', d.x)
 */
export function discrepancyPrinciple(A: MatrixLike, b: VectorLike, options: DiscrepancyOptions): Discrepancy {
  const where = 'discrepancyPrinciple'
  const { tau = 1 } = options
  if (!(tau >= 1 && Number.isFinite(tau))) throw new DomainError(where, `${where}: tau must be at least 1, got ${tau}`)
  const sp = spectral(A, b, where)
  const delta = options.delta ?? (options.sigma === undefined ? NaN : options.sigma * Math.sqrt(sp.m))
  if (!(delta >= 0 && Number.isFinite(delta)))
    throw new DomainError(where, `${where}: give the noise level as delta or sigma, a non-negative number`)
  const target = tau * delta
  const solution = (lambda: number, attained: boolean): Discrepancy => {
    const f = Number.isFinite(lambda)
      ? sp.s.map((si) => (lambda > 0 ? (si * si) / (si * si + lambda) : si > sp.cutoff ? 1 : 0))
      : new Float64Array(sp.k)
    const { x, rho, eta } = filtered(sp, f)
    return {
      lambda,
      x: dense.vec(x),
      residualNorm: Math.sqrt(rho),
      solutionNorm: Math.sqrt(eta),
      target,
      attained,
    }
  }
  const bNorm2 = sp.outside + dense.dot(sp.beta, sp.beta)
  if (target * target >= bNorm2) return solution(Infinity, false)
  if (
    target * target <=
    filtered(
      sp,
      sp.s.map((si) => (si > sp.cutoff ? 1 : 0)),
    ).rho
  )
    return solution(0, false)
  // g(t) = rho(e^t) - target^2 increases in t; bracket the root on a log scale around the squared singular values.
  const g = (t: number) => norms(sp, Math.exp(t)).rho - target * target
  const top = Math.log(sp.s[0] * sp.s[0])
  let lo = top - 40
  let hi = top + 10
  while (g(lo) > 0 && lo > -700) lo -= 40
  while (g(hi) < 0 && hi < 700) hi += 40
  // Bisection to the last bit: g is monotone, and a step costs O(k).
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi)
    if (mid <= lo || mid >= hi) break
    if (g(mid) < 0) lo = mid
    else hi = mid
  }
  const lambda = Math.exp(0.5 * (lo + hi))
  checkLambda(lambda, where)
  return solution(lambda, true)
}
