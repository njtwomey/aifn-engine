/**
 * Tikhonov regularisation of a linear least-squares problem: the $\xvec_\lambda$ minimising
 *
 * $$\lVert \Amat\xvec - \bvec \rVert^2 + \lambda \lVert \Lmat\xvec \rVert^2, \qquad \lambda \ge 0,$$
 *
 * with the penalty weight $\lambda$ on the squared norm (not $\lambda^2$, as Hansen writes it). With $\Lmat = \Imat$
 * (standard form) this is ridge regression without an intercept: `tikhonov(X, y, { lambda })` is
 * `linearRegression({ l2: lambda, intercept: false })` of `aifn-methods/learning/linear`, and scikit-learn's
 * `Ridge(alpha=lambda, fit_intercept=False)`.
 *
 * In standard form everything follows from the singular value decomposition
 * $\Amat = \sum_i s_i \uvec_i \vvec_i^\top$: with $\beta_i = \uvec_i^\top\bvec$, the solution is a filtered expansion
 *
 * $$\xvec_\lambda = \sum_i f_i \frac{\beta_i}{s_i} \vvec_i, \qquad f_i = \frac{s_i^2}{s_i^2 + \lambda},$$
 *
 * so one decomposition gives the solution, $\lVert \xvec_\lambda \rVert^2 = \sum_i (f_i \beta_i / s_i)^2$ and
 * $\lVert \Amat\xvec_\lambda - \bvec \rVert^2 = \sum_i ((1 - f_i)\beta_i)^2 + \lVert \bvec_\perp \rVert^2$ for every
 * $\lambda$, $\bvec_\perp$ the part of $\bvec$ outside the range of $\Amat$. Tikhonov (1963), "Solution of incorrectly
 * formulated problems and the regularization method", Soviet Math. Dokl. 4; Hansen (1998), "Rank-Deficient and Discrete
 * Ill-Posed Problems", SIAM, §4.2 and §5.1.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { lstsq, svd } from 'aifn-compute/numerics/linalg'

type F64 = dense.F64

/**
 * The singular value decomposition of $\Amat$ ($m \times n$, $k = \min(m, n)$) with the right-hand side in its
 * coordinates: what every spectral filter of this module is computed from.
 */
export type Spectral = {
  /** The number of rows $m$ of $\Amat$. */
  m: number
  /** The number of columns $n$ of $\Amat$. */
  n: number
  /** The number of singular values $k = \min(m, n)$. */
  k: number
  /** The singular values $s_i$, $k$ values in descending order. */
  s: F64
  /** The right singular vectors as columns, $n \times k$ row-major. */
  V: F64
  /** The coefficients $\beta_i = \uvec_i^\top\bvec$, $k$ values. */
  beta: F64
  /** $\lVert \bvec_\perp \rVert^2$, the squared norm of the part of $\bvec$ no $\xvec$ can fit (0 when $m \le n$). */
  outside: number
  /** Singular values at most this are treated as zero when $\lambda = 0$: $\max(m, n)\,\varepsilon\,s_1$. */
  cutoff: number
}

/**
 * Decompose $\Amat$ and express $\bvec$ in its left singular vectors. Throws `ShapeError` when `b` does not have $m$
 * entries.
 *
 * @param A The matrix $\Amat$, $m \times n$. Not modified.
 * @param b The right-hand side $\bvec$, $m$ values. Not modified.
 * @param where The caller's name for error messages.
 * @returns The decomposition with $\bvec$'s coefficients, as `Spectral`.
 */
export function spectral(A: MatrixLike, b: VectorLike, where: string): Spectral {
  const { data, m, n } = dense.toMatrixF64(A, where)
  const bb = dense.toF64(b, where)
  if (bb.length !== m) throw new ShapeError(where, `${where}: A has ${m} rows but b has ${bb.length} entries`)
  const { U, S, V } = svd(dense.mat(data, m, n))
  const k = Math.min(m, n)
  const u = dense.data(U)
  const beta = dense.matTVec(u, bb, m, k)
  const fitted = dense.matVec(u, beta, m, k)
  const rest = dense.sub(bb, fitted)
  const s = dense.data(S)
  return {
    m,
    n,
    k,
    s,
    V: dense.data(V),
    beta,
    outside: dense.dot(rest, rest),
    cutoff: Math.max(m, n) * 2.220446049250313e-16 * (s[0] ?? 0),
  }
}

/**
 * The Tikhonov filter factor of one singular value: $s^2/(s^2 + \lambda)$, or at $\lambda = 0$ 1 above the cutoff and
 * 0 at or below it (the minimum-norm least-squares solution).
 *
 * @param s The singular value $s \ge 0$.
 * @param lambda The penalty weight $\lambda \ge 0$.
 * @param cutoff The rank cutoff used at $\lambda = 0$.
 * @returns The filter factor, in $[0, 1]$.
 */
function factor(s: number, lambda: number, cutoff: number): number {
  if (lambda > 0) return (s * s) / (s * s + lambda)
  return s > cutoff ? 1 : 0
}

/**
 * The filtered solution $\sum_i f_i (\beta_i / s_i) \vvec_i$ with its squared residual and solution norms.
 *
 * @param sp The decomposition, as `spectral` returns it.
 * @param f The filter factors $f_i$, $k$ values; a zero factor drops its term whatever $s_i$ is.
 * @returns The solution `x` ($n$ values), $\lVert \Amat\xvec - \bvec \rVert^2$ as `rho` and
 *   $\lVert \xvec \rVert^2$ as `eta`.
 */
export function filtered(sp: Spectral, f: ArrayLike<number>): { x: F64; rho: number; eta: number } {
  const c = new Float64Array(sp.k)
  let rho = sp.outside
  let eta = 0
  for (let i = 0; i < sp.k; i++) {
    c[i] = f[i] > 0 ? (f[i] * sp.beta[i]) / sp.s[i] : 0
    rho += ((1 - f[i]) * sp.beta[i]) ** 2
    eta += c[i] * c[i]
  }
  return { x: dense.matVec(sp.V, c, sp.n, sp.k), rho, eta }
}

/**
 * Check a penalty weight.
 *
 * @param lambda The weight given.
 * @param where The caller's name for error messages.
 */
export function checkLambda(lambda: number, where: string): void {
  if (!(lambda >= 0 && Number.isFinite(lambda)))
    throw new DomainError(where, `${where}: lambda must be finite and non-negative, got ${lambda}`)
}

/**
 * The Tikhonov filter factors $f_i = s_i^2/(s_i^2 + \lambda)$ of the singular values $s_i$ for penalty weight
 * $\lambda$: how much of each spectral component of the least-squares solution survives. Components with
 * $s_i^2 \gg \lambda$ pass ($f_i \approx 1$), those with $s_i^2 \ll \lambda$ are damped ($f_i \approx s_i^2/\lambda$).
 * At $\lambda = 0$ every non-zero singular value has factor 1 and a zero one factor 0. Throws `DomainError` for a
 * negative or non-finite $\lambda$.
 *
 * @param s The singular values $s_i \ge 0$, in any order.
 * @param lambda The penalty weight $\lambda \ge 0$ on $\lVert \xvec \rVert^2$.
 * @returns The factors, one per singular value, each in $[0, 1]$.
 *
 * @example Large singular values pass, small ones are damped
 * print('f =', filterFactors([10, 1, 0.1], 1))
 * print('no penalty =', filterFactors([10, 1, 0], 0))
 */
export function filterFactors(s: VectorLike, lambda: number): Tensor {
  checkLambda(lambda, 'filterFactors')
  return dense.vec(dense.toF64(s, 'filterFactors').map((si) => factor(si, lambda, 0)))
}

/** Options of `tikhonov`. */
export type TikhonovOptions = {
  /** The penalty weight $\lambda \ge 0$ on $\lVert \Lmat\xvec \rVert^2$ (the `l2` of ridge regression). */
  lambda: number
  /**
   * The regularisation matrix $\Lmat$, $p \times n$ (default $\Imat$, standard form): a difference operator
   * (`differenceOperator`) penalises roughness instead of size.
   */
  L?: MatrixLike
}

/** The result of `tikhonov`. */
export type TikhonovResult = {
  /** The regularised solution $\xvec_\lambda$, $n$ values. */
  x: Tensor
  /** $\lVert \Amat\xvec_\lambda - \bvec \rVert$. */
  residualNorm: number
  /** $\lVert \Lmat\xvec_\lambda \rVert$ ($\lVert \xvec_\lambda \rVert$ in standard form). */
  solutionNorm: number
  /** The filter factors $f_i = s_i^2/(s_i^2 + \lambda)$, $\min(m, n)$ values; null for a general $\Lmat$. */
  filterFactors: Tensor | null
  /** The singular values of $\Amat$, descending; null for a general $\Lmat$. */
  singularValues: Tensor | null
}

/**
 * Tikhonov regularisation: the $\xvec$ minimising $\lVert \Amat\xvec - \bvec \rVert^2 + \lambda \lVert \Lmat\xvec
 * \rVert^2$. In standard form ($\Lmat = \Imat$) it is the SVD filter $\xvec = \sum_i f_i (\beta_i/s_i)\vvec_i$; with
 * a general $\Lmat$ ($p \times n$) it is the least-squares solution of the stacked system
 * $\begin{bmatrix}\Amat \\ \sqrt{\lambda}\,\Lmat\end{bmatrix}\xvec \approx \begin{bmatrix}\bvec \\ \zeros\end{bmatrix}$
 * (the minimum-norm one when $\Amat$ and $\Lmat$ share a null vector). At $\lambda = 0$ it is the minimum-norm
 * least-squares solution. Throws `DomainError` for a negative or non-finite $\lambda$ and `ShapeError` for
 * mismatched shapes.
 *
 * The weight is on the squared norm, so `tikhonov(X, y, { lambda })` equals ridge regression with `l2 = lambda` and
 * no intercept (`linearRegression` of `aifn-methods/learning/linear`; scikit-learn's `Ridge(alpha=lambda,
 * fit_intercept=False)`).
 *
 * @param A The matrix $\Amat$, $m \times n$. Not modified.
 * @param b The right-hand side $\bvec$, $m$ values. Not modified.
 * @param options The weight `lambda` and the optional regularisation matrix `L`, as `TikhonovOptions`.
 * @returns The solution with its residual and solution norms, and in standard form the filter factors and singular
 *   values, as `TikhonovResult`.
 *
 * @example Damping the small singular values of an ill-conditioned system
 * const A = tensor([[1, 1], [1, 1.0001]])
 * const b = tensor([2, 2.0001])
 * for (const lambda of [0, 1e-6, 1e-2]) {
 *   const r = tikhonov(A, b, { lambda })
 *   print('lambda =', lambda, ' x =', r.x, ' f =', r.filterFactors)
 * }
 *
 * @example Penalising roughness with a first-difference operator
 * const A = tensor([[1, 0, 0], [0, 0, 1]])
 * const b = tensor([0, 1])
 * const r = tikhonov(A, b, { lambda: 0.1, L: differenceOperator(3) })
 * print('x =', r.x)
 * print('residual norm =', r.residualNorm, ' roughness =', r.solutionNorm)
 */
export function tikhonov(A: MatrixLike, b: VectorLike, options: TikhonovOptions): TikhonovResult {
  const { lambda, L } = options
  checkLambda(lambda, 'tikhonov')
  if (L === undefined) {
    const sp = spectral(A, b, 'tikhonov')
    const f = sp.s.map((si) => factor(si, lambda, sp.cutoff))
    const { x, rho, eta } = filtered(sp, f)
    return {
      x: dense.vec(x),
      residualNorm: Math.sqrt(rho),
      solutionNorm: Math.sqrt(eta),
      filterFactors: dense.vec(f),
      singularValues: dense.vec(sp.s),
    }
  }
  const { data: a, m, n } = dense.toMatrixF64(A, 'tikhonov')
  const bb = dense.toF64(b, 'tikhonov')
  if (bb.length !== m) throw new ShapeError('tikhonov', `tikhonov: A has ${m} rows but b has ${bb.length} entries`)
  const { data: l, m: p, n: q } = dense.toMatrixF64(L, 'tikhonov')
  if (q !== n) throw new ShapeError('tikhonov', `tikhonov: A has ${n} columns but L has ${q}`)
  const stacked = new Float64Array((m + p) * n)
  stacked.set(a)
  const root = Math.sqrt(lambda)
  for (let i = 0; i < p * n; i++) stacked[m * n + i] = root * l[i]
  const rhs = new Float64Array(m + p)
  rhs.set(bb)
  const x = dense.data(lstsq(dense.mat(stacked, m + p, n), dense.vec(rhs)).x)
  return {
    x: dense.vec(x),
    residualNorm: dense.norm(dense.sub(dense.matVec(a, x, m, n), bb)),
    solutionNorm: dense.norm(dense.matVec(l, x, p, n)),
    filterFactors: null,
    singularValues: null,
  }
}

/** The result of `tikhonovPath`. */
export type TikhonovPath = {
  /** The penalty weights $\lambda_j$, as given ($r$ values). */
  lambdas: Tensor
  /** The solutions, $r \times n$: row $j$ is $\xvec_{\lambda_j}$. */
  x: Tensor
  /** $\lVert \Amat\xvec_{\lambda_j} - \bvec \rVert$, $r$ values. */
  residualNorm: Tensor
  /** $\lVert \xvec_{\lambda_j} \rVert$, $r$ values. */
  solutionNorm: Tensor
  /** The filter factors, $r \times k$: row $j$ holds $s_i^2/(s_i^2 + \lambda_j)$. */
  filterFactors: Tensor
  /** The singular values $s_i$ of $\Amat$, $k = \min(m, n)$ values, descending. */
  singularValues: Tensor
}

/**
 * Standard-form Tikhonov solutions for many penalty weights from one SVD of $\Amat$: the solutions, their residual
 * and solution norms and their filter factors, as a figure of the regularisation path (or the ridge path) needs.
 * Throws `DomainError` for a negative or non-finite weight and `ShapeError` for mismatched shapes.
 *
 * @param A The matrix $\Amat$, $m \times n$. Not modified.
 * @param b The right-hand side $\bvec$, $m$ values. Not modified.
 * @param lambdas The penalty weights $\lambda_j \ge 0$, in any order ($r$ values).
 * @returns The path, one row or entry per weight in the order given, as `TikhonovPath`.
 *
 * @example Residual against solution norm as the weight grows
 * const A = tensor([[3, 0], [0, 0.1], [0, 0]])
 * const b = tensor([3, 0.2, 0.1])
 * const path = tikhonovPath(A, b, [0, 0.01, 1, 100])
 * print('x =', path.x)
 * print('residual norms =', path.residualNorm)
 * print('solution norms =', path.solutionNorm)
 */
export function tikhonovPath(A: MatrixLike, b: VectorLike, lambdas: VectorLike): TikhonovPath {
  const ls = dense.toF64(lambdas, 'tikhonovPath')
  ls.forEach((l) => checkLambda(l, 'tikhonovPath'))
  const sp = spectral(A, b, 'tikhonovPath')
  const r = ls.length
  const xs = new Float64Array(r * sp.n)
  const fs = new Float64Array(r * sp.k)
  const residualNorm = new Float64Array(r)
  const solutionNorm = new Float64Array(r)
  for (let j = 0; j < r; j++) {
    const f = sp.s.map((si) => factor(si, ls[j], sp.cutoff))
    const { x, rho, eta } = filtered(sp, f)
    xs.set(x, j * sp.n)
    fs.set(f, j * sp.k)
    residualNorm[j] = Math.sqrt(rho)
    solutionNorm[j] = Math.sqrt(eta)
  }
  return {
    lambdas: dense.vec(ls),
    x: dense.mat(xs, r, sp.n),
    residualNorm: dense.vec(residualNorm),
    solutionNorm: dense.vec(solutionNorm),
    filterFactors: dense.mat(fs, r, sp.k),
    singularValues: dense.vec(sp.s),
  }
}

/**
 * The discrete derivative operator of order $d$ on $n$ points, the $(n - d) \times n$ matrix whose row $i$ holds the
 * coefficients $(-1)^{d - j}\binom{d}{j}$, $j = 0, \dots, d$, from column $i$: $[-1, 1]$ for $d = 1$ and
 * $[1, -2, 1]$ for $d = 2$ (Hansen, 1998, §1.3, Regularization Tools' `get_l`). As $\Lmat$ in `tikhonov` it
 * penalises roughness, and its null space (polynomials of degree below $d$) is left unpenalised. Order 0 is the
 * identity. Throws `DomainError` unless $0 \le d < n$ are integers.
 *
 * @param n The number of points $n$, the columns of the operator.
 * @param order The order $d$ of the difference (default 1).
 * @returns The operator, $(n - d) \times n$.
 *
 * @example First and second differences on four points
 * print('first =', differenceOperator(4))
 * print('second =', differenceOperator(4, 2))
 */
export function differenceOperator(n: number, order = 1): Tensor {
  if (!(Number.isInteger(n) && Number.isInteger(order) && order >= 0 && order < n))
    throw new DomainError('differenceOperator', `differenceOperator: need integers 0 <= order < n, got ${order}, ${n}`)
  const c = new Float64Array(order + 1)
  let binomial = 1
  for (let j = 0; j <= order; j++) {
    c[j] = ((order - j) % 2 === 0 ? 1 : -1) * binomial
    binomial = (binomial * (order - j)) / (j + 1)
  }
  const rows = n - order
  const out = new Float64Array(rows * n)
  for (let i = 0; i < rows; i++) for (let j = 0; j <= order; j++) out[i * n + i + j] = c[j]
  return dense.mat(out, rows, n)
}
