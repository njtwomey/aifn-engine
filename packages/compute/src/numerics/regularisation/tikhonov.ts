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
 *
 * A general $\Lmat$ ($p \times n$) is brought to standard form once (Eldén, 1982; Hansen, 1998, §2.3.1): with
 * $\Lmat = \Umat_1 \Dmat \Vmat_1^\top$ its SVD on its rank $r$, $\Wmat$ an orthonormal basis of its null space and
 * $\Pmat_\perp$ the projector onto the complement of the range of $\Amat\Wmat$, the problem in $\yvec = \Lmat\xvec$
 * has matrix $\bar\Amat = \Pmat_\perp \Amat \Vmat_1 \Dmat^{-1}$ and right-hand side $\bar\bvec = \Pmat_\perp\bvec$, and
 * $\xvec_\lambda = \Kmat\yvec_\lambda + \xvec_0$ with the unpenalised part $\xvec_0$ fixed. The singular values of
 * $\bar\Amat$ are the generalised singular values $\gamma_i$ of the pair $(\Amat, \Lmat)$, its filter factors
 * $\gamma_i^2/(\gamma_i^2 + \lambda)$, and $\lVert \Lmat\xvec_\lambda \rVert = \lVert \yvec_\lambda \rVert$, so one SVD
 * of $\bar\Amat$ again serves every $\lambda$.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { lstsq, qr, svd } from 'aifn-compute/numerics/linalg'

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

/** A general-form problem brought to standard form: the spectral data of $\bar\Amat$, $\bar\bvec$ and the map back. */
export type StandardForm = {
  /** The decomposition of $\bar\Amat$ ($m \times r$) with $\bar\bvec$ in its coordinates. */
  sp: Spectral
  /** $\Kmat$, $n \times r$ row-major: $\xvec = \Kmat\yvec + \xvec_0$. */
  K: F64
  /** $\xvec_0$, the part of every solution in the null space of $\Lmat$ that $\yvec$ does not set, $n$ values. */
  x0: F64
  /** The number of unknowns $n$. */
  n: number
}

/**
 * Bring $\min \lVert \Amat\xvec - \bvec \rVert^2 + \lambda \lVert \Lmat\xvec \rVert^2$ to standard form in
 * $\yvec = \Lmat\xvec$ (Eldén, 1982), for any $\Lmat$ ($p \times n$, any rank above 0): one SVD of $\Lmat$, a
 * complete QR for its null space $\Wmat$, one SVD of $\Amat\Wmat$ and one of $\bar\Amat$. When $\Amat$ and
 * $\Lmat$ share a null vector the part of $\xvec$ along it is set to zero (the minimum-norm solution). Throws
 * `ShapeError` for mismatched shapes and `DomainError` for $\Lmat = \zeros$.
 *
 * @param A The matrix $\Amat$, $m \times n$. Not modified.
 * @param b The right-hand side $\bvec$, $m$ values. Not modified.
 * @param L The regularisation matrix $\Lmat$, $p \times n$. Not modified.
 * @param where The caller's name for error messages.
 * @returns The standard-form problem and the map back, as `StandardForm`.
 */
export function standardForm(A: MatrixLike, b: VectorLike, L: MatrixLike, where: string): StandardForm {
  const { data: a, m, n } = dense.toMatrixF64(A, where)
  const bb = dense.toF64(b, where)
  if (bb.length !== m) throw new ShapeError(where, `${where}: A has ${m} rows but b has ${bb.length} entries`)
  const { data: l, m: p, n: q } = dense.toMatrixF64(L, where)
  if (q !== n) throw new ShapeError(where, `${where}: A has ${n} columns but L has ${q}`)
  const eps = Number.EPSILON
  // L = U diag(sigma) V^T, thin: V is n x k.
  const ls = svd(dense.mat(l, p, n))
  const sigma = dense.data(ls.S)
  const V = dense.data(ls.V)
  const k = sigma.length
  const r = sigma.filter((v) => v > Math.max(p, n) * eps * sigma[0]).length
  if (r === 0) throw new DomainError(where, `${where}: L is zero`)
  // M = V1 D^-1 (n x r), so that x = M y on the range of L^T.
  const M = new Float64Array(n * r)
  for (let i = 0; i < n; i++) for (let c = 0; c < r; c++) M[i * r + c] = V[i * k + c] / sigma[c]
  const AM = dense.matMul(a, M, m, n, r)
  const K = M
  const x0 = new Float64Array(n)
  let Abar = AM
  let bbar = bb
  const w = n - r
  if (w > 0) {
    // W: the columns of a complete QR of V1 beyond the r-th, an orthonormal basis of the null space of L.
    const V1 = new Float64Array(n * r)
    for (let i = 0; i < n; i++) for (let c = 0; c < r; c++) V1[i * r + c] = V[i * k + c]
    const Q = dense.data(qr(dense.mat(V1, n, r), { mode: 'complete' }).Q)
    const W = new Float64Array(n * w)
    for (let i = 0; i < n; i++) for (let c = 0; c < w; c++) W[i * w + c] = Q[i * n + r + c]
    // AW = Uw diag(sw) Vw^T; G = W (AW)^+ maps a residual to the unpenalised part.
    const aw = svd(dense.mat(dense.matMul(a, W, m, n, w), m, w))
    const sw = dense.data(aw.S)
    const kw = sw.length
    const rw = sw.filter((v) => v > Math.max(m, w) * eps * sw[0]).length
    const Uw = dense.data(aw.U)
    const Vw = dense.data(aw.V)
    // project(v, cols): v minus its part in the range of AW, for an m x cols block.
    const project = (v: F64, cols: number): F64 => {
      const out = Float64Array.from(v)
      for (let t = 0; t < rw; t++)
        for (let c = 0; c < cols; c++) {
          let dotted = 0
          for (let i = 0; i < m; i++) dotted += Uw[i * kw + t] * v[i * cols + c]
          for (let i = 0; i < m; i++) out[i * cols + c] -= dotted * Uw[i * kw + t]
        }
      return out
    }
    // coefficients(v, cols): Vw diag(1/sw) Uw^T v, the least-squares z with AW z ~ v, then W z.
    const solveNull = (v: F64, cols: number): F64 => {
      const z = new Float64Array(w * cols)
      for (let t = 0; t < rw; t++)
        for (let c = 0; c < cols; c++) {
          let dotted = 0
          for (let i = 0; i < m; i++) dotted += Uw[i * kw + t] * v[i * cols + c]
          for (let j = 0; j < w; j++) z[j * cols + c] += (Vw[j * kw + t] * dotted) / sw[t]
        }
      return dense.matMul(W, z, n, w, cols)
    }
    Abar = project(AM, r)
    bbar = project(bb, 1)
    const GAM = solveNull(AM, r)
    for (let i = 0; i < n * r; i++) K[i] -= GAM[i]
    x0.set(solveNull(bb, 1))
  }
  return { sp: spectral(dense.mat(Abar, m, r), dense.vec(bbar), where), K, x0, n }
}

/**
 * The standard-form solution for one weight.
 *
 * @param sp The decomposition of $\Amat$ with $\bvec$, as `spectral` returns it.
 * @param lambda The penalty weight $\lambda \ge 0$.
 * @returns The solution `x`, $\lVert \Amat\xvec - \bvec \rVert^2$ as `rho`, $\lVert \xvec \rVert^2$ as `eta`,
 *   and the filter factors `f`.
 */
function standardSolution(sp: Spectral, lambda: number): { x: F64; rho: number; eta: number; f: F64 } {
  const f = sp.s.map((si) => factor(si, lambda, sp.cutoff))
  return { ...filtered(sp, f), f }
}

/**
 * The general-form solution for one weight: the standard-form filter in $\yvec$, mapped back to $\xvec$.
 *
 * @param form The standard-form problem, as `standardForm` returns it.
 * @param lambda The penalty weight $\lambda \ge 0$.
 * @returns The solution `x` ($n$ values), $\lVert \Amat\xvec - \bvec \rVert^2$ as `rho`,
 *   $\lVert \Lmat\xvec \rVert^2$ as `eta`, and the filter factors `f`.
 */
function generalSolution(form: StandardForm, lambda: number): { x: F64; rho: number; eta: number; f: F64 } {
  const { sp, K, x0, n } = form
  const f = sp.s.map((si) => factor(si, lambda, sp.cutoff))
  const { x: y, rho, eta } = filtered(sp, f)
  const x = dense.matVec(K, y, n, sp.n)
  for (let i = 0; i < n; i++) x[i] += x0[i]
  return { x, rho, eta, f }
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
 * by a QR factorisation (the minimum-norm one, by the SVD, when $\Amat$ and $\Lmat$ share a null vector); for many
 * weights with the same $\Lmat$, `tikhonovPath` with `L` is cheaper. At $\lambda = 0$ it is the minimum-norm
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
  const x = stackedSolve(stacked, m + p, n, bb)
  return {
    x: dense.vec(x),
    residualNorm: dense.norm(dense.sub(dense.matVec(a, x, m, n), bb)),
    solutionNorm: dense.norm(dense.matVec(l, x, p, n)),
    filterFactors: null,
    singularValues: null,
  }
}

/**
 * The least-squares solution of the stacked system $\begin{bmatrix}\Amat \\ \sqrt{\lambda}\Lmat\end{bmatrix}\xvec
 * \approx \begin{bmatrix}\bvec \\ \zeros\end{bmatrix}$ by a Householder QR factorisation and back substitution,
 * or, when the stacked matrix is wide or its triangular factor has a negligible pivot (rank deficiency), the
 * minimum-norm solution by `lstsq`.
 *
 * @param stacked The stacked matrix, `rows` $\times n$ row-major. Not modified.
 * @param rows Its number of rows, $m + p$.
 * @param n Its number of columns.
 * @param b The first $m$ entries of the right-hand side; the rest are zero.
 * @returns The solution $\xvec$, $n$ values.
 */
function stackedSolve(stacked: F64, rows: number, n: number, b: F64): F64 {
  const rhs = new Float64Array(rows)
  rhs.set(b)
  const fallback = () => dense.data(lstsq(dense.mat(stacked, rows, n), dense.vec(rhs)).x)
  if (rows < n) return fallback()
  const f = qr(dense.mat(stacked, rows, n))
  const Q = dense.data(f.Q)
  const R = dense.data(f.R)
  let largest = 0
  for (let i = 0; i < n; i++) largest = Math.max(largest, Math.abs(R[i * n + i]))
  const tol = rows * Number.EPSILON * largest
  for (let i = 0; i < n; i++) if (!(Math.abs(R[i * n + i]) > tol)) return fallback()
  const c = dense.matTVec(Q, rhs, rows, n)
  const x = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    let v = c[i]
    for (let j = i + 1; j < n; j++) v -= R[i * n + j] * x[j]
    x[i] = v / R[i * n + i]
  }
  return x
}

/** The result of `tikhonovPath`. */
export type TikhonovPath = {
  /** The penalty weights $\lambda_j$, as given ($r$ values). */
  lambdas: Tensor
  /** The solutions, $r \times n$: row $j$ is $\xvec_{\lambda_j}$. */
  x: Tensor
  /** $\lVert \Amat\xvec_{\lambda_j} - \bvec \rVert$, $r$ values. */
  residualNorm: Tensor
  /** $\lVert \Lmat\xvec_{\lambda_j} \rVert$ ($\lVert \xvec_{\lambda_j} \rVert$ in standard form), $r$ values. */
  solutionNorm: Tensor
  /**
   * The filter factors, $r \times k$: row $j$ holds $s_i^2/(s_i^2 + \lambda_j)$ ($\gamma_i^2/(\gamma_i^2 + \lambda_j)$
   * with a general $\Lmat$).
   */
  filterFactors: Tensor
  /**
   * In standard form the singular values $s_i$ of $\Amat$, $k = \min(m, n)$ values; with a general $\Lmat$ of rank
   * $r$ the generalised singular values $\gamma_i$ of $(\Amat, \Lmat)$, the singular values of the standard-form
   * $\bar\Amat$, $k = \min(m, r)$ values. Descending.
   */
  singularValues: Tensor
}

/** Options of `tikhonovPath`. */
export type TikhonovPathOptions = {
  /**
   * The regularisation matrix $\Lmat$, $p \times n$ of any rank above 0 (default $\Imat$, standard form): the
   * problem is brought to standard form once (Eldén's transformation) and every weight costs a filter.
   */
  L?: MatrixLike
}

/**
 * Tikhonov solutions for many penalty weights from one decomposition: the solutions, their residual and solution
 * norms and their filter factors, as a figure of the regularisation path (or the ridge path) needs. In standard form
 * that is one SVD of $\Amat$; with a general $\Lmat$ the problem is first brought to standard form in
 * $\yvec = \Lmat\xvec$ (Eldén, 1982; see the top of this file), after which each weight costs a filter and a
 * matrix-vector product, and the factors are those of the generalised singular values. The solutions agree with
 * `tikhonov`'s. Throws `DomainError` for a negative or non-finite weight or $\Lmat = \zeros$, and `ShapeError` for
 * mismatched shapes.
 *
 * @param A The matrix $\Amat$, $m \times n$. Not modified.
 * @param b The right-hand side $\bvec$, $m$ values. Not modified.
 * @param lambdas The penalty weights $\lambda_j \ge 0$, in any order ($r$ values).
 * @param options The regularisation matrix `L`, as `TikhonovPathOptions`.
 * @returns The path, one row or entry per weight in the order given, as `TikhonovPath`.
 *
 * @example Residual against solution norm as the weight grows
 * const A = tensor([[3, 0], [0, 0.1], [0, 0]])
 * const b = tensor([3, 0.2, 0.1])
 * const path = tikhonovPath(A, b, [0, 0.01, 1, 100])
 * print('x =', path.x)
 * print('residual norms =', path.residualNorm)
 * print('solution norms =', path.solutionNorm)
 *
 * @example A roughness penalty on a blurred step: large weights leave a straight line
 * const A = convolutionMatrix([0.25, 0.5, 0.25], 5)
 * const b = tensor([0, 0.25, 0.75, 1, 0.75])
 * const path = tikhonovPath(A, b, [1e-4, 0.1, 1e3], { L: differenceOperator(5, 2) })
 * print('x =', path.x)
 * print('roughness =', path.solutionNorm)
 * print('generalised singular values =', path.singularValues)
 */
export function tikhonovPath(
  A: MatrixLike,
  b: VectorLike,
  lambdas: VectorLike,
  options: TikhonovPathOptions = {},
): TikhonovPath {
  const ls = dense.toF64(lambdas, 'tikhonovPath')
  ls.forEach((l) => checkLambda(l, 'tikhonovPath'))
  const form: StandardForm =
    options.L === undefined
      ? { sp: spectral(A, b, 'tikhonovPath'), K: new Float64Array(0), x0: new Float64Array(0), n: -1 }
      : standardForm(A, b, options.L, 'tikhonovPath')
  const { sp } = form
  const n = form.n < 0 ? sp.n : form.n
  const r = ls.length
  const xs = new Float64Array(r * n)
  const fs = new Float64Array(r * sp.k)
  const residualNorm = new Float64Array(r)
  const solutionNorm = new Float64Array(r)
  for (let j = 0; j < r; j++) {
    const { x, rho, eta, f } = form.n < 0 ? standardSolution(sp, ls[j]) : generalSolution(form, ls[j])
    xs.set(x, j * n)
    fs.set(f, j * sp.k)
    residualNorm[j] = Math.sqrt(rho)
    solutionNorm[j] = Math.sqrt(eta)
  }
  return {
    lambdas: dense.vec(ls),
    x: dense.mat(xs, r, n),
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

/**
 * The $n \times n$ matrix of a one-dimensional convolution with a kernel of $k$ taps, so that $\Kmat\xvec$ is the
 * blurred signal: $(\Kmat\xvec)_i = \sum_t w_t\, x_{i + c - t}$ with the centre $c = \lfloor k/2 \rfloor$, as
 * scipy's `ndimage.convolve` (a symmetric kernel, such as a Gaussian, gives the same as correlation). Samples beyond
 * the ends are zero (`zero`, the blur cut off at the ends), wrap round (`periodic`, a circulant matrix) or mirror the
 * signal about its ends with the edge sample repeated (`reflect`, $x_{-1} = x_0$; Hansen's reflexive boundary, scipy's
 * `reflect`). As $\Amat$ in `tikhonov` it poses a deblurring problem. Throws `DomainError` for no taps, a non-finite
 * tap or $n < 1$.
 *
 * @param kernel The taps $w_0, \dots, w_{k-1}$, at offsets $c, c - 1, \dots, c - k + 1$ from the output sample.
 * @param n The length $n$ of the signal.
 * @param options How samples beyond the ends are taken.
 * @param options.boundary `zero` (default), `periodic` or `reflect`.
 * @returns The matrix $\Kmat$, $n \times n$.
 *
 * @example A three-tap blur on five samples, cut off and wrapped round
 * print('zero =', convolutionMatrix([0.25, 0.5, 0.25], 5))
 * print('periodic =', convolutionMatrix([0.25, 0.5, 0.25], 5, { boundary: 'periodic' }))
 *
 * @example Blurring a step with reflected ends keeps the mean
 * const K = convolutionMatrix([0.25, 0.5, 0.25], 6, { boundary: 'reflect' })
 * print('blurred step =', matmul(K, tensor([0, 0, 0, 1, 1, 1])))
 */
export function convolutionMatrix(
  kernel: VectorLike,
  n: number,
  { boundary = 'zero' }: { boundary?: 'zero' | 'periodic' | 'reflect' } = {},
): Tensor {
  const where = 'convolutionMatrix'
  const w = dense.toF64(kernel, where)
  if (w.length === 0 || !w.every(Number.isFinite))
    throw new DomainError(where, `${where}: the kernel needs at least one finite tap`)
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError(where, `${where}: n must be a positive integer, got ${n}`)
  if (boundary !== 'zero' && boundary !== 'periodic' && boundary !== 'reflect')
    throw new DomainError(where, `${where}: boundary must be 'zero', 'periodic' or 'reflect', got ${boundary}`)
  const c = Math.floor(w.length / 2)
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let t = 0; t < w.length; t++) {
      let j = i + c - t
      if (j < 0 || j >= n) {
        if (boundary === 'zero') continue
        if (boundary === 'periodic') j = ((j % n) + n) % n
        else {
          // Mirror with the edge repeated: period 2n, the second half reversed.
          j = ((j % (2 * n)) + 2 * n) % (2 * n)
          if (j >= n) j = 2 * n - 1 - j
        }
      }
      out[i * n + j] += w[t]
    }
  return dense.mat(out, n, n)
}
