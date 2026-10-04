/**
 * B-spline bases and penalised regression splines: the Cox–de Boor basis and its derivatives on any knot vector,
 * knot constructions, least-squares splines, difference and derivative penalties, P-splines (Eilers and Marx, 1996,
 * "Flexible smoothing with B-splines and penalties", Statistical Science 11(2)), cyclic bases and tensor products
 * (Wood, 2017, "Generalized Additive Models", 2nd ed., §5.3–5.6).
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { cholesky, choleskySolve, inverse, kron, lstsq } from 'aifn-compute/numerics/linalg'
import { gaussLegendre } from 'aifn-compute/numerics/quadrature'
import { minimizeScalar } from 'aifn-compute/numerics/roots'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

type F64 = Float64Array

/**
 * Convert a tensor to a flat Float64Array.
 *
 * @param t - Input tensor.
 * @returns 64-bit float array containing the flattened tensor data.
 */
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

// ── Basis ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The $p + 1$ B-splines of degree $p$ that are non-zero on knot span $i$, $N_{i-p}, \dots, N_i$ at $x$,
 * computed by the triangular recurrence of de Boor (Piegl and Tiller, 1997, Algorithm A2.2).
 * Outside $[t_i, t_{i+1})$ the same polynomial pieces are continued.
 *
 * @param t - Non-decreasing knot vector.
 * @param i - Knot span index containing $x$.
 * @param p - Spline degree.
 * @param x - Evaluation coordinate.
 * @returns Array of length $p + 1$ with the non-zero basis values at $x$.
 */
function spanBasis(t: F64, i: number, p: number, x: number): F64 {
  const N = new Float64Array(p + 1)
  const left = new Float64Array(p + 1)
  const right = new Float64Array(p + 1)
  N[0] = 1
  for (let j = 1; j <= p; j++) {
    left[j] = x - t[i + 1 - j]
    right[j] = t[i + j] - x
    let saved = 0
    for (let r = 0; r < j; r++) {
      const temp = N[r] / (right[r + 1] + left[j - r])
      N[r] = saved + right[r + 1] * temp
      saved = left[j - r] * temp
    }
    N[j] = saved
  }
  return N
}

/**
 * The knot span index: the largest $i \in [p, N - 1]$ with $t_i \le x$ (clamped at both ends),
 * where $N$ is the number of B-splines.
 *
 * @param t - Knot vector.
 * @param p - Spline degree.
 * @param count - Total number of B-spline basis functions $N$.
 * @param x - Evaluation coordinate.
 * @returns Knot span index $i$ in $[p, N - 1]$.
 */
function spanOf(t: F64, p: number, count: number, x: number): number {
  let lo = p
  let hi = count - 1
  if (!(x >= t[lo])) return lo
  if (x >= t[hi]) return hi
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (t[mid] <= x) lo = mid
    else hi = mid - 1
  }
  return lo
}

/**
 * Number of B-splines of degree $p$ on a knot vector of length $m$: $m - p - 1$.
 *
 * @param knots - Knot vector tensor of length $m$.
 * @param degree - Spline degree $p$.
 * @returns Number of basis functions $N = m - p - 1$.
 * @example Basis function count
 * const knots = uniformKnots(0, 1, 5, 3)
 * const count = bsplineCount(knots, 3)
 * print('count =', count)
 */
export function bsplineCount(knots: Tensor, degree: Size): Size {
  return knots.shape[0] - degree - 1
}

/**
 * How a basis treats coordinates $x$ outside its base interval $[t_p, t_N]$.
 *
 * - `'continue'`: Continues the polynomial pieces of the end spans.
 * - `'zero'`: Returns 0 for points outside $[t_p, t_N]$.
 * - `'clamp'`: Clamps coordinates to the boundary before evaluation.
 */
export type Extrapolation = 'continue' | 'zero' | 'clamp'

/**
 * The B-spline basis matrix of shape $[n, N]$ (or its $d$-th derivative) of degree $p$ on a non-decreasing knot
 * vector $\mathbf{t}$ of length $N + p + 1$ evaluated at $\mathbf{x}$ of length $n$, by the Cox–de Boor recurrence
 * (de Boor, 1978, ch. IX) and the derivative relation:
 * $$
 * B'_{i,p}(x) = p \left( \frac{B_{i,p-1}(x)}{t_{i+p} - t_i} - \frac{B_{i+1,p-1}(x)}{t_{i+p+1} - t_{i+1}} \right)
 * $$
 *
 * Outside $[t_p, t_N]$, `extrapolation` continues the end polynomials (`'continue'`, default, matching SciPy's
 * `BSpline`), evaluates to zero (`'zero'`), or clamps coordinates to the boundary (`'clamp'`). At $x = t_N$ the
 * last span is closed, so the basis covers the right endpoint.
 *
 * @param x - Evaluation points tensor of length $n$.
 * @param knots - Knot vector tensor of length $N + p + 1$.
 * @param degree - Spline degree $p \ge 0$.
 * @param options - Basis options.
 * @param options.derivative - Order of derivative to evaluate (default 0).
 * @param options.extrapolation - Extrapolation mode outside $[t_p, t_N]$ (default `'continue'`).
 * @returns Basis matrix tensor of shape $[n, N]$.
 * @example Evaluating cubic B-spline basis
 * const knots = uniformKnots(0, 1, 4, 3)
 * const x = tensor([0.2, 0.5, 0.8])
 * const B = bsplineBasis(x, knots, 3)
 * print('basis shape =', B.shape)
 */
export function bsplineBasis(
  x: Tensor,
  knots: Tensor,
  degree: number,
  { derivative = 0, extrapolation = 'continue' }: { derivative?: number; extrapolation?: Extrapolation } = {},
): Tensor {
  const t = f64(knots)
  const p = degree
  const count = t.length - p - 1
  if (count < 1) throw new ShapeError('bsplineBasis', `bsplineBasis: ${t.length} knots are too few for degree ${p}`)
  for (let k = 1; k < t.length; k++)
    if (t[k] < t[k - 1]) throw new DomainError('bsplineBasis', 'bsplineBasis: knots must be non-decreasing')
  const xs = f64(x)
  const n = xs.length
  const out = new Float64Array(n * count)
  const lo = t[p]
  const hi = t[count]
  const q = p - derivative
  for (let r = 0; r < n; r++) {
    let xi = xs[r]
    if (xi < lo || xi > hi) {
      if (extrapolation === 'zero') continue
      if (extrapolation === 'clamp') xi = xi < lo ? lo : hi
    }
    const span = spanOf(t, p, count, xi)
    if (q < 0) continue
    // Values of the degree-q B-splines that are non-zero on this span, indices span − q … span in degree-q indexing.
    const local = spanBasis(t, span, q, xi)
    let row = new Float64Array(t.length - q - 1)
    for (let a = 0; a <= q; a++) row[span - q + a] = local[a]
    // Raise to degree p one derivative at a time.
    for (let deg = q + 1; deg <= p; deg++) {
      const next = new Float64Array(t.length - deg - 1)
      for (let j = 0; j < next.length; j++) {
        const a = t[j + deg] - t[j]
        const b = t[j + deg + 1] - t[j + 1]
        next[j] = deg * ((a > 0 ? row[j] / a : 0) - (b > 0 ? row[j + 1] / b : 0))
      }
      row = next
    }
    out.set(row, r * count)
  }
  return fromData(out, [n, count])
}

/**
 * Construct equally spaced knots for P-splines: `segments` intervals on $[lo, hi]$ with spacing
 * $h = (hi - lo) / \text{segments}$, extended by `degree` knots beyond each end (Eilers and Marx, 1996),
 * giving $\text{segments} + \text{degree}$ B-splines that form a partition of unity on $[lo, hi]$.
 *
 * @param lo - Lower bound of the interval.
 * @param hi - Upper bound of the interval ($hi > lo$).
 * @param segments - Number of equal intervals $\ge 1$.
 * @param degree - Spline degree (default 3).
 * @returns Knot vector tensor of length $\text{segments} + 2 \cdot \text{degree} + 1$.
 * @example Generating uniform knots
 * const knots = uniformKnots(0, 1, 4, 3)
 * print('knots length =', knots.shape[0])
 */
export function uniformKnots(lo: Scalar, hi: Scalar, segments: Size, degree: Size = 3): Tensor {
  if (!(hi > lo) || !(segments >= 1))
    throw new DomainError('uniformKnots', 'uniformKnots: needs hi > lo and at least one segment')
  const h = (hi - lo) / segments
  return fromData(
    Float64Array.from({ length: segments + 2 * degree + 1 }, (_, i) => lo + (i - degree) * h),
    [segments + 2 * degree + 1],
  )
}

/**
 * Construct a clamped (open) knot vector from breakpoints $b_0 < \dots < b_m$: each endpoint is repeated
 * $\text{degree} + 1$ times, ensuring the spline interpolates its endpoint coefficients; produces $m + \text{degree}$ B-splines.
 *
 * @param breaks - Strict non-decreasing breakpoints tensor of length $m + 1$.
 * @param degree - Spline degree $p$ (default 3).
 * @returns Clamped knot vector tensor of length $m + 2 \cdot \text{degree} + 1$.
 * @example Creating clamped knots
 * const breaks = tensor([0, 0.5, 1.0])
 * const knots = clampedKnots(breaks, 3)
 * print('clamped knots length =', knots.shape[0])
 */
export function clampedKnots(breaks: Tensor, degree: Size = 3): Tensor {
  const b = f64(breaks)
  const m = b.length - 1
  const out = [...Array(degree).fill(b[0]), ...b, ...Array(degree).fill(b[m])]
  return fromData(Float64Array.from(out), [out.length])
}

/**
 * A spline in B-spline form:
 * $$
 * f(x) = \sum_{j=0}^{N-1} c_j B_j(x)
 * $$
 */
export type BSpline = {
  /** Discriminator identifying this as a B-spline representation. */
  readonly kind: 'bspline'
  /** Non-decreasing knot vector tensor. */
  readonly knots: Tensor
  /** Spline polynomial degree. */
  readonly degree: number
  /** Spline basis coefficients tensor of length $N$. */
  readonly coefficients: Tensor
  /**
   * Evaluate the spline (or its derivative) at coordinates $\mathbf{x}$.
   *
   * @param x - Evaluation points tensor of length $n$.
   * @param derivative - Order of derivative to evaluate (default 0).
   * @returns Evaluated spline values of shape $[n]$.
   */
  evaluate(x: Tensor, derivative?: number): Tensor
}

/**
 * Construct a spline object $f(x) = \sum_j c_j B_j(x)$ with given knots, degree, and coefficients.
 *
 * @param knots - Knot vector tensor of length $N + p + 1$.
 * @param degree - Spline degree $p$.
 * @param coefficients - Coefficient vector tensor of length $N$.
 * @returns B-spline object supporting evaluation and derivatives.
 * @example Constructing and evaluating a B-spline
 * const knots = uniformKnots(0, 1, 4, 3)
 * const c = tensor([1, 2, 1, 3, 2, 1, 0])
 * const s = bspline(knots, 3, c)
 * const y = s.evaluate(tensor([0.25, 0.75]))
 * print('spline values =', y)
 */
export function bspline(knots: Tensor, degree: Size, coefficients: Tensor): BSpline {
  const c = f64(coefficients)
  const count = bsplineCount(knots, degree)
  if (c.length !== count) throw new ShapeError('bspline', `bspline: ${count} B-splines but ${c.length} coefficients`)
  return {
    kind: 'bspline',
    knots,
    degree,
    coefficients,
    evaluate: (x, derivative = 0) => {
      const B = f64(bsplineBasis(x, knots, degree, { derivative }))
      const n = x.shape[0]
      const out = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < count; j++) s += B[i * count + j] * c[j]
        out[i] = s
      }
      return fromData(out, [n])
    },
  }
}

/**
 * Penalised least squares regression on a design matrix $\mathbf{B}$ of shape $[n, N]$:
 * minimise $\sum_{i=1}^n w_i (y_i - (\mathbf{B}\boldsymbol{\beta})_i)^2 + \boldsymbol{\beta}^\top \mathbf{P} \boldsymbol{\beta}$.
 *
 * @param B - Flat row-major basis matrix values of shape $[n, N]$.
 * @param n - Number of data observations.
 * @param N - Number of basis functions.
 * @param y - Response vector values of length $n$.
 * @param w - Observation weights of length $n$.
 * @param P - Penalty matrix values of shape $[N, N]$, or `null` for unpenalised least squares.
 * @returns Object containing estimated coefficients `beta`, system matrix `A`, and normal matrix `BtWB`.
 */
export function penalisedLeastSquares(
  B: F64,
  n: number,
  N: number,
  y: F64,
  w: F64,
  P: F64 | null,
): { beta: F64; A: F64; BtWB: F64 } {
  const BtWB = new Float64Array(N * N)
  const b = new Float64Array(N)
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < N; a++) {
      const v = B[i * N + a] * w[i]
      if (v === 0) continue
      b[a] += v * y[i]
      for (let c = 0; c < N; c++) BtWB[a * N + c] += v * B[i * N + c]
    }
  }
  const A = Float64Array.from(BtWB)
  if (P) for (let k = 0; k < N * N; k++) A[k] += P[k]
  const f = cholesky(fromData(A, [N, N]), { jitter: false })
  const beta = f.failed
    ? f64(lstsq(fromData(A, [N, N]), fromData(b, [N])).x)
    : f64(choleskySolve(f.L, fromData(b, [N])) as Tensor)
  return { beta, A, BtWB }
}

/**
 * Fit a least-squares spline with specified knots and degree to data points $(x_i, y_i)$,
 * with optional observation weights (equivalent to SciPy's `make_lsq_spline`).
 *
 * @param x - Predictor coordinates tensor of shape $[n]$.
 * @param y - Response values tensor of shape $[n]$.
 * @param knots - Knot vector tensor of length $N + p + 1$.
 * @param degree - Spline degree $p$ (default 3).
 * @param options - Fitting options.
 * @param options.weights - Optional positive weights tensor of shape $[n]$.
 * @returns Fitted B-spline object.
 * @example Fitting a least-squares spline
 * const x = tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
 * const y = tensor([0.0, 0.04, 0.16, 0.36, 0.64, 1.0])
 * const knots = uniformKnots(0, 1, 2, 3)
 * const fit = leastSquaresSpline(x, y, knots, 3)
 * const pred = fit.evaluate(tensor([0.5]))
 * print('prediction =', pred)
 */
export function leastSquaresSpline(
  x: Tensor,
  y: Tensor,
  knots: Tensor,
  degree = 3,
  { weights }: { weights?: Tensor } = {},
): BSpline {
  const n = x.shape[0]
  const N = bsplineCount(knots, degree)
  const B = f64(bsplineBasis(x, knots, degree))
  const w = weights ? f64(weights) : new Float64Array(n).fill(1)
  const { beta } = penalisedLeastSquares(B, n, N, f64(y), w, null)
  return bspline(knots, degree, fromData(beta, [N]))
}

// ── Penalties ────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Construct the order-$d$ difference matrix $\mathbf{D}$ of shape $[p - d, p]$, where
 * $(\mathbf{D}\boldsymbol{\beta})_j = \Delta^d \beta_j = \sum_{a=0}^d (-1)^{d-a} \binom{d}{a} \beta_{j+a}$.
 *
 * @param p - Dimension of the coefficient vector $\boldsymbol{\beta}$ ($p > d$).
 * @param order - Difference order $d \ge 0$.
 * @returns Difference matrix tensor of shape $[p - d, p]$.
 * @example Building a second-order difference matrix
 * const D = differenceMatrix(6, 2)
 * print('D shape =', D.shape)
 */
export function differenceMatrix(p: Size, order: Size): Tensor {
  if (!(order >= 0 && order < p)) throw new ShapeError('differenceMatrix', 'differenceMatrix: need 0 ≤ order < p')
  const c = new Float64Array(order + 1)
  let binom = 1
  for (let a = 0; a <= order; a++) {
    c[a] = ((order - a) % 2 === 0 ? 1 : -1) * binom
    binom = (binom * (order - a)) / (a + 1)
  }
  const rows = p - order
  const D = new Float64Array(rows * p)
  for (let r = 0; r < rows; r++) for (let a = 0; a <= order; a++) D[r * p + r + a] = c[a]
  return fromData(D, [rows, p])
}

/**
 * Construct the P-spline penalty matrix $\mathbf{D}^\top \mathbf{D}$ of shape $[p, p]$ from the order-$d$ difference matrix.
 *
 * @param p - Dimension of the coefficient vector ($p > d$).
 * @param order - Difference order $d \ge 0$.
 * @returns Penalty matrix tensor $\mathbf{D}^\top \mathbf{D}$ of shape $[p, p]$.
 * @example Constructing a difference penalty
 * const P = differencePenalty(6, 2)
 * print('P shape =', P.shape)
 */
export function differencePenalty(p: Size, order: Size): Tensor {
  return crossProduct(differenceMatrix(p, order))
}

/**
 * Construct the cyclic order-$d$ difference matrix of shape $[p, p]$, taking differences circularly around $[0, p-1]$.
 *
 * @param p - Dimension of the coefficient vector ($p > d$).
 * @param order - Difference order $d \ge 0$.
 * @returns Cyclic difference matrix tensor of shape $[p, p]$.
 * @example Creating a cyclic difference matrix
 * const D = cyclicDifferenceMatrix(6, 2)
 * print('cyclic D shape =', D.shape)
 */
export function cyclicDifferenceMatrix(p: Size, order: Size): Tensor {
  const D0 = f64(differenceMatrix(order + 1, order))
  const out = new Float64Array(p * p)
  for (let r = 0; r < p; r++) for (let a = 0; a <= order; a++) out[r * p + ((r + a) % p)] += D0[a]
  return fromData(out, [p, p])
}

/**
 * Construct the cyclic difference penalty matrix $\mathbf{D}^\top \mathbf{D}$ of shape $[p, p]$.
 *
 * @param p - Dimension of the coefficient vector ($p > d$).
 * @param order - Difference order $d \ge 0$.
 * @returns Cyclic penalty matrix tensor of shape $[p, p]$.
 * @example Creating a cyclic difference penalty
 * const P = cyclicDifferencePenalty(6, 2)
 * print('cyclic P shape =', P.shape)
 */
export function cyclicDifferencePenalty(p: Size, order: Size): Tensor {
  return crossProduct(cyclicDifferenceMatrix(p, order))
}

/**
 * Compute the Gram matrix $\mathbf{A}^\top \mathbf{A}$ for matrix $\mathbf{A}$.
 *
 * @param A - Input matrix tensor of shape $[m, p]$.
 * @returns Symmetric matrix tensor $\mathbf{A}^\top \mathbf{A}$ of shape $[p, p]$.
 */
function crossProduct(A: Tensor): Tensor {
  const [m, p] = A.shape
  const a = f64(A)
  const out = new Float64Array(p * p)
  for (let r = 0; r < m; r++)
    for (let i = 0; i < p; i++) {
      const v = a[r * p + i]
      if (v === 0) continue
      for (let j = 0; j < p; j++) out[i * p + j] += v * a[r * p + j]
    }
  return fromData(out, [p, p])
}

/**
 * Construct the exact derivative penalty matrix $\mathbf{S}$ of shape $[N, N]$ with entries
 * $S_{k,l} = \int_{lo}^{hi} B_k^{(m)}(x) B_l^{(m)}(x) \, dx$ (such that $\boldsymbol{\beta}^\top \mathbf{S} \boldsymbol{\beta} = \int [f^{(m)}(x)]^2 \, dx$),
 * evaluated via Gauss–Legendre quadrature with $\text{degree} + 1$ nodes per knot span (O'Sullivan, 1986; Wood, 2017, §5.3.3).
 *
 * @param knots - Knot vector tensor of length $N + p + 1$.
 * @param degree - Spline degree $p$.
 * @param m - Derivative order to penalise.
 * @param range - Integration interval $[lo, hi]$ (defaults to the base interval $[t_p, t_N]$).
 * @returns Penalty matrix tensor of shape $[N, N]$.
 * @example Computing a derivative penalty matrix
 * const knots = uniformKnots(0, 1, 4, 3)
 * const S = derivativePenalty(knots, 3, 2)
 * print('S shape =', S.shape)
 */
export function derivativePenalty(knots: Tensor, degree: Size, m: Size, range?: readonly [Scalar, Scalar]): Tensor {
  const t = f64(knots)
  const N = bsplineCount(knots, degree)
  const [lo, hi] = range ?? [t[degree], t[N]]
  const S = new Float64Array(N * N)
  const rule = gaussLegendre(degree + 1)
  const u = f64(rule.nodes)
  const w = f64(rule.weights)
  for (let s = 0; s < t.length - 1; s++) {
    const a = Math.max(t[s], lo)
    const b = Math.min(t[s + 1], hi)
    if (!(b > a)) continue
    const xs = Float64Array.from(u, (v) => ((b - a) * v + a + b) / 2)
    const D = f64(bsplineBasis(fromData(xs, [xs.length]), knots, degree, { derivative: m }))
    for (let g = 0; g < xs.length; g++) {
      const wg = (w[g] * (b - a)) / 2
      for (let k = 0; k < N; k++) {
        const dk = D[g * N + k]
        if (dk === 0) continue
        for (let l = 0; l < N; l++) S[k * N + l] += wg * dk * D[g * N + l]
      }
    }
  }
  return fromData(S, [N, N])
}

// ── Cyclic and tensor-product bases ──────────────────────────────────────────────────────────────────────────────

/**
 * Construct the cyclic B-spline basis of $k$ periodic functions of degree $p$ on $[lo, hi)$:
 * formed by wrapping the last $p$ functions of a $(k + p)$-function uniform B-spline basis onto the first $p$,
 * ensuring $f(lo) = f(hi)$ and matching derivatives up to order $p - 1$ (Eilers and Marx, 2010; Wood, 2017, §5.3.2).
 * Coordinates in $\mathbf{x}$ are reduced modulo the period $(hi - lo)$.
 *
 * @param x - Evaluation coordinates tensor.
 * @param lo - Lower bound of the periodic domain.
 * @param hi - Upper bound of the periodic domain.
 * @param k - Number of cyclic basis functions ($k > \text{degree}$).
 * @param degree - Spline degree $p$ (default 3).
 * @returns Basis matrix tensor of shape $[n, k]$.
 * @example Evaluating cyclic B-spline basis
 * const x = tensor([0.0, 0.25, 0.5, 0.75])
 * const B = cyclicBsplineBasis(x, 0, 1, 6, 3)
 * print('basis shape =', B.shape)
 */
export function cyclicBsplineBasis(x: Tensor, lo: Scalar, hi: Scalar, k: Size, degree: Size = 3): Tensor {
  if (k <= degree)
    throw new DomainError('cyclicBsplineBasis', 'cyclicBsplineBasis: needs more functions than the degree')
  const period = hi - lo
  const wrapped = Float64Array.from(toFlat(x), (v) => lo + ((((v - lo) % period) + period) % period))
  const knots = uniformKnots(lo, hi, k, degree)
  const B = f64(bsplineBasis(fromData(wrapped, [wrapped.length]), knots, degree))
  const n = wrapped.length
  const wide = k + degree
  const out = new Float64Array(n * k)
  for (let i = 0; i < n; i++) for (let j = 0; j < wide; j++) out[i * k + (j % k)] += B[i * wide + j]
  return fromData(out, [n, k])
}

/**
 * Compute the row-wise Kronecker product (tensor product) of two basis matrices $\mathbf{A}$ of shape $[n, p]$
 * and $\mathbf{B}$ of shape $[n, q]$, yielding a combined basis matrix of shape $[n, p \cdot q]$ with $\mathbf{B}$'s index
 * running fastest.
 *
 * @param A - First basis matrix tensor of shape $[n, p]$.
 * @param B - Second basis matrix tensor of shape $[n, q]$.
 * @returns Tensor-product basis matrix of shape $[n, p \cdot q]$.
 * @example Forming a tensor-product basis
 * const knots = uniformKnots(0, 1, 2, 2)
 * const x = tensor([0.2, 0.8])
 * const B1 = bsplineBasis(x, knots, 2)
 * const B2 = bsplineBasis(x, knots, 2)
 * const BTensor = tensorProductBasis(B1, B2)
 * print('BTensor shape =', BTensor.shape)
 */
export function tensorProductBasis(A: Tensor, B: Tensor): Tensor {
  const [n, p] = A.shape
  const [m, q] = B.shape
  if (n !== m) throw new ShapeError('tensorProductBasis', `tensorProductBasis: ${n} rows against ${m}`)
  const a = f64(A)
  const b = f64(B)
  const out = new Float64Array(n * p * q)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < p; j++) {
      const v = a[i * p + j]
      if (v === 0) continue
      for (let l = 0; l < q; l++) out[i * p * q + j * q + l] = v * b[i * q + l]
    }
  return fromData(out, [n, p * q])
}

/**
 * Construct the two marginal roughness penalty matrices for a tensor-product smooth from marginal penalties
 * $\mathbf{S}_A$ of shape $[p, p]$ and $\mathbf{S}_B$ of shape $[q, q]$:
 * $\mathbf{S}_A \otimes \mathbf{I}_q$ (roughness along the first coordinate) and
 * $\mathbf{I}_p \otimes \mathbf{S}_B$ (roughness along the second coordinate), each weighted by its own smoothing parameter $\lambda$
 * (Wood, 2006).
 *
 * @param SA - First marginal penalty matrix tensor of shape $[p, p]$.
 * @param SB - Second marginal penalty matrix tensor of shape $[q, q]$.
 * @returns Pair of penalty matrices $[\mathbf{S}_A \otimes \mathbf{I}_q, \mathbf{I}_p \otimes \mathbf{S}_B]$ each of shape $[p \cdot q, p \cdot q]$.
 * @example Generating tensor-product penalties
 * const P1 = differencePenalty(4, 2)
 * const P2 = differencePenalty(4, 2)
 * const [S1, S2] = tensorProductPenalties(P1, P2)
 * print('S1 shape =', S1.shape)
 */
export function tensorProductPenalties(SA: Tensor, SB: Tensor): [Tensor, Tensor] {
  const eye = (k: number) => {
    const e = new Float64Array(k * k)
    for (let i = 0; i < k; i++) e[i * k + i] = 1
    return fromData(e, [k, k])
  }
  return [kron(SA, eye(SB.shape[0])) as Tensor, kron(eye(SA.shape[0]), SB) as Tensor]
}

// ── P-splines ────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for configuring a P-spline smoother in `pspline`. */
export type PSplineOptions = {
  /** Number of equal intervals on $[lo, hi]$ (default 20), giving $\text{segments} + \text{degree}$ B-splines. */
  segments?: number
  /** Spline polynomial degree (default 3). */
  degree?: number
  /** Order of the difference penalty $\Delta^d$ (default 2). */
  order?: number
  /** Smoothing parameter $\lambda \ge 0$, or `'gcv'` to minimise generalised cross-validation (default 1). */
  lambda?: number | 'gcv'
  /** Domain interval $[lo, hi]$ for the knot sequence (defaults to the range of $\mathbf{x}$). */
  range?: readonly [number, number]
  /** Optional positive observation weights of shape $[n]$. */
  weights?: Tensor
}

/** A fitted P-spline smoothing model with inferential diagnostics. */
export type PSplineFit = BSpline & {
  /** Smoothing parameter $\lambda$ used for the fit. */
  readonly lambda: number
  /** Fitted values at the training data coordinates, shape $[n]$. */
  readonly fitted: Tensor
  /** Effective degrees of freedom $\trace(\mathbf{H})$, where $\mathbf{H} = \mathbf{B}(\mathbf{B}^\top \mathbf{W} \mathbf{B} + \lambda \mathbf{D}^\top \mathbf{D})^{-1}\mathbf{B}^\top \mathbf{W}$. */
  readonly edf: number
  /** Diagonal entries of the hat matrix $\mathbf{H}$ (leverages), shape $[n]$. */
  readonly leverage: Tensor
  /** Weighted residual sum of squares $\sum_i w_i (y_i - \hat{y}_i)^2$. */
  readonly rss: number
  /** Generalised cross-validation score $n \cdot \text{RSS} / (n - \text{edf})^2$ (Craven and Wahba, 1979). */
  readonly gcv: number
  /** Residual variance estimate $\hat{\sigma}^2 = \text{RSS} / (n - \text{edf})$. */
  readonly sigma2: number
  /** Bayesian posterior covariance matrix $\hat{\sigma}^2 (\mathbf{B}^\top \mathbf{W} \mathbf{B} + \lambda \mathbf{P})^{-1}$ of the basis coefficients (Wahba, 1983; Wood, 2017, §6.10). */
  readonly covariance: Tensor
  /**
   * Compute pointwise posterior standard errors of the fitted function at coordinates $\mathbf{x}$.
   *
   * @param x - Evaluation coordinates tensor of shape $[m]$.
   * @returns Pointwise standard errors tensor of shape $[m]$.
   */
  standardError(x: Tensor): Tensor
}

/**
 * Fit a P-spline smoother at a single specified smoothing parameter $\lambda$.
 *
 * @param X - Flattened predictor array.
 * @param y - Flattened response array.
 * @param w - Flattened weights array.
 * @param knots - Knot vector tensor.
 * @param degree - Spline degree.
 * @param P - Flat penalty matrix values.
 * @param lambda - Smoothing parameter $\lambda \ge 0$.
 * @returns Fitted smoother properties and diagnostics without the BSpline evaluate method.
 */
function psplineAt(
  X: F64,
  y: F64,
  w: F64,
  knots: Tensor,
  degree: number,
  P: F64,
  lambda: number,
): Omit<PSplineFit, keyof BSpline> & { coefficients: Tensor } {
  const n = X.length
  const N = bsplineCount(knots, degree)
  const B = f64(bsplineBasis(fromData(X, [n]), knots, degree))
  const lp = Float64Array.from(P, (v) => v * lambda)
  const { beta, A } = penalisedLeastSquares(B, n, N, y, w, lp)
  const Ainv = f64(inverse(fromData(A, [N, N])) as Tensor)
  const leverage = new Float64Array(n)
  const fitted = new Float64Array(n)
  let rss = 0
  for (let i = 0; i < n; i++) {
    let f = 0
    let h = 0
    for (let a = 0; a < N; a++) {
      const ba = B[i * N + a]
      if (ba === 0) continue
      f += ba * beta[a]
      let s = 0
      for (let c = 0; c < N; c++) s += Ainv[a * N + c] * B[i * N + c]
      h += ba * s
    }
    fitted[i] = f
    leverage[i] = h * w[i]
    rss += w[i] * (y[i] - f) ** 2
  }
  const edf = leverage.reduce((a, b) => a + b, 0)
  const sigma2 = rss / (n - edf)
  const covariance = Float64Array.from(Ainv, (v) => v * sigma2)
  return {
    coefficients: fromData(beta, [N]),
    lambda,
    fitted: fromData(fitted, [n]),
    edf,
    leverage: fromData(leverage, [n]),
    rss,
    gcv: (n * rss) / (n - edf) ** 2,
    sigma2,
    covariance: fromData(covariance, [N, N]),
    standardError: (x) => {
      const Bx = f64(bsplineBasis(x, knots, degree))
      const m = x.shape[0]
      return fromData(
        Float64Array.from({ length: m }, (_, i) => {
          let s = 0
          for (let a = 0; a < N; a++) {
            const ba = Bx[i * N + a]
            if (ba === 0) continue
            for (let c = 0; c < N; c++) s += ba * covariance[a * N + c] * Bx[i * N + c]
          }
          return Math.sqrt(Math.max(s, 0))
        }),
        [m],
      )
    },
  }
}

/**
 * Extract grid, knots, and penalty structures for P-spline fitting.
 *
 * @param x - Predictor coordinates tensor.
 * @param options - P-spline configuration options.
 * @returns Precomputed arrays, knots, degree, penalty matrix, and weights.
 */
function psplineSetup(x: Tensor, options: PSplineOptions) {
  const { segments = 20, degree = 3, order = 2 } = options
  const X = f64(x)
  const lo = options.range?.[0] ?? Math.min(...X)
  const hi = options.range?.[1] ?? Math.max(...X)
  const knots = uniformKnots(lo, hi, segments, degree)
  const N = bsplineCount(knots, degree)
  const P = f64(differencePenalty(N, order))
  const w = options.weights ? f64(options.weights) : new Float64Array(X.length).fill(1)
  return { X, knots, degree, P, w }
}

/**
 * Compute the generalised cross-validation (GCV) score and effective degrees of freedom across a grid
 * of $\log_{10} \lambda$ values in `logLambdas` for a P-spline model, useful for plotting the GCV profile curve.
 *
 * @param x - Predictor coordinates tensor of shape $[n]$.
 * @param y - Response values tensor of shape $[n]$.
 * @param logLambdas - Tensor containing base-10 logarithms of smoothing parameters $\log_{10} \lambda$.
 * @param options - P-spline options excluding `lambda`.
 * @returns Object with tensors `gcv` and `edf` evaluated at each parameter in `logLambdas`.
 * @example Evaluating a GCV profile path
 * const x = tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
 * const y = tensor([0.1, 0.3, 0.5, 0.7, 0.8, 1.1])
 * const lams = tensor([-2, -1, 0, 1, 2])
 * const path = psplineGcvPath(x, y, lams, { segments: 5 })
 * print('gcv length =', path.gcv.shape[0])
 */
export function psplineGcvPath(x: Tensor, y: Tensor, logLambdas: Tensor, options: Omit<PSplineOptions, 'lambda'> = {}) {
  const s = psplineSetup(x, options)
  const Y = f64(y)
  const ls = f64(logLambdas)
  const gcv = new Float64Array(ls.length)
  const edf = new Float64Array(ls.length)
  ls.forEach((l, i) => {
    const f = psplineAt(s.X, Y, s.w, s.knots, s.degree, s.P, 10 ** l)
    gcv[i] = f.gcv
    edf[i] = f.edf
  })
  return { gcv: fromData(gcv, [ls.length]), edf: fromData(edf, [ls.length]) }
}

/**
 * Fit a P-spline regression smoother (Eilers and Marx, 1996): equally spaced B-splines with a difference penalty
 * on neighbouring coefficients, minimising $\sum_{i=1}^n w_i (y_i - f(x_i))^2 + \lambda \|\mathbf{D}\boldsymbol{\beta}\|^2$.
 * When `lambda: 'gcv'` is specified, $\lambda$ is automatically selected by minimising the GCV score over
 * $\log_{10} \lambda \in [-6, 6]$ via grid evaluation followed by golden-section search refinement.
 *
 * @param x - Predictor coordinates tensor of shape $[n]$.
 * @param y - Response values tensor of shape $[n]$.
 * @param options - P-spline smoothing options.
 * @returns Fitted P-spline model object with evaluation and inferential diagnostics.
 * @example Smoothing noisy observations with a P-spline
 * const x = tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
 * const y = tensor([0.05, 0.18, 0.42, 0.61, 0.79, 1.02])
 * const fit = pspline(x, y, { segments: 5, lambda: 0.1 })
 * const yHat = fit.evaluate(tensor([0.5]))
 * print('fitted value =', yHat)
 */
export function pspline(x: Tensor, y: Tensor, options: PSplineOptions = {}): PSplineFit {
  const s = psplineSetup(x, options)
  const Y = f64(y)
  if (Y.length !== s.X.length)
    throw new ShapeError('pspline', `pspline: ${s.X.length} x values but ${Y.length} y values`)
  let lambda = options.lambda ?? 1
  if (lambda === 'gcv') {
    const score = (l: number) => psplineAt(s.X, Y, s.w, s.knots, s.degree, s.P, 10 ** l).gcv
    let best = -6
    let bestScore = Infinity
    for (let l = -6; l <= 6; l += 0.5) {
      const v = score(l)
      if (v < bestScore) [best, bestScore] = [l, v]
    }
    lambda = 10 ** minimizeScalar(score, { bounds: [best - 0.5, best + 0.5], method: 'golden' }).x
  }
  if (!(lambda >= 0)) throw new DomainError('pspline', 'pspline: λ must be ≥ 0')
  const fit = psplineAt(s.X, Y, s.w, s.knots, s.degree, s.P, lambda)
  return { ...bspline(s.knots, s.degree, fit.coefficients), ...fit }
}
