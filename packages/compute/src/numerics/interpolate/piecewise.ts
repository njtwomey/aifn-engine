/**
 * Piecewise polynomials and the one-dimensional interpolants built on them: piecewise linear, cubic splines with four
 * end conditions, cubic Hermite, PCHIP, Akima, and the cubic smoothing spline. Pieces are stored in the local power
 * basis, as SciPy's `PPoly`: on $[b_i, b_{i+1}]$, $f(x) = \sum_k c_{ik} (x - b_i)^k$.
 *
 * References: de Boor (1978), "A Practical Guide to Splines", ch. IV; Fritsch and Carlson (1980) and Fritsch and
 * Butland (1984) for PCHIP; Akima (1970); Reinsch (1967) and Green and Silverman (1994), §2.3, for smoothing splines.
 * Each is checked against scipy.interpolate.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Scalar } from 'aifn-compute/foundation/contracts'
import { solve } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** A piecewise polynomial representation in the local power basis. */
export type PiecewisePolynomial = {
  /** Discriminator kind tag. */
  readonly kind: 'piecewise-polynomial'
  /** Breakpoints $b_0 < b_1 < \dots < b_m$, shape $[m + 1]$. */
  readonly breaks: Tensor
  /** Coefficients of shape $[m, \text{order}]$: row $i$ holds $c_{i0}, c_{i1}, \dots$ (ascending powers of $x - b_i$). */
  readonly coefficients: Tensor
}

type F64 = Float64Array

/**
 * Wrap numeric array data as a 1D tensor.
 *
 * @param a Input numeric array.
 * @returns 1D tensor wrapping the data.
 */
const vec = (a: ArrayLike<number>) => fromData(Float64Array.from(a), [a.length])

/**
 * Extract flat Float64Array view from tensor or array-like input.
 *
 * @param t Input tensor or numeric array.
 * @returns Flattened 64-bit float array.
 */
const f64 = (t: Tensor | ArrayLike<number>): F64 =>
  Float64Array.from('shape' in (t as Tensor) ? toFlat(t as Tensor) : (t as ArrayLike<number>))

/**
 * Construct a piecewise polynomial from breakpoint coordinates and coefficient rows.
 *
 * @param breaks Monotonically increasing breakpoint coordinates $[b_0, \dots, b_m]$ of length $m + 1$.
 * @param rows Array of $m$ rows, each containing ascending power polynomial coefficients for piece $i$.
 * @returns A `PiecewisePolynomial` object in local power basis.
 *
 * @example Construct piecewise linear polynomial
 * const pp = piecewisePolynomial([0, 1, 2], [[0, 1], [1, 2]])
 * print('breaks =', pp.breaks.shape)
 */
export function piecewisePolynomial(
  breaks: ArrayLike<number>,
  rows: readonly ArrayLike<number>[],
): PiecewisePolynomial {
  const order = Math.max(...rows.map((r) => r.length))
  const c = new Float64Array(rows.length * order)
  rows.forEach((r, i) => {
    for (let k = 0; k < r.length; k++) c[i * order + k] = r[k]
  })
  return { kind: 'piecewise-polynomial', breaks: vec(breaks), coefficients: fromData(c, [rows.length, order]) }
}

/**
 * Binary search for the interval piece index $[b_i, b_{i+1}]$ containing coordinate $x$.
 *
 * @param b Breakpoints array.
 * @param x Evaluation coordinate.
 * @returns Piece index $i \in \{0, \dots, m - 1\}$.
 */
function pieceOf(b: F64, x: number): number {
  const last = b.length - 2
  if (!(x > b[0])) return 0
  if (x >= b[last]) return last
  let lo = 0
  let hi = last
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (b[mid] <= x) lo = mid
    else hi = mid - 1
  }
  return lo
}

/**
 * Evaluate the piecewise polynomial (or its $r$-th derivative) at coordinates $x$.
 *
 * Outside the breakpoints, end pieces are continued when `extrapolate` is true (matching SciPy's
 * `PPoly(..., extrapolate=True)`). If `extrapolate` is false, values outside $[b_0, b_m]$ evaluate to NaN.
 *
 * @param pp Piecewise polynomial representation.
 * @param x Query coordinates tensor.
 * @param options Evaluation options specifying derivative order and extrapolation behaviour.
 * @param options.derivative Derivative order to evaluate (default 0).
 * @param options.extrapolate Whether to extrapolate outside the breakpoint range (default true).
 * @returns Evaluated polynomial or derivative tensor matching the shape of $x$.
 *
 * @example Evaluate piecewise polynomial
 * const pp = piecewisePolynomial([0, 1, 2], [[0, 1], [1, 2]])
 * const y = evaluatePiecewise(pp, tensor([0.5, 1.5]))
 * print('evaluated =', y)
 */
export function evaluatePiecewise(
  pp: PiecewisePolynomial,
  x: Tensor,
  { derivative = 0, extrapolate = true }: { derivative?: number; extrapolate?: boolean } = {},
): Tensor {
  const b = f64(pp.breaks)
  const [m, order] = pp.coefficients.shape
  const c = f64(pp.coefficients)
  const xs = f64(x)
  const out = new Float64Array(xs.length)
  for (let j = 0; j < xs.length; j++) {
    const t = xs[j]
    if (!extrapolate && (t < b[0] || t > b[m])) {
      out[j] = NaN
      continue
    }
    const i = pieceOf(b, t)
    const h = t - b[i]
    // Horner on the derivative's coefficients k!/(k − r)! cₖ.
    let s = 0
    for (let k = order - 1; k >= derivative; k--) {
      let f = 1
      for (let r = 0; r < derivative; r++) f *= k - r
      s = s * h + f * c[i * order + k]
    }
    out[j] = s
  }
  return fromData(out, x.shape)
}

/**
 * Definite integral $\int_a^b f(x)\,dx$ of a piecewise polynomial.
 *
 * Outside the breakpoints, the end polynomial pieces are integrated analytically.
 *
 * @param pp Piecewise polynomial to integrate.
 * @param a Lower integration limit.
 * @param b Upper integration limit.
 * @returns Exact integral value over $[a, b]$.
 *
 * @example Integrate piecewise polynomial
 * const pp = piecewisePolynomial([0, 1, 2], [[0, 1], [1, 1]])
 * const area = integratePiecewise(pp, 0, 2)
 * print('integral =', area)
 */
export function integratePiecewise(pp: PiecewisePolynomial, a: Scalar, b: Scalar): Scalar {
  if (a > b) return -integratePiecewise(pp, b, a)
  const br = f64(pp.breaks)
  const [m, order] = pp.coefficients.shape
  const c = f64(pp.coefficients)
  // Antiderivative of piece i from bᵢ to t.
  const F = (i: number, t: number) => {
    const h = t - br[i]
    let s = 0
    for (let k = order - 1; k >= 0; k--) s = s * h + c[i * order + k] / (k + 1)
    return s * h
  }
  let total = 0
  let t = a
  while (t < b) {
    const i = pieceOf(br, t)
    const end = i === m - 1 ? b : Math.min(b, br[i + 1])
    total += F(i, end) - F(i, t)
    if (end === t) break
    t = end
  }
  return total
}

/**
 * Validate that interpolation nodes are strictly increasing and match response array length.
 *
 * @param x Abscissa coordinates array.
 * @param y Function values array.
 * @param where Calling function name for descriptive error reporting.
 */
function checkData(x: F64, y: F64, where: string) {
  if (x.length !== y.length) throw new ShapeError(where, `${where}: ${x.length} x values but ${y.length} y values`)
  if (x.length < 2) throw new DomainError(where, `${where}: needs at least two points`)
  for (let i = 1; i < x.length; i++)
    if (!(x[i] > x[i - 1])) throw new DomainError(where, `${where}: x must be strictly increasing`)
}

/**
 * Construct the piecewise linear interpolant through distinct points $(x_i, y_i)$.
 *
 * @param x Strictly increasing node coordinates tensor of length $n \ge 2$.
 * @param y Observed values tensor of length $n$.
 * @returns Piecewise polynomial of order 2 (degree 1).
 *
 * @example Linear interpolation
 * const interp = linearInterpolant(tensor([0, 1, 2]), tensor([0, 1, 0]))
 * const y = evaluatePiecewise(interp, tensor([0.5, 1.5]))
 * print('y =', y)
 */
export function linearInterpolant(x: Tensor, y: Tensor): PiecewisePolynomial {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'linearInterpolant')
  return piecewisePolynomial(
    X,
    Array.from({ length: X.length - 1 }, (_, i) => [Y[i], (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])]),
  )
}

/**
 * Compute cubic Hermite polynomial pieces from data values and specified slopes.
 *
 * @param X Breakpoint coordinates array.
 * @param Y Response values array.
 * @param M First derivative slope values at each breakpoint.
 * @returns Piecewise cubic polynomial.
 */
function hermite(X: F64, Y: F64, M: ArrayLike<number>): PiecewisePolynomial {
  const rows = Array.from({ length: X.length - 1 }, (_, i) => {
    const h = X[i + 1] - X[i]
    const delta = (Y[i + 1] - Y[i]) / h
    return [Y[i], M[i], (3 * delta - 2 * M[i] - M[i + 1]) / h, (M[i] + M[i + 1] - 2 * delta) / (h * h)]
  })
  return piecewisePolynomial(X, rows)
}

/**
 * Construct the cubic Hermite interpolant with prescribed first derivative slopes at the data points.
 *
 * Matches SciPy's `CubicHermiteSpline(x, y, dydx)`.
 *
 * @param x Strictly increasing node coordinates tensor of length $n$.
 * @param y Observed function values tensor of length $n$.
 * @param slopes Prescribed first derivative values $y'_i$ at the nodes, length $n$.
 * @returns Piecewise cubic polynomial interpolant.
 *
 * @example Cubic Hermite interpolation
 * const x = tensor([0, 1, 2])
 * const y = tensor([0, 1, 0])
 * const slopes = tensor([1, 0, -1])
 * const spline = hermiteSpline(x, y, slopes)
 * print('evaluated =', evaluatePiecewise(spline, tensor([0.5])))
 */
export function hermiteSpline(x: Tensor, y: Tensor, slopes: Tensor): PiecewisePolynomial {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'hermiteSpline')
  return hermite(X, Y, f64(slopes))
}

/**
 * Construct piecewise cubic polynomial from data values and second derivatives (moments) $M_i$.
 *
 * @param X Node coordinates array.
 * @param Y Response values array.
 * @param M Second derivative moments $M_i = f''(x_i)$ at each breakpoint.
 * @returns Piecewise cubic polynomial.
 */
function fromMoments(X: F64, Y: ArrayLike<number>, M: ArrayLike<number>): PiecewisePolynomial {
  const rows = Array.from({ length: X.length - 1 }, (_, i) => {
    const h = X[i + 1] - X[i]
    const b = (Y[i + 1] - Y[i]) / h - (h * (2 * M[i] + M[i + 1])) / 6
    return [Y[i], b, M[i] / 2, (M[i + 1] - M[i]) / (6 * h)]
  })
  return piecewisePolynomial(X, rows)
}

/** End conditions configuring boundary constraints of a cubic spline (SciPy's `bc_type`). */
export type EndCondition =
  'not-a-knot' | 'natural' | 'clamped' | 'periodic' | { first: [number, number] } | { second: [number, number] }

/**
 * Solve a small dense linear system $A x = r$ of dimension $n \times n$.
 *
 * @param A Coefficient matrix in row-major order.
 * @param r Right-hand side vector.
 * @param n Dimension of the square system.
 * @returns Solution vector as a Float64Array.
 */
function denseSolve(A: F64, r: F64, n: number): F64 {
  return f64(solve(fromData(A, [n, n]), fromData(r, [n])) as Tensor)
}

/**
 * Construct the $C^2$ interpolating cubic spline through $(x_i, y_i)$.
 *
 * Solves the tridiagonal system for second derivatives $M_i$:
 * $h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i + h_i M_{i+1} = 6(\delta_i - \delta_{i-1})$
 * (de Boor, 1978, ch. IV). Supports standard boundary conditions: `'not-a-knot'` (default),
 * `'natural'` ($M_0 = M_n = 0$), `'clamped'` ($f'(x_0) = f'(x_n) = 0$), `'periodic'` ($y_0 = y_n$),
 * or specified boundary first derivatives `{ first }` or second derivatives `{ second }`.
 *
 * @param x Strictly increasing node coordinates tensor of length $n$.
 * @param y Observed values tensor of length $n$.
 * @param options Configuration options specifying the boundary condition `bc`.
 * @param options.bc End boundary condition type (default `'not-a-knot'`).
 * @returns $C^2$ continuous piecewise cubic polynomial interpolant.
 *
 * @example Natural cubic spline
 * const x = tensor([0, 1, 2, 3])
 * const y = tensor([0, 1, 0, 1])
 * const spline = cubicSpline(x, y, { bc: 'natural' })
 * print('spline at 1.5 =', evaluatePiecewise(spline, tensor([1.5])))
 */
export function cubicSpline(
  x: Tensor,
  y: Tensor,
  { bc = 'not-a-knot' }: { bc?: EndCondition } = {},
): PiecewisePolynomial {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'cubicSpline')
  const n = X.length - 1
  if (bc === 'periodic' && Math.abs(Y[n] - Y[0]) > 1e-12 * (1 + Math.abs(Y[0])))
    throw new DomainError('cubicSpline', 'cubicSpline: periodic end conditions need y[0] = y[n]')
  const h = Float64Array.from({ length: n }, (_, i) => X[i + 1] - X[i])
  const delta = Float64Array.from({ length: n }, (_, i) => (Y[i + 1] - Y[i]) / h[i])
  if (n === 1 && (bc === 'not-a-knot' || bc === 'natural' || bc === 'periodic')) return linearInterpolant(x, y)
  const size = bc === 'periodic' ? n : n + 1
  const A = new Float64Array(size * size)
  const r = new Float64Array(size)
  const col = (j: number) => (bc === 'periodic' ? ((j % n) + n) % n : j)
  for (let i = 1; i < n; i++) {
    A[i * size + col(i - 1)] += h[i - 1]
    A[i * size + col(i)] += 2 * (h[i - 1] + h[i])
    A[i * size + col(i + 1)] += h[i]
    r[i] = 6 * (delta[i] - delta[i - 1])
  }
  if (bc === 'periodic') {
    A[col(n - 1)] += h[n - 1]
    A[0] += 2 * (h[n - 1] + h[0])
    A[col(1)] += h[0]
    r[0] = 6 * (delta[0] - delta[n - 1])
    const M = denseSolve(A, r, size)
    return fromMoments(X, Y, [...M, M[0]])
  }
  const last = n * size
  if (bc === 'natural' || (typeof bc === 'object' && 'second' in bc)) {
    const [a, b] = bc === 'natural' ? [0, 0] : bc.second
    A[0] = 1
    r[0] = a
    A[last + n] = 1
    r[n] = b
  } else if (bc === 'clamped' || (typeof bc === 'object' && 'first' in bc)) {
    const [s0, s1] = bc === 'clamped' ? [0, 0] : bc.first
    // f′(x₀) = s₀ and f′(xₙ) = s₁ in terms of the moments.
    A[0] = 2 * h[0]
    A[1] = h[0]
    r[0] = 6 * (delta[0] - s0)
    A[last + n - 1] = h[n - 1]
    A[last + n] = 2 * h[n - 1]
    r[n] = 6 * (s1 - delta[n - 1])
  } else if (n === 2) {
    // Three points: the two not-a-knot conditions coincide; the spline is the interpolating parabola.
    A[0] = 1
    A[1] = -1
    A[last + 1] = 1
    A[last + 2] = -1
  } else {
    // Not-a-knot: the third derivative is continuous at x₁ and xₙ₋₁.
    A[0] = h[1]
    A[1] = -(h[0] + h[1])
    A[2] = h[0]
    A[last + n - 2] = h[n - 1]
    A[last + n - 1] = -(h[n - 2] + h[n - 1])
    A[last + n] = h[n - 2]
  }
  return fromMoments(X, Y, denseSolve(A, r, size))
}

/**
 * Construct the natural cubic spline ($M_0 = M_n = 0$) through points $(x_i, y_i)$.
 *
 * The natural cubic spline is the unique interpolant minimising total curvature $\int_a^b (f''(x))^2\,dx$
 * (Holladay's theorem, 1957).
 *
 * @param x Strictly increasing node coordinates tensor of length $n$.
 * @param y Observed values tensor of length $n$.
 * @returns Piecewise cubic polynomial with zero second derivatives at both endpoints.
 *
 * @example Natural cubic spline interpolation
 * const spline = naturalCubicSpline(tensor([0, 1, 2, 3]), tensor([0, 1, 0, 1]))
 * print('value at 0.5 =', evaluatePiecewise(spline, tensor([0.5])))
 */
export function naturalCubicSpline(x: Tensor, y: Tensor): PiecewisePolynomial {
  return cubicSpline(x, y, { bc: 'natural' })
}

/**
 * Compute slopes for piecewise cubic Hermite interpolating polynomial (PCHIP).
 *
 * Uses weighted harmonic means of adjacent secants (Fritsch & Butland, 1984), setting slopes to zero
 * wherever the data change monotonicity direction.
 *
 * @param X Node coordinates array.
 * @param Y Response values array.
 * @returns Slopes array of length $n$.
 */
function pchipSlopes(X: F64, Y: F64): F64 {
  const n = X.length
  const h = Float64Array.from({ length: n - 1 }, (_, i) => X[i + 1] - X[i])
  const m = Float64Array.from({ length: n - 1 }, (_, i) => (Y[i + 1] - Y[i]) / h[i])
  if (n === 2) return Float64Array.of(m[0], m[0])
  const d = new Float64Array(n)
  for (let k = 1; k < n - 1; k++) {
    if (m[k - 1] === 0 || m[k] === 0 || Math.sign(m[k - 1]) !== Math.sign(m[k])) continue
    const w1 = 2 * h[k] + h[k - 1]
    const w2 = h[k] + 2 * h[k - 1]
    d[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])
  }
  const edge = (h0: number, h1: number, m0: number, m1: number) => {
    const e = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if (Math.sign(e) !== Math.sign(m0)) return 0
    if (Math.sign(m0) !== Math.sign(m1) && Math.abs(e) > 3 * Math.abs(m0)) return 3 * m0
    return e
  }
  d[0] = edge(h[0], h[1], m[0], m[1])
  d[n - 1] = edge(h[n - 2], h[n - 3], m[n - 2], m[n - 3])
  return d
}

/**
 * Construct the shape-preserving piecewise cubic Hermite interpolating polynomial (PCHIP).
 *
 * Guarantees monotonicity preservation without overshoot (Fritsch & Carlson, 1980; Fritsch & Butland, 1984).
 * Matches SciPy's `PchipInterpolator`.
 *
 * @param x Strictly increasing node coordinates tensor of length $n$.
 * @param y Observed values tensor of length $n$.
 * @returns Shape-preserving $C^1$ piecewise cubic polynomial.
 *
 * @example Monotone PCHIP interpolation
 * const spline = pchip(tensor([0, 1, 2, 3]), tensor([0, 1, 1, 0]))
 * print('value at 1.5 =', evaluatePiecewise(spline, tensor([1.5])))
 */
export function pchip(x: Tensor, y: Tensor): PiecewisePolynomial {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'pchip')
  return hermite(X, Y, pchipSlopes(X, Y))
}

/**
 * Compute local slopes for Akima or modified Akima (makima) interpolation.
 *
 * Under Akima (1970), $t_i = \frac{|m_{i+1} - m_i| m_{i-1} + |m_{i-1} - m_{i-2}| m_i}{|m_{i+1} - m_i| + |m_{i-1} - m_{i-2}|}$.
 * Under `makima`, an additional average secant term is added to prevent overshoot along flat regions.
 *
 * @param X Node coordinates array.
 * @param Y Response values array.
 * @param method Slope estimation formula: `'akima'` or `'makima'`.
 * @returns Slopes array of length $n$.
 */
function akimaSlopes(X: F64, Y: F64, method: 'akima' | 'makima'): F64 {
  const n = X.length
  if (n === 2) {
    const s = (Y[1] - Y[0]) / (X[1] - X[0])
    return Float64Array.of(s, s)
  }
  const m = new Float64Array(n + 3)
  for (let i = 0; i < n - 1; i++) m[i + 2] = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])
  m[1] = 2 * m[2] - m[3]
  m[0] = 2 * m[1] - m[2]
  m[n + 1] = 2 * m[n] - m[n - 1]
  m[n + 2] = 2 * m[n + 1] - m[n]
  const dm = Float64Array.from({ length: n + 2 }, (_, i) => Math.abs(m[i + 1] - m[i]))
  const pm = Float64Array.from({ length: n + 2 }, (_, i) => Math.abs(m[i + 1] + m[i]))
  const out = new Float64Array(n)
  let maxF = 0
  const f1 = new Float64Array(n)
  const f2 = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    f1[i] = dm[i + 2] + (method === 'makima' ? 0.5 * pm[i + 2] : 0)
    f2[i] = dm[i] + (method === 'makima' ? 0.5 * pm[i] : 0)
    maxF = Math.max(maxF, f1[i] + f2[i])
  }
  const cutoff = 1e-9 * maxF
  for (let i = 0; i < n; i++) {
    const f12 = f1[i] + f2[i]
    out[i] = f12 > cutoff ? (f1[i] * m[i + 1] + f2[i] * m[i + 2]) / f12 : 0.5 * (m[i + 1] + m[i + 2])
  }
  return out
}

/**
 * Construct the Akima sub-spline interpolant through points $(x_i, y_i)$.
 *
 * Avoids the oscillations and wiggles of standard cubic splines near outliers and sharp transitions
 * (Akima, 1970). With `method: 'makima'` (modified Akima), adds a safeguard against overshoot on flat stretches.
 *
 * @param x Strictly increasing node coordinates tensor of length $n$.
 * @param y Observed values tensor of length $n$.
 * @param options Interpolation options specifying the slope method.
 * @param options.method Slope formula: `'akima'` (default) or `'makima'`.
 * @returns Locally determined $C^1$ piecewise cubic polynomial.
 *
 * @example Akima spline interpolation
 * const spline = akima(tensor([0, 1, 2, 3, 4]), tensor([0, 0, 1, 1, 1]))
 * print('value at 2.5 =', evaluatePiecewise(spline, tensor([2.5])))
 */
export function akima(x: Tensor, y: Tensor, { method = 'akima' }: { method?: 'akima' | 'makima' } = {}) {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'akima')
  return hermite(X, Y, akimaSlopes(X, Y, method))
}

/** Result returned by the cubic smoothing spline estimator `smoothingSpline`. */
export type SmoothingSpline = {
  /** The fitted natural cubic spline with knots at the observed data points. */
  spline: PiecewisePolynomial
  /** Fitted values $g(x_i)$ evaluated at the data points, shape $[n]$. */
  fitted: Tensor
  /** Regularisation parameter $\lambda \ge 0$. */
  lambda: number
}

/**
 * Fit a cubic smoothing spline minimising $\sum_i w_i (y_i - f(x_i))^2 + \lambda \int (f''(x))^2\,dx$.
 *
 * Solves the Reinsch algorithm (Reinsch, 1967; Green & Silverman, 1994, §2.3.3) for a natural cubic spline
 * with knots at the data points. Matches SciPy's `make_smoothing_spline(x, y, w, lam)`.
 *
 * @param x Strictly increasing node coordinates tensor of length $n$.
 * @param y Observed values tensor of length $n$.
 * @param options Smoothing configuration including penalty weight $\lambda$ and point weights.
 * @param options.lambda Non-negative smoothing parameter $\lambda \ge 0$.
 * @param options.weights Optional positive observation weights tensor of length $n$.
 * @returns A `SmoothingSpline` containing the fitted spline, predicted values, and $\lambda$.
 *
 * @example Cubic smoothing spline
 * const x = tensor([0, 1, 2, 3, 4])
 * const y = tensor([0, 0.9, 2.1, 2.9, 4.2])
 * const fit = smoothingSpline(x, y, { lambda: 1.0 })
 * print('fitted =', fit.fitted)
 */
export function smoothingSpline(
  x: Tensor,
  y: Tensor,
  { lambda, weights }: { lambda: number; weights?: Tensor },
): SmoothingSpline {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'smoothingSpline')
  if (!(lambda >= 0)) throw new DomainError('smoothingSpline', 'smoothingSpline: λ must be ≥ 0')
  const n = X.length
  const w = weights ? f64(weights) : new Float64Array(n).fill(1)
  if (n < 3) return { spline: linearInterpolant(x, y), fitted: vec(Y), lambda }
  const h = Float64Array.from({ length: n - 1 }, (_, i) => X[i + 1] - X[i])
  const m = n - 2
  const Q = new Float64Array(n * m)
  for (let j = 0; j < m; j++) {
    Q[j * m + j] = 1 / h[j]
    Q[(j + 1) * m + j] = -1 / h[j] - 1 / h[j + 1]
    Q[(j + 2) * m + j] = 1 / h[j + 1]
  }
  const A = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) {
      let s = 0
      for (let k = 0; k < n; k++) s += (Q[k * m + i] * Q[k * m + j]) / w[k]
      const R = i === j ? (h[i] + h[i + 1]) / 3 : Math.abs(i - j) === 1 ? h[Math.max(i, j)] / 6 : 0
      A[i * m + j] = R + lambda * s
    }
  const rhs = new Float64Array(m)
  for (let j = 0; j < m; j++) for (let k = 0; k < n; k++) rhs[j] += Q[k * m + j] * Y[k]
  const gamma = denseSolve(A, rhs, m)
  const g = Float64Array.from(Y, (yk, k) => {
    let s = 0
    for (let j = 0; j < m; j++) s += Q[k * m + j] * gamma[j]
    return yk - (lambda * s) / w[k]
  })
  return { spline: fromMoments(X, g, [0, ...gamma, 0]), fitted: vec(g), lambda }
}
