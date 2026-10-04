/**
 * Piecewise polynomials and the one-dimensional interpolants built on them: piecewise linear, cubic splines with four
 * end conditions, cubic Hermite, PCHIP, Akima, and the cubic smoothing spline. Pieces are stored in the local power
 * basis, as scipy's `PPoly`: on [bᵢ, bᵢ₊₁], f(x) = Σₖ cᵢₖ (x − bᵢ)ᵏ.
 *
 * References: de Boor (1978), "A Practical Guide to Splines", ch. IV; Fritsch and Carlson (1980) and Fritsch and
 * Butland (1984) for PCHIP; Akima (1970); Reinsch (1967) and Green and Silverman (1994), §2.3, for smoothing splines.
 * Each is checked against scipy.interpolate.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Scalar } from 'aifn-compute/foundation/contracts'
import { solve } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** A piecewise polynomial in the local power basis. */
export type PiecewisePolynomial = {
  readonly kind: 'piecewise-polynomial'
  /** Breakpoints b₀ < b₁ < … < b_m, [m + 1]. */
  readonly breaks: Tensor
  /** Coefficients [m, order]: row i holds cᵢ₀, cᵢ₁, … (ascending powers of x − bᵢ). */
  readonly coefficients: Tensor
}

type F64 = Float64Array

const vec = (a: ArrayLike<number>) => fromData(Float64Array.from(a), [a.length])
const f64 = (t: Tensor | ArrayLike<number>): F64 =>
  Float64Array.from('shape' in (t as Tensor) ? toFlat(t as Tensor) : (t as ArrayLike<number>))

/** A piecewise polynomial from breaks and rows of ascending coefficients. */
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

/** The index of the piece holding x; outside the breaks, the end pieces (extrapolation). */
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
 * The piecewise polynomial (or its `derivative`-th derivative) at x [n] → [n]. Outside the breaks the end pieces are
 * continued (scipy's `extrapolate=True`); with `extrapolate: false` those values are NaN.
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

/** ∫ₐᵇ f(x) dx of a piecewise polynomial (end pieces continued outside the breaks). */
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

function checkData(x: F64, y: F64, where: string) {
  if (x.length !== y.length) throw new ShapeError(where, `${where}: ${x.length} x values but ${y.length} y values`)
  if (x.length < 2) throw new DomainError(where, `${where}: needs at least two points`)
  for (let i = 1; i < x.length; i++)
    if (!(x[i] > x[i - 1])) throw new DomainError(where, `${where}: x must be strictly increasing`)
}

/** The piecewise linear interpolant through (xᵢ, yᵢ), x strictly increasing. */
export function linearInterpolant(x: Tensor, y: Tensor): PiecewisePolynomial {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'linearInterpolant')
  return piecewisePolynomial(
    X,
    Array.from({ length: X.length - 1 }, (_, i) => [Y[i], (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])]),
  )
}

/** Cubic Hermite pieces from values and slopes at the breaks. */
function hermite(X: F64, Y: F64, M: ArrayLike<number>): PiecewisePolynomial {
  const rows = Array.from({ length: X.length - 1 }, (_, i) => {
    const h = X[i + 1] - X[i]
    const delta = (Y[i + 1] - Y[i]) / h
    return [Y[i], M[i], (3 * delta - 2 * M[i] - M[i + 1]) / h, (M[i] + M[i + 1] - 2 * delta) / (h * h)]
  })
  return piecewisePolynomial(X, rows)
}

/** The cubic Hermite interpolant with given slopes at the data (scipy's `CubicHermiteSpline`). */
export function hermiteSpline(x: Tensor, y: Tensor, slopes: Tensor): PiecewisePolynomial {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'hermiteSpline')
  return hermite(X, Y, f64(slopes))
}

/** Cubic pieces from values and second derivatives M at the breaks. */
function fromMoments(X: F64, Y: ArrayLike<number>, M: ArrayLike<number>): PiecewisePolynomial {
  const rows = Array.from({ length: X.length - 1 }, (_, i) => {
    const h = X[i + 1] - X[i]
    const b = (Y[i + 1] - Y[i]) / h - (h * (2 * M[i] + M[i + 1])) / 6
    return [Y[i], b, M[i] / 2, (M[i + 1] - M[i]) / (6 * h)]
  })
  return piecewisePolynomial(X, rows)
}

/** End conditions of an interpolating cubic spline (scipy's `bc_type`). */
export type EndCondition =
  'not-a-knot' | 'natural' | 'clamped' | 'periodic' | { first: [number, number] } | { second: [number, number] }

/** Solve a small dense system through `aifn-compute/numerics/linalg`. */
function denseSolve(A: F64, r: F64, n: number): F64 {
  return f64(solve(fromData(A, [n, n]), fromData(r, [n])) as Tensor)
}

/**
 * The interpolating cubic spline through (xᵢ, yᵢ), C² at the interior breaks, solved for the second derivatives Mᵢ
 * from hᵢ₋₁Mᵢ₋₁ + 2(hᵢ₋₁ + hᵢ)Mᵢ + hᵢMᵢ₊₁ = 6(δᵢ − δᵢ₋₁) (de Boor, 1978, ch. IV). End conditions as scipy's
 * `CubicSpline`: `not-a-knot` (default), `natural` (M = 0), `clamped` (zero end slopes), `periodic` (y₀ = yₙ
 * required), or given end slopes `{ first }` or second derivatives `{ second }`.
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

/** The natural cubic spline (M₀ = Mₙ = 0): the interpolant minimising ∫f″² (Holladay, 1957). */
export function naturalCubicSpline(x: Tensor, y: Tensor): PiecewisePolynomial {
  return cubicSpline(x, y, { bc: 'natural' })
}

/**
 * Slopes of the monotone piecewise cubic interpolant (PCHIP) as scipy's `PchipInterpolator`: weighted harmonic means
 * of neighbouring secants (Fritsch and Butland, 1984), zero where the data turn, and a shape-preserving three-point
 * formula at the ends.
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

/** The monotone piecewise cubic Hermite interpolant (PCHIP): no overshoot, monotone where the data are. */
export function pchip(x: Tensor, y: Tensor): PiecewisePolynomial {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'pchip')
  return hermite(X, Y, pchipSlopes(X, Y))
}

/**
 * Akima's slopes (Akima, 1970; scipy's `Akima1DInterpolator`): tᵢ = (|mᵢ₊₁ − mᵢ| mᵢ₋₁ + |mᵢ₋₁ − mᵢ₋₂| mᵢ) /
 * (|mᵢ₊₁ − mᵢ| + |mᵢ₋₁ − mᵢ₋₂|) with two secants extrapolated linearly at each end; `makima` (modified Akima) adds
 * |mᵢ₊₁ + mᵢ|/2 and |mᵢ₋₁ + mᵢ₋₂|/2 to the weights, which removes overshoot on flat stretches.
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

/** Akima's interpolant (or the modified `makima`): local, with less wiggle than a cubic spline near outliers. */
export function akima(x: Tensor, y: Tensor, { method = 'akima' }: { method?: 'akima' | 'makima' } = {}) {
  const X = f64(x)
  const Y = f64(y)
  checkData(X, Y, 'akima')
  return hermite(X, Y, akimaSlopes(X, Y, method))
}

/** The result of `smoothingSpline`. */
export type SmoothingSpline = {
  /** The natural cubic spline with knots at the data. */
  spline: PiecewisePolynomial
  /** Fitted values g(xᵢ), [n]. */
  fitted: Tensor
  lambda: number
}

/**
 * The cubic smoothing spline minimising Σ wᵢ(yᵢ − f(xᵢ))² + λ∫f″² (Reinsch, 1967; Green and Silverman, 1994,
 * §2.3.3): a natural cubic spline with knots at the data, found from (R + λQᵀW⁻¹Q)γ = Qᵀy with Q the n × (n − 2)
 * second-difference matrix and R the tridiagonal Gram matrix; γ holds the interior second derivatives and
 * g = y − λW⁻¹Qγ the fitted values. Matches scipy's `make_smoothing_spline(x, y, w, lam)`.
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
