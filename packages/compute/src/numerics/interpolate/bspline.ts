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
const f64 = (t: Tensor): F64 => Float64Array.from(toFlat(t))

// ── Basis ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The p + 1 B-splines of degree p that are non-zero on knot span i, N_{i−p} … N_i at x, by the triangular recurrence
 * of de Boor (Piegl and Tiller, 1997, "The NURBS Book", Algorithm A2.2). Outside [tᵢ, tᵢ₊₁) the same polynomial
 * pieces are continued.
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

/** The span: the largest i in [p, N − 1] with tᵢ ≤ x (clamped at both ends), N the number of B-splines. */
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

/** Number of B-splines of degree p on a knot vector of length m: m − p − 1. */
export function bsplineCount(knots: Tensor, degree: Size): Size {
  return knots.shape[0] - degree - 1
}

/** How a basis treats x outside its base interval [t_p, t_N]. */
export type Extrapolation = 'continue' | 'zero' | 'clamp'

/**
 * The B-spline basis matrix [n, N] (or its `derivative`-th derivative) of degree p on a non-decreasing knot vector
 * t [N + p + 1] at x [n], by Cox–de Boor (de Boor, 1978, "A Practical Guide to Splines", ch. IX) and the derivative
 * rule B′ᵢ,ₚ = p(Bᵢ,ₚ₋₁/(tᵢ₊ₚ − tᵢ) − Bᵢ₊₁,ₚ₋₁/(tᵢ₊ₚ₊₁ − tᵢ₊₁)). Outside [t_p, t_N], `extrapolation` continues the
 * end polynomials (`'continue'`, default, as scipy's `BSpline`), gives zeros (`'zero'`) or holds the boundary values
 * (`'clamp'`). At x = t_N the last span is closed, so the basis reaches the end.
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
 * Equally spaced knots for P-splines: `segments` intervals on [lo, hi] with spacing h = (hi − lo)/segments, extended
 * by `degree` knots beyond each end (Eilers and Marx, 1996), giving segments + degree B-splines that sum to 1 on
 * [lo, hi].
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
 * A clamped (open) knot vector from breakpoints b₀ < … < b_m: each end repeated degree + 1 times, so the spline
 * interpolates its end coefficients; m + degree B-splines.
 */
export function clampedKnots(breaks: Tensor, degree: Size = 3): Tensor {
  const b = f64(breaks)
  const m = b.length - 1
  const out = [...Array(degree).fill(b[0]), ...b, ...Array(degree).fill(b[m])]
  return fromData(Float64Array.from(out), [out.length])
}

/** A spline in B-spline form: f(x) = Σⱼ cⱼ Bⱼ(x). */
export type BSpline = {
  readonly kind: 'bspline'
  readonly knots: Tensor
  readonly degree: number
  readonly coefficients: Tensor
  /** f (or a derivative) at x [n] → [n]. */
  evaluate(x: Tensor, derivative?: number): Tensor
}

/** The spline Σⱼ cⱼ Bⱼ with given knots, degree and coefficients [N]. */
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

/** Least squares on a design [n, N]: minimise Σ wᵢ(yᵢ − (Bβ)ᵢ)² + βᵀPβ. Returns β and the pieces of its inference. */
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
 * The least-squares spline with given knots and degree through (xᵢ, yᵢ), weights optional (scipy's
 * `make_lsq_spline`).
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

/** The order-d difference matrix D [p − d, p]: (Dβ)ⱼ = Δᵈβⱼ = Σₐ (−1)ᵈ⁻ᵃ C(d, a) βⱼ₊ₐ. */
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

/** The P-spline penalty DᵀD [p, p] for the order-d difference matrix. */
export function differencePenalty(p: Size, order: Size): Tensor {
  return crossProduct(differenceMatrix(p, order))
}

/** The cyclic order-d difference matrix [p, p]: differences taken around a circle, so the penalty wraps. */
export function cyclicDifferenceMatrix(p: Size, order: Size): Tensor {
  const D0 = f64(differenceMatrix(order + 1, order))
  const out = new Float64Array(p * p)
  for (let r = 0; r < p; r++) for (let a = 0; a <= order; a++) out[r * p + ((r + a) % p)] += D0[a]
  return fromData(out, [p, p])
}

/** The cyclic difference penalty DᵀD [p, p]. */
export function cyclicDifferencePenalty(p: Size, order: Size): Tensor {
  return crossProduct(cyclicDifferenceMatrix(p, order))
}

/** AᵀA. */
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
 * The derivative penalty S [N, N] with S_kl = ∫ₗₒʰⁱ B⁽ᵐ⁾ₖ(x) B⁽ᵐ⁾ₗ(x) dx (so βᵀSβ = ∫f⁽ᵐ⁾²), exact by
 * Gauss–Legendre quadrature with degree + 1 nodes on each knot span (O'Sullivan, 1986; Wood, 2017, §5.3.3). The range
 * defaults to the base interval [t_p, t_N].
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
 * The k periodic B-splines of degree p on [lo, hi) with equally spaced knots: the P-spline basis of k + p functions
 * with the last p folded onto the first p, so f(lo) = f(hi) with all derivatives up to p − 1 (Eilers and Marx, 2010;
 * Wood, 2017, §5.3.2). x is reduced modulo the period first.
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

/** The row-wise Kronecker product of two bases A [n, p] and B [n, q]: [n, p·q], B's index running fastest. */
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
 * The two penalties of a tensor-product smooth from marginal penalties S_A [p, p] and S_B [q, q]: S_A ⊗ I_q (roughness
 * along the first variable) and I_p ⊗ S_B (along the second), each with its own λ (Wood, 2006, "Low-rank scale-invariant
 * tensor product smooths").
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

/** Options of `pspline`. */
export type PSplineOptions = {
  /** Number of equal segments on [lo, hi] (default 20): segments + degree B-splines. */
  segments?: number
  degree?: number
  /** Order of the difference penalty (default 2). */
  order?: number
  /** Smoothing parameter λ ≥ 0, or `'gcv'` to minimise the generalised cross-validation score (default 1). */
  lambda?: number | 'gcv'
  /** Range of the basis (default: the range of x). */
  range?: readonly [number, number]
  weights?: Tensor
}

/** A fitted P-spline smoother. */
export type PSplineFit = BSpline & {
  readonly lambda: number
  /** Fitted values at the data, [n]. */
  readonly fitted: Tensor
  /** Effective degrees of freedom tr(H), H = B(BᵀWB + λDᵀD)⁻¹BᵀW. */
  readonly edf: number
  /** Diagonal of the hat matrix H, [n] (leverages). */
  readonly leverage: Tensor
  /** Weighted residual sum of squares. */
  readonly rss: number
  /** Generalised cross-validation score n·RSS/(n − edf)² (Craven and Wahba, 1979). */
  readonly gcv: number
  /** Residual variance estimate σ̂² = RSS/(n − edf). */
  readonly sigma2: number
  /** Bayesian posterior covariance σ̂²(BᵀWB + λP)⁻¹ of the coefficients (Wahba, 1983; Wood, 2017, §6.10). */
  readonly covariance: Tensor
  /** Pointwise posterior standard errors of f at x [m]. */
  standardError(x: Tensor): Tensor
}

/** One P-spline fit at a given λ. */
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
 * The GCV score at each log₁₀ λ in `logLambdas` for a P-spline (other options as `pspline`), e.g. to draw the GCV
 * curve a λ was chosen from.
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
 * A P-spline smoother (Eilers and Marx, 1996): equally spaced B-splines with a difference penalty on neighbouring
 * coefficients, minimising Σwᵢ(yᵢ − f(xᵢ))² + λ‖Dβ‖². With `lambda: 'gcv'`, λ minimises the GCV score over
 * log₁₀ λ ∈ [−6, 6] (a grid, then golden-section refinement).
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
