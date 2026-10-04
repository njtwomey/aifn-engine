/**
 * Local likelihood logistic regression (Loader 1999, "Local regression and likelihood", ch. 4; as used by Yang et al.
 * 2024, AAAI): near a point x the log-odds are a polynomial in z − x, so the fit at x maximises the kernel-weighted
 * Bernoulli log-likelihood
 *
 *   ℓ_x(β) = Σᵢ w_{i,h}(x) [yᵢ log σ(⟨β, A_p(xᵢ − x)⟩) + (1 − yᵢ) log(1 − σ(⟨β, A_p(xᵢ − x)⟩))],
 *
 * with w_{i,h}(x) = K((x − xᵢ)/h), K the Gaussian kernel, and A_p the polynomial basis of order p (p = 1: [1, v₀, v₁];
 * p = 2 adds [v₀²/2, v₀v₁, v₁²/2], Loader's Eq. 2.9). The estimate is r̂(x) = σ(β̂₀). Its variance is the sandwich
 * V[β̂] = B⁻¹CB⁻¹ with B = Σ wᵢ pᵢ(1 − pᵢ) aᵢaᵢᵀ and C = Σ (yᵢ − pᵢ)² wᵢ² aᵢaᵢᵀ (Frölich 2006); the covariance of the
 * fits at two points x and x′ is B_x⁻¹ (Σᵢ s_{i,x} s_{i,x′}ᵀ) B_{x′}⁻¹ with sᵢ = wᵢ(yᵢ − pᵢ)aᵢ the score of point i, which
 * reduces to the sandwich when x = x′. The fit is the binomial GLM with prior weights wᵢ, solved by Newton's method
 * (`aifn-compute/optim/second-order`) on the weighted negative log-likelihood with autodiff gradients and Hessian; the
 * parametric noise test fits its global logistic regression with the same routine (all weights 1).
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, permutation, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { hessian, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { matmul, mul, reshape, sub, sum, unwrap, type Value } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { inverse } from 'aifn-compute/numerics/linalg'
import { sigmoid, softplus } from 'aifn-compute/numerics/special'
import { newton } from 'aifn-compute/optim/second-order'

/**
 * The weighted logistic MLE: β̂ maximising Σᵢ wᵢ [yᵢ zᵢ − log(1 + e^{zᵢ})], z = Aβ, for a design A [n, q], labels
 * y ∈ {0, 1} and weights w ≥ 0, by damped Newton from β = 0; returns β̂ and the fitted pᵢ = σ(zᵢ).
 */
export function weightedLogistic(
  design: Float64Array,
  y: ArrayLike<number>,
  weights: ArrayLike<number>,
  q: Size,
): { beta: Float64Array; p: Float64Array; converged: boolean } {
  const n = y.length
  const A = fromData(design, [n, q])
  const Y = fromData(Float64Array.from(y), [n])
  const W = fromData(Float64Array.from(weights), [n])
  const nll = (b: Value) => {
    const z = reshape(matmul(A, reshape(b, [q, 1])), [n])
    return sum(mul(W, sub(softplus(z), mul(Y, z))))
  }
  const vg = valueAndGrad(nll)
  const H = hessian(nll)
  const state = run(
    newton(
      (b) => {
        const r = vg(b)
        return { value: unwrap(r.value) as number, grad: r.grad as Tensor }
      },
      { hessian: (b) => H(b) as Tensor, tolerance: 1e-10 },
    ),
    { x0: new Float64Array(q) },
    100,
  )
  const beta = Float64Array.from(toFlat(state.x))
  const p = Float64Array.from(toFlat(sigmoid(reshape(matmul(A, fromData(beta, [q, 1])), [n])) as Tensor))
  return { beta, p, converged: state.converged === true }
}

/** The local polynomial basis A_p(v) of order p ∈ {0, 1, 2} (Loader's Eq. 2.9 for p = 2). */
export function localPolynomialBasis(v: ArrayLike<number>, degree: 0 | 1 | 2 = 1): number[] {
  const out = [1]
  if (degree >= 1) for (let j = 0; j < v.length; j++) out.push(v[j])
  if (degree >= 2)
    for (let j = 0; j < v.length; j++)
      for (let k = j; k < v.length; k++) out.push(j === k ? (v[j] * v[j]) / 2 : v[j] * v[k])
  return out
}

/** Options of a local logistic fit. */
export interface LocalLogisticOptions {
  /** The kernel bandwidth h (default 1). */
  bandwidth?: number
  /** The polynomial order p (default 1, local linear). */
  degree?: 0 | 1 | 2
}

/** A local logistic fit at one point. */
export interface LocalLogisticFit {
  readonly at: readonly number[]
  /** r̂(x) = σ(β̂₀). */
  readonly estimate: number
  readonly coefficients: Float64Array
  /** V[β̂] = B⁻¹CB⁻¹ (q × q, row-major) and its (0, 0) entry, the variance of the local log-odds β̂₀. */
  readonly covariance: Float64Array
  readonly logitVariance: number
  /** B⁻¹ (q × q) and the per-point scores sᵢ = wᵢ(yᵢ − pᵢ)aᵢ (n × q), for covariances between points. */
  readonly breadInverse: Float64Array
  readonly scores: Float64Array
  readonly basisSize: Size
  /** The effective number of points Σ wᵢ / max wᵢ (small when the neighbourhood is empty). */
  readonly effectivePoints: number
}

function rowsOf(x: MatrixLike, where: string) {
  const m = dense.toMatrixF64(x, where)
  return { data: m.data, n: m.m, d: m.n }
}

/** Fit the local logistic model at `at` to points x [n, d] with labels y ∈ {0, 1} (module notes). */
export function localLogistic(
  x: MatrixLike,
  y: ArrayLike<number>,
  at: ArrayLike<number>,
  options: LocalLogisticOptions = {},
): LocalLogisticFit {
  const { bandwidth: h = 1, degree = 1 } = options
  if (!(h > 0)) throw new DomainError('localLogistic', 'localLogistic: the bandwidth must be positive')
  const { data, n, d } = rowsOf(x, 'localLogistic')
  if (y.length !== n) throw new ShapeError('localLogistic', `localLogistic: ${y.length} labels for ${n} points`)
  if (at.length !== d)
    throw new ShapeError('localLogistic', `localLogistic: the point has ${at.length} coordinates, not ${d}`)
  const q = localPolynomialBasis(new Float64Array(d), degree).length
  const design = new Float64Array(n * q)
  const w = new Float64Array(n)
  let top = 0
  let total = 0
  for (let i = 0; i < n; i++) {
    const v = Array.from({ length: d }, (_, j) => data[i * d + j] - at[j])
    design.set(localPolynomialBasis(v, degree), i * q)
    // The Gaussian kernel's normalising constant cancels in the estimate and in the sandwich.
    w[i] = Math.exp(-0.5 * v.reduce((s, u) => s + (u / h) ** 2, 0))
    top = Math.max(top, w[i])
    total += w[i]
  }
  const { beta, p } = weightedLogistic(design, y, w, q)
  const B = new Float64Array(q * q)
  const scores = new Float64Array(n * q)
  for (let i = 0; i < n; i++) {
    const a = design.subarray(i * q, (i + 1) * q)
    const bw = w[i] * p[i] * (1 - p[i])
    const r = w[i] * (y[i] - p[i])
    for (let j = 0; j < q; j++) {
      scores[i * q + j] = r * a[j]
      for (let k = 0; k < q; k++) B[j * q + k] += bw * a[j] * a[k]
    }
  }
  const Binv = dense.data(inverse(fromData(B, [q, q])) as Tensor)
  const covariance = sandwich(Binv, scores, Binv, scores, n, q)
  return {
    at: Array.from(at),
    estimate: 1 / (1 + Math.exp(-beta[0])),
    coefficients: beta,
    covariance,
    logitVariance: covariance[0],
    breadInverse: Float64Array.from(Binv),
    scores,
    basisSize: q,
    effectivePoints: top > 0 ? total / top : 0,
  }
}

/** A⁻¹ (Σᵢ sᵢ tᵢᵀ) B⁻¹ for score rows s, t (n × q). */
function sandwich(
  Ainv: ArrayLike<number>,
  s: Float64Array,
  Binv: ArrayLike<number>,
  t: Float64Array,
  n: number,
  q: number,
) {
  const C = new Float64Array(q * q)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < q; j++) for (let k = 0; k < q; k++) C[j * q + k] += s[i * q + j] * t[i * q + k]
  return dense.matMul(dense.matMul(Ainv, C, q, q, q), Binv, q, q, q)
}

/** The covariance of the local log-odds β̂₀ of two fits on the same data (module notes). */
export function localLogisticCovariance(a: LocalLogisticFit, b: LocalLogisticFit): number {
  if (a.basisSize !== b.basisSize || a.scores.length !== b.scores.length)
    throw new ShapeError('localLogisticCovariance', 'localLogisticCovariance: the fits must share data and degree')
  const q = a.basisSize
  return sandwich(a.breadInverse, a.scores, b.breadInverse, b.scores, a.scores.length / q, q)[0]
}

/**
 * The bandwidth from a grid that minimises the leave-one-out log loss −Σᵢ log p̂₋ᵢ(yᵢ | xᵢ) over a random subsample of
 * `subsample` points (default 100; Yang et al. use sub-sampled LOO-CV), each point refitted without itself.
 */
export function localLogisticBandwidth(
  s: Stream,
  x: MatrixLike,
  y: ArrayLike<number>,
  bandwidths: readonly number[],
  options: { degree?: 0 | 1 | 2; subsample?: Size } = {},
): { bandwidth: number; losses: number[] } {
  const { degree = 1, subsample = 100 } = options
  const { data, n, d } = rowsOf(x, 'localLogisticBandwidth')
  const pick = Array.from(toFlat(permutation(child(s, 'subsample'), n))).slice(0, Math.min(subsample, n))
  const losses = bandwidths.map((h) => {
    let loss = 0
    for (const i of pick) {
      const keep = Array.from({ length: n }, (_, j) => j).filter((j) => j !== i)
      const xs = new Float64Array(keep.length * d)
      keep.forEach((j, r) => xs.set(data.subarray(j * d, (j + 1) * d), r * d))
      const fit = localLogistic(
        fromData(xs, [keep.length, d]),
        keep.map((j) => y[j]),
        data.subarray(i * d, (i + 1) * d),
        {
          bandwidth: h,
          degree,
        },
      )
      const p = Math.min(1 - 1e-12, Math.max(1e-12, fit.estimate))
      loss -= y[i] === 1 ? Math.log(p) : Math.log(1 - p)
    }
    return loss / pick.length
  })
  let best = 0
  losses.forEach((l, k) => {
    if (l < losses[best]) best = k
  })
  return { bandwidth: bandwidths[best], losses }
}
