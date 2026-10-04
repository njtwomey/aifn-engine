import type { Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { conv, type ConvMethod } from 'aifn-compute/foundation/convolution'
import { toSequence, vectorOf, type Data } from './input'
import { mean, requireNonEmpty, requireSameLength, type SampleOption } from './descriptive'
import { DomainError } from 'aifn-compute/foundation/errors'

// ---------------------------------------------------------------------------------------------------------------------
// Running moments (Welford 1962; merging by Chan, Golub and LeVeque 1983).

/**
 * Running moments of a stream of values: the count, the mean and M₂ = Σ(xᵢ − x̄)². A plain, immutable record: update
 * it with `momentsPush` and combine two with `momentsMerge`.
 */
export type Moments = { readonly count: number; readonly mean: number; readonly m2: number }

/** Moments of no values. */
export const emptyMoments: Moments = { count: 0, mean: 0, m2: 0 }

/** Adds one value with Welford's update, which stays accurate when the mean is large relative to the spread. */
export function momentsPush(m: Moments, x: number): Moments {
  const count = m.count + 1
  const delta = x - m.mean
  const meanNext = m.mean + delta / count
  return { count, mean: meanNext, m2: m.m2 + delta * (x - meanNext) }
}

/** The moments of the union of two sets of values (Chan, Golub and LeVeque 1983), e.g. to merge per-chunk results. */
export function momentsMerge(a: Moments, b: Moments): Moments {
  if (a.count === 0) return b
  if (b.count === 0) return a
  const count = a.count + b.count
  const delta = b.mean - a.mean
  return {
    count,
    mean: a.mean + (delta * b.count) / count,
    m2: a.m2 + b.m2 + (delta * delta * a.count * b.count) / count,
  }
}

/** The variance from running moments: M₂ / n, or M₂ / (n − 1) with `sample`. */
export function momentsVariance(m: Moments, options: SampleOption = {}): number {
  return m.m2 / (m.count - (options.sample ? 1 : 0))
}

/** The running mean: entry i is the mean of x₀ … xᵢ. */
export function runningMean(xData: Data): Tensor {
  const x = toSequence(xData, 'runningMean')
  const out = new Float64Array(x.length)
  let m = emptyMoments
  for (let i = 0; i < x.length; i++) out[i] = (m = momentsPush(m, x[i])).mean
  return vectorOf(out)
}

/**
 * The running variance by Welford's method: entry i is the variance of x₀ … xᵢ (population by default; with
 * `sample`, entry 0 is NaN).
 */
export function runningVariance(xData: Data, options: SampleOption = {}): Tensor {
  const x = toSequence(xData, 'runningVariance')
  const out = new Float64Array(x.length)
  let m = emptyMoments
  for (let i = 0; i < x.length; i++) out[i] = momentsVariance((m = momentsPush(m, x[i])), options)
  return vectorOf(out)
}

/**
 * Lagged products r(k) = Σₜ aₜ b_{t+k} for k = −maxLag … maxLag (index k + maxLag): the cross-correlation of b with
 * a, zero-padded by maxLag on each side, i.e. one `conv` of the convolution family (no flip; the FFT or direct kernel).
 */
function laggedProducts(a: Float64Array, b: Float64Array, maxLag: number, method: ConvMethod): Float64Array {
  const r = conv(fromData(b), fromData(a), { flip: false, padding: maxLag, method }) as Tensor
  return Float64Array.from(dense.data(r))
}

/** Options for autocovariance and cross-covariance. */
export type LagOptions = {
  /** Largest lag; default n − 1 (every lag). */
  maxLag?: Size
  /** `direct` sums each lag (O(n·maxLag)); `fft` uses an FFT (O(n log n)); `auto` (default) picks the cheaper. */
  method?: ConvMethod
  /** Divide lag k by n − k instead of n. The default (n) is the usual estimator, positive semi-definite. */
  adjusted?: boolean
  /** Subtract the mean first (default true). */
  demean?: boolean
}

function lagged(x: ArrayLike<number>, y: ArrayLike<number>, options: LagOptions): Float64Array {
  const n = x.length
  const maxLag = Math.min(options.maxLag ?? n - 1, n - 1)
  if (!(maxLag >= 0) || !Number.isInteger(maxLag))
    throw new DomainError('stats', 'stats: maxLag must be a non-negative integer')
  const demean = options.demean ?? true
  const mx = demean ? mean(x) : 0
  const my = demean ? mean(y) : 0
  const a = Float64Array.from(x, (v) => v - mx)
  const b = Float64Array.from(y, (v) => v - my)
  const sums = laggedProducts(a, b, maxLag, options.method ?? 'auto')
  for (let j = 0; j < sums.length; j++) sums[j] /= options.adjusted ? n - Math.abs(j - maxLag) : n
  return sums
}

function autocovarianceOf(x: ArrayLike<number>, options: LagOptions): Float64Array {
  requireNonEmpty(x, 'autocovariance')
  const both = lagged(x, x, options)
  return both.slice((both.length - 1) / 2)
}

/** The variance (÷ n) of a sequence, as the autocovariance at lag 0. */
const lagZero = (x: ArrayLike<number>, options: LagOptions) =>
  autocovarianceOf(x, { ...options, maxLag: 0, adjusted: false })[0]

/**
 * The sample autocovariance γ̂(k) = (1/n) Σ_{t=0}^{n−1−k} (xₜ − x̄)(x_{t+k} − x̄) for k = 0 … maxLag (Box and Jenkins
 * 1976, §2.1.5), as a rank-1 tensor. The FFT method gives the same values to rounding.
 */
export function autocovariance(xData: Data, options: LagOptions = {}): Tensor {
  return vectorOf(autocovarianceOf(toSequence(xData, 'autocovariance'), options))
}

/**
 * The sample autocorrelation ρ̂(k) = γ̂(k) / γ̂(0) for k = 0 … maxLag, so ρ̂(0) = 1, as a rank-1 tensor; by FFT for
 * long sequences (`method`). NaN when x is constant.
 */
export function autocorrelation(xData: Data, options: LagOptions = {}): Tensor {
  const x = toSequence(xData, 'autocorrelation')
  const g = autocovarianceOf(x, options)
  const g0 = options.adjusted ? lagZero(x, options) : g[0]
  return vectorOf(g.map((v) => v / g0))
}

/**
 * The sample cross-covariance c(k) = (1/n) Σₜ (xₜ − x̄)(y_{t+k} − ȳ) for k = −maxLag … maxLag; entry j is lag
 * j − maxLag. A peak at positive k means y follows x by k steps.
 */
export function crossCovariance(xData: Data, yData: Data, options: LagOptions = {}): { lags: Tensor; values: Tensor } {
  const x = toSequence(xData, 'crossCovariance')
  const y = toSequence(yData, 'crossCovariance')
  const { lags, values } = crossCovarianceOf(x, y, options)
  return { lags: vectorOf(lags), values: vectorOf(values) }
}

function crossCovarianceOf(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  options: LagOptions,
): { lags: Int32Array; values: Float64Array } {
  requireNonEmpty(x, 'crossCovariance')
  requireSameLength(x, y, 'crossCovariance')
  const values = lagged(x, y, options)
  const maxLag = (values.length - 1) / 2
  return { lags: Int32Array.from({ length: values.length }, (_, j) => j - maxLag), values }
}

/**
 * The sample cross-correlation r(k) = c(k) / (σ̂ₓ σ̂ᵧ) with population standard deviations, for k = −maxLag … maxLag
 * (entry j is lag j − maxLag). A peak at positive k means y follows x by k steps.
 */
export function crossCorrelation(xData: Data, yData: Data, options: LagOptions = {}): { lags: Tensor; values: Tensor } {
  const x = toSequence(xData, 'crossCorrelation')
  const y = toSequence(yData, 'crossCorrelation')
  const { lags, values } = crossCovarianceOf(x, y, options)
  const sx = Math.sqrt(lagZero(x, options))
  const sy = Math.sqrt(lagZero(y, options))
  return { lags: vectorOf(lags), values: vectorOf(values.map((v) => v / (sx * sy))) }
}
