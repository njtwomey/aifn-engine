/**
 * Statistics of a sequence in order: running moments (Welford's update and the merge of Chan, Golub and LeVeque),
 * running means and variances, and the sample autocovariance, autocorrelation, cross-covariance and cross-correlation
 * at a range of lags, computed directly or by FFT through `aifn-compute/foundation/convolution`.
 *
 * Sequences are arrays or rank-1 tensors. The lagged estimators divide by $n$ by default (the positive semi-definite
 * estimator of Box and Jenkins), or by $n - k$ with `adjusted`.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { conv, type ConvMethod } from 'aifn-compute/foundation/convolution'
import { toSequence, vectorOf, type Data } from './input'
import { mean, requireNonEmpty, requireSameLength, type SampleOption } from './descriptive'
import { DomainError } from 'aifn-compute/foundation/errors'

// ---------------------------------------------------------------------------------------------------------------------
// Running moments (Welford 1962; merging by Chan, Golub and LeVeque 1983).

/**
 * Running moments of a stream of values: the `count` $n$, the `mean` $\bar{x}$ and `m2`,
 * $M_2 = \sum_i (x_i - \bar{x})^2$. A plain, immutable record: update it with `momentsPush` and combine two with
 * `momentsMerge`.
 */
export type Moments = { readonly count: number; readonly mean: number; readonly m2: number }

/** Moments of no values. */
export const emptyMoments: Moments = { count: 0, mean: 0, m2: 0 }

/**
 * Adds one value with Welford's update, which stays accurate when the mean is large relative to the spread.
 *
 * @param m The moments so far (`emptyMoments` to start); not modified.
 * @param x The new value.
 * @returns New moments that include `x`.
 *
 * @example Push values one at a time
 * let m = emptyMoments
 * for (const x of [2, 4, 4, 4, 5, 5, 7, 9]) m = momentsPush(m, x)
 * print('moments =', m)
 * print('variance =', momentsVariance(m))
 */
export function momentsPush(m: Moments, x: number): Moments {
  const count = m.count + 1
  const delta = x - m.mean
  const meanNext = m.mean + delta / count
  return { count, mean: meanNext, m2: m.m2 + delta * (x - meanNext) }
}

/**
 * The moments of the union of two sets of values (Chan, Golub and LeVeque 1983), e.g. to merge per-chunk results.
 *
 * @param a The moments of the first set; not modified.
 * @param b The moments of the second set; not modified.
 * @returns The moments of both together (`a` or `b` itself when the other is empty).
 *
 * @example Two chunks merge to the moments of the whole
 * const push = (xs) => xs.reduce(momentsPush, emptyMoments)
 * const merged = momentsMerge(push([1, 2, 3]), push([4, 5]))
 * print('merged =', merged)
 * print('all at once =', push([1, 2, 3, 4, 5]))
 */
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

/**
 * The variance from running moments: $M_2 / n$, or $M_2 / (n - 1)$ with `sample`.
 *
 * @param m The moments.
 * @param options `sample` for the $n - 1$ divisor.
 * @returns The variance (NaN for no values, or for one value with `sample`).
 *
 * @example Population and sample variance of a stream
 * const m = [1, 2, 3, 4, 5].reduce(momentsPush, emptyMoments)
 * print('population =', momentsVariance(m))
 * print('sample =', momentsVariance(m, { sample: true }))
 */
export function momentsVariance(m: Moments, options: SampleOption = {}): number {
  return m.m2 / (m.count - (options.sample ? 1 : 0))
}

/**
 * The running mean: entry $i$ is the mean of $x_0, \dots, x_i$.
 *
 * @param xData The sequence: an array or a rank-1 tensor.
 * @returns The running means, one per value.
 *
 * @example Means of growing prefixes
 * print('running mean =', runningMean([2, 4, 6, 8]))
 */
export function runningMean(xData: Data): Tensor {
  const x = toSequence(xData, 'runningMean')
  const out = new Float64Array(x.length)
  let m = emptyMoments
  for (let i = 0; i < x.length; i++) out[i] = (m = momentsPush(m, x[i])).mean
  return vectorOf(out)
}

/**
 * The running variance by Welford's method: entry $i$ is the variance of $x_0, \dots, x_i$ (population by default;
 * with `sample`, entry 0 is NaN).
 *
 * @param xData The sequence: an array or a rank-1 tensor.
 * @param options `sample` for the $n - 1$ divisor.
 * @returns The running variances, one per value.
 *
 * @example Population and sample forms
 * print('population =', runningVariance([2, 4, 6, 8]))
 * print('sample =', runningVariance([2, 4, 6, 8], { sample: true }))
 */
export function runningVariance(xData: Data, options: SampleOption = {}): Tensor {
  const x = toSequence(xData, 'runningVariance')
  const out = new Float64Array(x.length)
  let m = emptyMoments
  for (let i = 0; i < x.length; i++) out[i] = momentsVariance((m = momentsPush(m, x[i])), options)
  return vectorOf(out)
}

/**
 * Lagged products $r(k) = \sum_t a_t b_{t+k}$ for $k = -\text{maxLag}, \dots, \text{maxLag}$ (index
 * $k + \text{maxLag}$): the cross-correlation of $\bvec$ with $\avec$, zero-padded by maxLag on each side, i.e. one
 * `conv` of the convolution family (no flip; the FFT or direct kernel).
 *
 * @param a The first sequence (already centred, if it is to be).
 * @param b The second sequence, of the same length.
 * @param maxLag The largest lag, at most $n - 1$.
 * @param method The `conv` method: `direct`, `fft` or `auto`.
 * @returns The $2 \cdot \text{maxLag} + 1$ sums, from lag $-\text{maxLag}$.
 */
function laggedProducts(a: Float64Array, b: Float64Array, maxLag: number, method: ConvMethod): Float64Array {
  const r = conv(fromData(b), fromData(a), { flip: false, padding: maxLag, method }) as Tensor
  return Float64Array.from(dense.data(r))
}

/** Options for autocovariance and cross-covariance. */
export type LagOptions = {
  /**
   * Largest lag, a non-negative integer; default $n - 1$ (every lag), and a larger one is cut to $n - 1$.
   */
  maxLag?: Size
  /**
   * `direct` sums each lag ($O(n \cdot \text{maxLag})$); `fft` uses an FFT ($O(n \log n)$); `auto` (default) picks the
   * cheaper.
   */
  method?: ConvMethod
  /** Divide lag $k$ by $n - k$ instead of $n$. The default ($n$) is the usual estimator, positive semi-definite. */
  adjusted?: boolean
  /** Subtract the mean first (default true). */
  demean?: boolean
}

/**
 * The lagged covariance sums of two sequences at lags $-\text{maxLag}, \dots, \text{maxLag}$, centred and divided as
 * the options say. Throws `DomainError` for a negative or fractional `maxLag`.
 *
 * @param x The first sequence ($n$ values).
 * @param y The second sequence, of the same length (`x` again for the autocovariance).
 * @param options The largest lag, the method, the divisor and whether to subtract the means.
 * @returns The $2 \cdot \text{maxLag} + 1$ values, entry $j$ at lag $j - \text{maxLag}$.
 */
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

/**
 * The autocovariance of a sequence at lags $0, \dots, \text{maxLag}$. Throws `DomainError` when it is empty.
 *
 * @param x The sequence.
 * @param options The largest lag, the method, the divisor and whether to subtract the mean.
 * @returns $\hat{\gamma}(k)$ for each lag $k$ from 0.
 */
function autocovarianceOf(x: ArrayLike<number>, options: LagOptions): Float64Array {
  requireNonEmpty(x, 'autocovariance')
  const both = lagged(x, x, options)
  return both.slice((both.length - 1) / 2)
}

/**
 * The variance (divisor $n$) of a sequence, as the autocovariance at lag 0 (the mean square when `demean` is false).
 *
 * @param x The sequence.
 * @param options The options of the caller; only `method` and `demean` are used.
 * @returns $\hat{\gamma}(0)$.
 */
const lagZero = (x: ArrayLike<number>, options: LagOptions) =>
  autocovarianceOf(x, { ...options, maxLag: 0, adjusted: false })[0]

/**
 * The sample autocovariance $\hat{\gamma}(k) = \frac{1}{n} \sum_{t=0}^{n-1-k} (x_t - \bar{x})(x_{t+k} - \bar{x})$ for
 * $k = 0, \dots, \text{maxLag}$ (Box and Jenkins 1976, §2.1.5), as a rank-1 tensor. The FFT method gives the same
 * values to rounding. Throws `DomainError` for an empty sequence or a bad `maxLag`.
 *
 * @param xData The sequence: an array or a rank-1 tensor.
 * @param options The largest lag, the method, `adjusted` (divide by $n - k$) and `demean`.
 * @returns $\hat{\gamma}(k)$ for $k = 0, \dots, \text{maxLag}$.
 *
 * @example Every lag of a short ramp
 * // Lag 0 is the population variance, 2.
 * print('gamma =', autocovariance([1, 2, 3, 4, 5]))
 * print('adjusted, to lag 2 =', autocovariance([1, 2, 3, 4, 5], { maxLag: 2, adjusted: true }))
 */
export function autocovariance(xData: Data, options: LagOptions = {}): Tensor {
  return vectorOf(autocovarianceOf(toSequence(xData, 'autocovariance'), options))
}

/**
 * The sample autocorrelation $\hat{\rho}(k) = \hat{\gamma}(k) / \hat{\gamma}(0)$ for $k = 0, \dots, \text{maxLag}$,
 * so $\hat{\rho}(0) = 1$, as a rank-1 tensor; by FFT for long sequences (`method`). NaN when $\xvec$ is constant.
 * With `adjusted`, only the numerator is divided by $n - k$.
 *
 * @param xData The sequence: an array or a rank-1 tensor.
 * @param options The largest lag, the method, `adjusted` and `demean`.
 * @returns $\hat{\rho}(k)$ for $k = 0, \dots, \text{maxLag}$.
 *
 * @example A ramp, and an alternating sequence
 * print('ramp =', autocorrelation([1, 2, 3, 4, 5]))
 * print('alternating =', autocorrelation([1, -1, 1, -1, 1, -1], { maxLag: 2 }))
 */
export function autocorrelation(xData: Data, options: LagOptions = {}): Tensor {
  const x = toSequence(xData, 'autocorrelation')
  const g = autocovarianceOf(x, options)
  const g0 = options.adjusted ? lagZero(x, options) : g[0]
  return vectorOf(g.map((v) => v / g0))
}

/**
 * The sample cross-covariance $c(k) = \frac{1}{n} \sum_t (x_t - \bar{x})(y_{t+k} - \bar{y})$ for
 * $k = -\text{maxLag}, \dots, \text{maxLag}$; entry $j$ is lag $j - \text{maxLag}$. A peak at positive $k$ means
 * $\yvec$ follows $\xvec$ by $k$ steps. Throws `ShapeError` when the lengths differ and `DomainError` when they are
 * empty or `maxLag` is bad.
 *
 * @param xData The first sequence: an array or a rank-1 tensor.
 * @param yData The second sequence, of the same length.
 * @param options The largest lag, the method, `adjusted` (divide by $n - \lvert k \rvert$) and `demean`.
 * @returns The `lags` (int32) and the `values` $c(k)$ at each.
 *
 * @example y is x delayed by one step
 * const { lags, values } = crossCovariance([0, 1, 3, 1, 0, 0], [0, 0, 1, 3, 1, 0], { maxLag: 2 })
 * print('lags =', lags)
 * print('c =', values)
 */
export function crossCovariance(xData: Data, yData: Data, options: LagOptions = {}): { lags: Tensor; values: Tensor } {
  const x = toSequence(xData, 'crossCovariance')
  const y = toSequence(yData, 'crossCovariance')
  const { lags, values } = crossCovarianceOf(x, y, options)
  return { lags: vectorOf(lags), values: vectorOf(values) }
}

/**
 * The cross-covariance of two sequences with its lags. Throws `DomainError` when empty and `ShapeError` when the
 * lengths differ.
 *
 * @param x The first sequence.
 * @param y The second sequence, of the same length.
 * @param options The largest lag, the method, the divisor and whether to subtract the means.
 * @returns The `lags` $-\text{maxLag}, \dots, \text{maxLag}$ and the `values` at each.
 */
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
 * The sample cross-correlation $r(k) = c(k) / (\hat{\sigma}_x \hat{\sigma}_y)$ with population standard
 * deviations (root mean squares when `demean` is false), for $k = -\text{maxLag}, \dots, \text{maxLag}$ (entry $j$
 * is lag $j - \text{maxLag}$). A peak at positive $k$ means $\yvec$ follows $\xvec$ by $k$ steps. Throws as
 * `crossCovariance` does.
 *
 * @param xData The first sequence: an array or a rank-1 tensor.
 * @param yData The second sequence, of the same length.
 * @param options The largest lag, the method, `adjusted` (for $c(k)$ only) and `demean`.
 * @returns The `lags` (int32) and the `values` $r(k)$ at each.
 *
 * @example The peak is at lag +1: y follows x by one step
 * const { lags, values } = crossCorrelation([0, 1, 3, 1, 0, 0], [0, 0, 1, 3, 1, 0], { maxLag: 2 })
 * print('lags =', lags)
 * print('r =', values)
 */
export function crossCorrelation(xData: Data, yData: Data, options: LagOptions = {}): { lags: Tensor; values: Tensor } {
  const x = toSequence(xData, 'crossCorrelation')
  const y = toSequence(yData, 'crossCorrelation')
  const { lags, values } = crossCovarianceOf(x, y, options)
  const sx = Math.sqrt(lagZero(x, options))
  const sy = Math.sqrt(lagZero(y, options))
  return { lags: vectorOf(lags), values: vectorOf(values.map((v) => v / (sx * sy))) }
}
