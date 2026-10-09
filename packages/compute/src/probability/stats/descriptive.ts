/**
 * Descriptive statistics: moments, extremes, the mode, covariance and correlation, and standardisation.
 *
 * Every function takes a `number[]`, a typed array or an `aifn-compute/foundation/tensor` tensor. Reductions (`sum`,
 * `mean`, `variance`, `standardDeviation`, `min`, `max`, `range`, `skewness`, `kurtosis`) reduce every element of a
 * tensor to a number, or take `{ axis, keepDims }` to reduce a tensor along one axis and return a tensor; pairwise and
 * elementwise functions need rank-1 tensors. Variances default to the population form (divide by $n$, like `np.var`);
 * pass `{ sample: true }` for the unbiased sample form (divide by $n - 1$, like `np.var(ddof=1)`). The plain `sum`,
 * `mean`, `variance`, `min` and `max` here serve the module's other files; the public ones are
 * `aifn-compute/foundation/tensor`'s.
 */

import type { Scalar } from 'aifn-compute/foundation/contracts'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { allValues, reduce, toSequence, vectorOf, type AxisOption, type Data } from './input'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options for moments that have a population and a sample form. */
export type SampleOption = {
  /** Divide by $n - 1$ (the unbiased sample variance) instead of $n$ (the population variance). Default false. */
  sample?: boolean
}

/**
 * Throws `DomainError` when the values are empty.
 *
 * @param x The values to check.
 * @param what The caller's name for error messages.
 */
function requireNonEmpty(x: ArrayLike<number>, what: string) {
  if (x.length === 0) throw new DomainError('stats', `stats: ${what} of an empty array`)
}

/**
 * Throws `ShapeError` when two sequences differ in length.
 *
 * @param x The first sequence.
 * @param y The second sequence, which must have the length of `x`.
 * @param what The caller's name for error messages.
 */
function requireSameLength(x: ArrayLike<number>, y: ArrayLike<number>, what: string) {
  if (x.length !== y.length)
    throw new ShapeError('stats', `stats: ${what} needs arrays of equal length (${x.length} and ${y.length})`)
}

/** A reduction's options with the axis given: the result is a tensor. */
export type Along<O = unknown> = O & AxisOption & { axis: number }
/** A reduction's options without an axis: the result is a number. */
export type Whole<O = unknown> = O & { axis?: undefined; keepDims?: boolean }

/**
 * Neumaier's compensated sum: exact to about one rounding for ill-conditioned sums (Neumaier 1974).
 *
 * @param x The values to add (0 when empty).
 * @returns Their sum.
 */
function sumOf(x: ArrayLike<number>): number {
  let s = 0
  let c = 0
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    const t = s + v
    c += Math.abs(s) >= Math.abs(v) ? s - t + v : v - t + s
    s = t
  }
  return s + c
}

/**
 * Neumaier's compensated sum (Neumaier 1974): exact to about one rounding for ill-conditioned sums. Over every element,
 * or along `axis` of a tensor.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options The axis of a tensor to sum along (none: every element) and `keepDims`.
 * @returns The sum, or a tensor of sums along the axis.
 */
export function sum(x: Data, options?: Whole): number
export function sum(x: Tensor, options: Along): Tensor
export function sum(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, sumOf, 'sum')
}

/**
 * The arithmetic mean by the compensated sum. Throws `DomainError` when empty.
 *
 * @param x The values.
 * @returns Their mean.
 */
function meanOf(x: ArrayLike<number>): number {
  requireNonEmpty(x, 'mean')
  return sumOf(x) / x.length
}

/**
 * The arithmetic mean, over every element or along `axis` of a tensor. Throws `DomainError` on an empty array.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options The axis of a tensor to average along (none: every element) and `keepDims`.
 * @returns The mean, or a tensor of means along the axis.
 */
export function mean(x: Data, options?: Whole): number
export function mean(x: Tensor, options: Along): Tensor
export function mean(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, meanOf, 'mean')
}

/**
 * The two-pass variance of the values, population or sample. Throws `DomainError` when empty.
 *
 * @param x The values.
 * @param options Whether to divide by $n - 1$ instead of $n$.
 * @returns The variance (NaN for the sample variance of one value).
 */
function varianceOf(x: ArrayLike<number>, options: SampleOption): number {
  requireNonEmpty(x, 'variance')
  const m = meanOf(x)
  let ss = 0
  for (let i = 0; i < x.length; i++) ss += (x[i] - m) ** 2
  return ss / (x.length - (options.sample ? 1 : 0))
}

/**
 * The variance: $\sum_i (x_i - \bar{x})^2 / n$, or $/ (n - 1)$ with `sample`. Two-pass (the mean first), which is
 * accurate when the mean is large relative to the spread. A sample variance of one value is NaN ($0/0$), as in numpy.
 * Over every element, or along `axis` of a tensor. Throws `DomainError` on an empty array.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options `sample` for the $n - 1$ divisor, and the axis of a tensor to reduce along with `keepDims`.
 * @returns The variance, or a tensor of variances along the axis.
 */
export function variance(x: Data, options?: Whole<SampleOption>): number
export function variance(x: Tensor, options: Along<SampleOption>): Tensor
export function variance(x: Data, options: SampleOption & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => varianceOf(v, options), 'variance')
}

/**
 * The standard deviation: the square root of `variance` (population by default, `{ sample: true }` for the $n - 1$
 * divisor). Over every element, or along `axis` of a tensor. Throws `DomainError` on an empty array.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options `sample` for the $n - 1$ divisor, and the axis of a tensor to reduce along with `keepDims`.
 * @returns The standard deviation, or a tensor of them along the axis.
 *
 * @example Population and sample forms, as np.std(x) and np.std(x, ddof=1)
 * const x = [2, 4, 4, 4, 5, 5, 7, 9]
 * print('population sd =', standardDeviation(x))
 * print('sample sd =', standardDeviation(x, { sample: true }))
 *
 * @example Along an axis of a tensor
 * const m = tensor([[1, 2, 3], [2, 4, 6]])
 * print('sd of each row =', standardDeviation(m, { axis: 1 }))
 * print('sd of each column =', standardDeviation(m, { axis: 0 }))
 */
export function standardDeviation(x: Data, options?: Whole<SampleOption>): number
export function standardDeviation(x: Tensor, options: Along<SampleOption>): Tensor
export function standardDeviation(x: Data, options: SampleOption & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => Math.sqrt(varianceOf(v, options)), 'standardDeviation')
}

/**
 * The weighted mean $\sum_i w_i x_i / \sum_i w_i$ of a sequence $\xvec$ (length $n$). Throws `DomainError` for an
 * empty sequence, a negative weight or weights that sum to 0, and `ShapeError` when the lengths differ.
 *
 * @param xs The values $x_i$: an array or a rank-1 tensor.
 * @param ws The weights $w_i$, one per value: non-negative, with a positive sum. Only their ratios matter.
 * @returns The weighted mean.
 *
 * @example Weights as repeat counts
 * // The same as the plain mean of [1, 2, 3, 3].
 * print('weighted mean =', weightedMean([1, 2, 3], [1, 1, 2]))
 */
export function weightedMean(xs: Data, ws: Data): number {
  const x = toSequence(xs, 'weightedMean')
  const w = toSequence(ws, 'weightedMean')
  requireNonEmpty(x, 'weightedMean')
  requireSameLength(x, w, 'weightedMean')
  let sw = 0
  let swx = 0
  for (let i = 0; i < x.length; i++) {
    if (w[i] < 0) throw new DomainError('stats', 'stats: weights must be non-negative')
    sw += w[i]
    swx += w[i] * x[i]
  }
  if (!(sw > 0)) throw new DomainError('stats', 'stats: weights must have a positive sum')
  return swx / sw
}

/**
 * The weighted variance $S / D$ where $S = \sum_i w_i (x_i - \bar{x}_w)^2$ and $\bar{x}_w$ is the weighted mean.
 * The divisor depends on what the weights mean:
 * - `population` (default): $D = \sum_i w_i$. Weights as a distribution; `np.cov(x, aweights=w, ddof=0)`.
 * - `frequency`: $D = \sum_i w_i - 1$. Integer weights that count repeats; `np.cov(x, fweights=w)`.
 * - `reliability`: $D = \sum_i w_i - \sum_i w_i^2 / \sum_i w_i$. Unbiased for weights as relative precisions;
 *   `np.cov(x, aweights=w, ddof=1)`.
 *
 * Throws as `weightedMean` does.
 *
 * @param xs The values $x_i$: an array or a rank-1 tensor.
 * @param ws The weights $w_i$, one per value: non-negative, with a positive sum.
 * @param options What the weights mean, which sets the divisor.
 * @param options.weights `'population'`, `'frequency'` or `'reliability'`, as above (default `'population'`).
 * @returns The weighted variance.
 *
 * @example The three divisors, as numpy's np.cov with weights
 * const x = [1, 2, 3]
 * const w = [1, 1, 2]
 * print('population =', weightedVariance(x, w))
 * print('frequency =', weightedVariance(x, w, { weights: 'frequency' }))
 * print('reliability =', weightedVariance(x, w, { weights: 'reliability' }))
 * // Frequency weights are repeats: the sample variance of [1, 2, 3, 3].
 * print('sample variance of the repeats =', standardDeviation([1, 2, 3, 3], { sample: true }) ** 2)
 */
export function weightedVariance(
  xs: Data,
  ws: Data,
  options: { weights?: 'population' | 'frequency' | 'reliability' } = {},
): number {
  const x = toSequence(xs, 'weightedVariance')
  const w = toSequence(ws, 'weightedVariance')
  const m = weightedMean(x, w)
  let sw = 0
  let sw2 = 0
  let ss = 0
  for (let i = 0; i < x.length; i++) {
    sw += w[i]
    sw2 += w[i] * w[i]
    ss += w[i] * (x[i] - m) ** 2
  }
  const kind = options.weights ?? 'population'
  const divisor = kind === 'population' ? sw : kind === 'frequency' ? sw - 1 : sw - sw2 / sw
  return ss / divisor
}

/**
 * The smallest value, or NaN if any value is NaN. Throws `DomainError` when empty.
 *
 * @param x The values.
 * @returns The minimum.
 */
function minOf(x: ArrayLike<number>): number {
  requireNonEmpty(x, 'min')
  let m = Infinity
  for (let i = 0; i < x.length; i++) {
    if (Number.isNaN(x[i])) return NaN
    if (x[i] < m) m = x[i]
  }
  return m
}

/**
 * The largest value, or NaN if any value is NaN. Throws `DomainError` when empty.
 *
 * @param x The values.
 * @returns The maximum.
 */
function maxOf(x: ArrayLike<number>): number {
  requireNonEmpty(x, 'max')
  let m = -Infinity
  for (let i = 0; i < x.length; i++) {
    if (Number.isNaN(x[i])) return NaN
    if (x[i] > m) m = x[i]
  }
  return m
}

/**
 * The smallest value; NaN if any value is NaN (as `np.min`). Throws `DomainError` on an empty array. Over every
 * element, or along `axis` of a tensor.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options The axis of a tensor to reduce along (none: every element) and `keepDims`.
 * @returns The minimum, or a tensor of minima along the axis.
 */
export function min(x: Data, options?: Whole): number
export function min(x: Tensor, options: Along): Tensor
export function min(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, minOf, 'min')
}

/**
 * The largest value; NaN if any value is NaN (as `np.max`). Throws `DomainError` on an empty array. Over every element,
 * or along `axis` of a tensor.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options The axis of a tensor to reduce along (none: every element) and `keepDims`.
 * @returns The maximum, or a tensor of maxima along the axis.
 */
export function max(x: Data, options?: Whole): number
export function max(x: Tensor, options: Along): Tensor
export function max(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, maxOf, 'max')
}

/**
 * `[min, max]` over every element, e.g. for an axis range. Both are NaN if any value is NaN; throws `DomainError` on an
 * empty array.
 *
 * @param x The data: an array, or a tensor of any rank (every element counts).
 * @returns The smallest and the largest value.
 *
 * @example The extent of a tensor's elements
 * print('extent =', extent(tensor([[3, -1], [7, 2]])))
 */
export function extent(x: Data): [number, number] {
  const v = allValues(x)
  return [minOf(v), maxOf(v)]
}

/**
 * The range $\max - \min$ (`np.ptp`), over every element or along `axis` of a tensor. Throws `DomainError` on an empty
 * array.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options The axis of a tensor to reduce along (none: every element) and `keepDims`.
 * @returns The range, or a tensor of ranges along the axis.
 *
 * @example Over everything, and along each axis
 * const m = tensor([[1, 5], [2, 9]])
 * print('range =', range(m))
 * print('range of each column =', range(m, { axis: 0 }))
 * print('range of each row, kept as a column =', range(m, { axis: 1, keepDims: true }))
 */
export function range(x: Data, options?: Whole): number
export function range(x: Tensor, options: Along): Tensor
export function range(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => maxOf(v) - minOf(v), 'range')
}

/**
 * The most frequent value and its count. Ties go to the smallest value, as in `scipy.stats.mode`. Values are compared
 * exactly, so this is meant for discrete data. A tensor contributes every element. Throws `DomainError` on an empty
 * array.
 *
 * @param data The data: an array, or a tensor of any rank.
 * @returns The modal `value` and its `count`.
 *
 * @example A tie goes to the smaller value
 * // 2 and 3 both occur twice.
 * print('mode =', mode([1, 2, 2, 3, 3, 5]))
 */
export function mode(data: Data): { value: number; count: number } {
  const x = allValues(data)
  requireNonEmpty(x, 'mode')
  const counts = new Map<number, number>()
  for (let i = 0; i < x.length; i++) counts.set(x[i], (counts.get(x[i]) ?? 0) + 1)
  let value = NaN
  let count = 0
  for (const [v, c] of counts) {
    if (c > count || (c === count && v < value)) {
      value = v
      count = c
    }
  }
  return { value, count }
}

/**
 * Options of `kurtosis`. `excess` (default true) subtracts 3, so that a Gaussian scores 0; `biasCorrected` (default
 * false) gives the estimator that is unbiased under normality.
 */
export type KurtosisOptions = { excess?: boolean; biasCorrected?: boolean }

/**
 * The $k$-th central moment $\frac{1}{n} \sum_i (x_i - m)^k$ about a given centre $m$.
 *
 * @param x The values ($n$ of them).
 * @param k The order of the moment.
 * @param m The centre, normally the mean $\bar{x}$.
 * @returns The moment.
 */
function centralMoment(x: ArrayLike<number>, k: number, m: number): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += (x[i] - m) ** k
  return s / x.length
}

/**
 * The skewness $g_1 = m_3 / m_2^{3/2}$ from the central moments $m_k$. With `biasCorrected`, the adjusted
 * Fisher–Pearson coefficient $G_1 = g_1 \sqrt{n(n - 1)} / (n - 2)$ (Joanes and Gill 1998). Matches
 * `scipy.stats.skew(x, bias=...)`. NaN when the data are constant. Over every element, or along `axis` of a tensor.
 * Throws `DomainError` on an empty array.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options `biasCorrected` (default false) for $G_1$, and the axis of a tensor to reduce along with `keepDims`.
 * @returns The skewness, or a tensor of it along the axis.
 *
 * @example A long right tail, as scipy.stats.skew
 * const x = [2, 4, 4, 4, 5, 5, 7, 9]
 * print('g1 =', skewness(x))
 * print('G1 =', skewness(x, { biasCorrected: true }))
 * print('symmetric data =', skewness([1, 2, 3, 4, 5]))
 */
export function skewness(x: Data, options?: Whole<{ biasCorrected?: boolean }>): number
export function skewness(x: Tensor, options: Along<{ biasCorrected?: boolean }>): Tensor
export function skewness(x: Data, options: { biasCorrected?: boolean } & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => skewnessOf(v, options), 'skewness')
}

/**
 * The skewness of one sequence of values, $g_1$ or $G_1$.
 *
 * @param x The values ($n$ of them, at least one).
 * @param options `biasCorrected` for $G_1$.
 * @returns The skewness; NaN when the values are constant.
 */
function skewnessOf(x: ArrayLike<number>, options: { biasCorrected?: boolean }): number {
  const n = x.length
  const m = meanOf(x)
  const m2 = centralMoment(x, 2, m)
  const m3 = centralMoment(x, 3, m)
  const g1 = m2 === 0 ? NaN : m3 / m2 ** 1.5
  if (!options.biasCorrected) return g1
  return (g1 * Math.sqrt(n * (n - 1))) / (n - 2)
}

/**
 * The kurtosis $m_4 / m_2^2$, minus 3 by default (`excess`, so a Gaussian scores 0). With `biasCorrected`, the unbiased
 * estimator under normality $G_2 = ((n + 1) g_2 + 6)(n - 1) / ((n - 2)(n - 3))$ of the excess $g_2$ (Joanes and Gill
 * 1998), with 3 added back when `excess` is false. Matches
 * `scipy.stats.kurtosis(x, fisher=excess, bias=not biasCorrected)`. NaN when the data are constant. Over every element,
 * or along `axis` of a tensor. Throws `DomainError` on an empty array.
 *
 * @param x The data: an array, or a tensor of any rank.
 * @param options `excess` and `biasCorrected` (see `KurtosisOptions`), and the axis of a tensor to reduce along with
 *   `keepDims`.
 * @returns The kurtosis, or a tensor of it along the axis.
 *
 * @example Excess, plain and bias-corrected, as scipy.stats.kurtosis
 * const x = [2, 4, 4, 4, 5, 5, 7, 9]
 * print('excess g2 =', kurtosis(x))
 * print('m4 / m2^2 =', kurtosis(x, { excess: false }))
 * print('G2 =', kurtosis(x, { biasCorrected: true }))
 */
export function kurtosis(x: Data, options?: Whole<KurtosisOptions>): number
export function kurtosis(x: Tensor, options: Along<KurtosisOptions>): Tensor
export function kurtosis(x: Data, options: KurtosisOptions & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => kurtosisOf(v, options), 'kurtosis')
}

/**
 * The kurtosis of one sequence of values, as `kurtosis` describes.
 *
 * @param x The values ($n$ of them, at least one).
 * @param options Whether to subtract 3 (`excess`) and whether to apply the bias correction.
 * @returns The kurtosis; NaN when the values are constant.
 */
function kurtosisOf(x: ArrayLike<number>, options: KurtosisOptions): number {
  const n = x.length
  const m = meanOf(x)
  const m2 = centralMoment(x, 2, m)
  const m4 = centralMoment(x, 4, m)
  let g2 = m2 === 0 ? NaN : m4 / (m2 * m2) - 3
  if (options.biasCorrected) g2 = (((n + 1) * g2 + 6) * (n - 1)) / ((n - 2) * (n - 3))
  return options.excess === false ? g2 + 3 : g2
}

/**
 * The covariance $\sum_i (x_i - \bar{x})(y_i - \bar{y}) / n$, or $/ (n - 1)$ with `sample`
 * (`np.cov(x, y, ddof=...)[0, 1]`). Throws `ShapeError` when the lengths differ and `DomainError` when they are empty.
 *
 * @param xs The first sequence: an array or a rank-1 tensor.
 * @param ys The second sequence, of the same length.
 * @param options `sample` for the $n - 1$ divisor (default: $n$).
 * @returns The covariance.
 *
 * @example Population and sample covariance
 * const x = [1, 2, 3, 4, 5]
 * const y = [2, 4, 5, 4, 5]
 * print('population =', covariance(x, y))
 * print('sample =', covariance(x, y, { sample: true }))
 */
export function covariance(xs: Data, ys: Data, options: SampleOption = {}): number {
  const x = toSequence(xs, 'covariance')
  const y = toSequence(ys, 'covariance')
  requireSameLength(x, y, 'covariance')
  const mx = meanOf(x)
  const my = meanOf(y)
  let s = 0
  for (let i = 0; i < x.length; i++) s += (x[i] - mx) * (y[i] - my)
  return s / (x.length - (options.sample ? 1 : 0))
}

/**
 * Pearson's correlation coefficient $r = \cov(x, y) / (\sigma_x \sigma_y)$, clamped to $[-1, 1]$; NaN when either
 * input is constant. Throws `ShapeError` when the lengths differ and `DomainError` when they are empty.
 *
 * @param xs The first sequence: an array or a rank-1 tensor.
 * @param ys The second sequence, of the same length.
 * @returns The correlation $r$.
 *
 * @example As np.corrcoef(x, y)[0, 1]
 * print('r =', correlation([1, 2, 3, 4, 5], [2, 4, 5, 4, 5]))
 * print('a perfect negative line =', correlation([1, 2, 3], [9, 6, 3]))
 * print('a constant input =', correlation([1, 2, 3], [4, 4, 4]))
 */
export function correlation(xs: Data, ys: Data): number {
  const x = toSequence(xs, 'correlation')
  const y = toSequence(ys, 'correlation')
  requireSameLength(x, y, 'correlation')
  const mx = meanOf(x)
  const my = meanOf(y)
  let sxy = 0
  let sxx = 0
  let syy = 0
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - mx
    const dy = y[i] - my
    sxy += dx * dy
    sxx += dx * dx
    syy += dy * dy
  }
  if (sxx === 0 || syy === 0) return NaN
  // Clamp rounding just outside [−1, 1]; the true value cannot lie outside it.
  return Math.max(-1, Math.min(1, sxy / Math.sqrt(sxx * syy)))
}

/**
 * z-scores $(x_i - \bar{x}) / s$ as a rank-1 tensor, with $s$ the population standard deviation by default
 * (`scipy.stats.zscore`, `ddof=0`). All values are NaN when the data are constant ($0/0$), as in scipy; `standardise`
 * reports that case with a flag. Throws `DomainError` on an empty sequence.
 *
 * @param xs The values: an array or a rank-1 tensor.
 * @param options `sample` to divide by the sample standard deviation ($n - 1$) instead.
 * @returns The z-score of each value, in order.
 *
 * @example As scipy.stats.zscore, with ddof=0 and ddof=1
 * print('z =', zScores([1, 2, 3, 4, 5]))
 * print('z (sample sd) =', zScores([1, 2, 3, 4, 5], { sample: true }))
 */
export function zScores(xs: Data, options: SampleOption = {}): Tensor {
  return standardise(xs, options).values
}

/**
 * Standardises $\xvec$ to zero mean and unit standard deviation, returning the values with the centre and scale used,
 * so that new data can be transformed the same way ($(v - \text{mean}) / \text{scale}$) and results mapped back.
 * `constant` is true when the scale is 0; the values are then NaN rather than silently set to 0. Throws `DomainError`
 * on an empty sequence.
 *
 * @param xs The values: an array or a rank-1 tensor.
 * @param options `sample` to scale by the sample standard deviation ($n - 1$) instead of the population one.
 * @returns The standardised `values` (a rank-1 tensor), the `mean` and `scale` used, and whether the data were
 *   `constant`.
 *
 * @example Standardise, then transform a new value the same way
 * const { values, mean, scale } = standardise([2, 4, 4, 4, 5, 5, 7, 9])
 * print('values =', values)
 * print('mean =', mean, ' scale =', scale)
 * print('11 standardised =', (11 - mean) / scale)
 *
 * @example Constant data are flagged
 * const { values, constant } = standardise([3, 3, 3])
 * print('values =', values)
 * print('constant =', constant)
 */
export function standardise(
  xs: Data,
  options: SampleOption = {},
): { values: Tensor; mean: number; scale: number; constant: boolean } {
  const x = toSequence(xs, 'standardise')
  const m = meanOf(x)
  const scale = Math.sqrt(varianceOf(x, options))
  return { values: vectorOf(Float64Array.from(x, (v) => (v - m) / scale)), mean: m, scale, constant: scale === 0 }
}

export { meanOf, requireNonEmpty, requireSameLength, varianceOf }
