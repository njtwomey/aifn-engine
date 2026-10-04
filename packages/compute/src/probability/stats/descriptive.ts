/**
 * Descriptive statistics. Every function takes a `number[]`, a typed array or an `aifn-compute/foundation/tensor` tensor. Reductions
 * (`sum`, `mean`, `variance`, `standardDeviation`, `min`, `max`, `range`, `skewness`, `kurtosis`) reduce every element
 * of a tensor to a number, or take `{ axis, keepDims }` to reduce a tensor along one axis and return a tensor; pairwise
 * and elementwise functions need rank-1 tensors. Variances default to the population form (divide by n, like `np.var`); pass `{ sample: true }` for the
 * unbiased sample form (divide by n − 1, like `np.var(ddof=1)`).
 */

import type { Scalar } from 'aifn-compute/foundation/contracts'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { allValues, reduce, toSequence, vectorOf, type AxisOption, type Data } from './input'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options for moments that have a population and a sample form. */
export type SampleOption = {
  /** Divide by n − 1 (the unbiased sample variance) instead of n (the population variance). Default false. */
  sample?: boolean
}

function requireNonEmpty(x: ArrayLike<number>, what: string) {
  if (x.length === 0) throw new DomainError('stats', `stats: ${what} of an empty array`)
}

function requireSameLength(x: ArrayLike<number>, y: ArrayLike<number>, what: string) {
  if (x.length !== y.length)
    throw new ShapeError('stats', `stats: ${what} needs arrays of equal length (${x.length} and ${y.length})`)
}

/** A reduction's options with the axis given: the result is a tensor. */
export type Along<O = unknown> = O & AxisOption & { axis: number }
/** A reduction's options without an axis: the result is a number. */
export type Whole<O = unknown> = O & { axis?: undefined; keepDims?: boolean }

/** Neumaier's compensated sum: exact to about one rounding for ill-conditioned sums (Neumaier 1974). */
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
 */
export function sum(x: Data, options?: Whole): number
export function sum(x: Tensor, options: Along): Tensor
export function sum(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, sumOf, 'sum')
}

function meanOf(x: ArrayLike<number>): number {
  requireNonEmpty(x, 'mean')
  return sumOf(x) / x.length
}

/** The arithmetic mean, over every element or along `axis` of a tensor. Throws on an empty array. */
export function mean(x: Data, options?: Whole): number
export function mean(x: Tensor, options: Along): Tensor
export function mean(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, meanOf, 'mean')
}

function varianceOf(x: ArrayLike<number>, options: SampleOption): number {
  requireNonEmpty(x, 'variance')
  const m = meanOf(x)
  let ss = 0
  for (let i = 0; i < x.length; i++) ss += (x[i] - m) ** 2
  return ss / (x.length - (options.sample ? 1 : 0))
}

/**
 * The variance: Σ(xᵢ − x̄)² / n, or / (n − 1) with `sample`. Two-pass (the mean first), which is accurate when the mean
 * is large relative to the spread. A sample variance of one value is NaN (0/0), as in numpy. Over every element, or
 * along `axis` of a tensor.
 */
export function variance(x: Data, options?: Whole<SampleOption>): number
export function variance(x: Tensor, options: Along<SampleOption>): Tensor
export function variance(x: Data, options: SampleOption & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => varianceOf(v, options), 'variance')
}

/**
 * The standard deviation: the square root of `variance` (population by default, `{ sample: true }` for n − 1). Over
 * every element, or along `axis` of a tensor.
 */
export function standardDeviation(x: Data, options?: Whole<SampleOption>): number
export function standardDeviation(x: Tensor, options: Along<SampleOption>): Tensor
export function standardDeviation(x: Data, options: SampleOption & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => Math.sqrt(varianceOf(v, options)), 'standardDeviation')
}

/** The weighted mean Σwᵢxᵢ / Σwᵢ of a sequence x (length n). Weights must be non-negative with a positive sum. */
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
 * The weighted variance S / D where S = Σwᵢ(xᵢ − x̄_w)². The divisor depends on what the weights mean:
 * - `population` (default): D = Σw. Weights as a distribution; `np.cov(x, aweights=w, ddof=0)`.
 * - `frequency`: D = Σw − 1. Integer weights that count repeats; `np.cov(x, fweights=w)`.
 * - `reliability`: D = Σw − Σw²/Σw. Unbiased for weights as relative precisions; `np.cov(x, aweights=w, ddof=1)`.
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

function minOf(x: ArrayLike<number>): number {
  requireNonEmpty(x, 'min')
  let m = Infinity
  for (let i = 0; i < x.length; i++) {
    if (Number.isNaN(x[i])) return NaN
    if (x[i] < m) m = x[i]
  }
  return m
}

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
 * The smallest value; NaN if any value is NaN (as `np.min`). Throws on an empty array. Over every element, or along
 * `axis` of a tensor.
 */
export function min(x: Data, options?: Whole): number
export function min(x: Tensor, options: Along): Tensor
export function min(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, minOf, 'min')
}

/**
 * The largest value; NaN if any value is NaN (as `np.max`). Throws on an empty array. Over every element, or along
 * `axis` of a tensor.
 */
export function max(x: Data, options?: Whole): number
export function max(x: Tensor, options: Along): Tensor
export function max(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, maxOf, 'max')
}

/** `[min, max]` over every element, e.g. for an axis range. */
export function extent(x: Data): [number, number] {
  const v = allValues(x)
  return [minOf(v), maxOf(v)]
}

/** The range max − min (`np.ptp`), over every element or along `axis` of a tensor. */
export function range(x: Data, options?: Whole): number
export function range(x: Tensor, options: Along): Tensor
export function range(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => maxOf(v) - minOf(v), 'range')
}

/**
 * The most frequent value and its count. Ties go to the smallest value, as in `scipy.stats.mode`. Values are compared
 * exactly, so this is meant for discrete data. A tensor contributes every element.
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

/** Options of `kurtosis`. */
export type KurtosisOptions = { excess?: boolean; biasCorrected?: boolean }

/** The k-th central moment (1/n) Σ(xᵢ − x̄)^k. */
function centralMoment(x: ArrayLike<number>, k: number, m: number): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += (x[i] - m) ** k
  return s / x.length
}

/**
 * The skewness g₁ = m₃ / m₂^{3/2} from central moments mₖ. With `biasCorrected`, the adjusted Fisher–Pearson
 * coefficient G₁ = g₁ √(n(n − 1)) / (n − 2) (Joanes and Gill 1998). Matches `scipy.stats.skew(x, bias=...)`.
 * NaN when the data are constant. Over every element, or along `axis` of a tensor.
 */
export function skewness(x: Data, options?: Whole<{ biasCorrected?: boolean }>): number
export function skewness(x: Tensor, options: Along<{ biasCorrected?: boolean }>): Tensor
export function skewness(x: Data, options: { biasCorrected?: boolean } & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => skewnessOf(v, options), 'skewness')
}

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
 * The kurtosis m₄ / m₂², minus 3 by default (`excess`, so a Gaussian scores 0). With `biasCorrected`, the unbiased
 * estimator under normality G₂ = ((n + 1) g₂ + 6)(n − 1) / ((n − 2)(n − 3)) (Joanes and Gill 1998). Matches
 * `scipy.stats.kurtosis(x, fisher=excess, bias=not biasCorrected)`. NaN when the data are constant. Over every element, or along `axis` of a tensor.
 */
export function kurtosis(x: Data, options?: Whole<KurtosisOptions>): number
export function kurtosis(x: Tensor, options: Along<KurtosisOptions>): Tensor
export function kurtosis(x: Data, options: KurtosisOptions & AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => kurtosisOf(v, options), 'kurtosis')
}

function kurtosisOf(x: ArrayLike<number>, options: KurtosisOptions): number {
  const n = x.length
  const m = meanOf(x)
  const m2 = centralMoment(x, 2, m)
  const m4 = centralMoment(x, 4, m)
  let g2 = m2 === 0 ? NaN : m4 / (m2 * m2) - 3
  if (options.biasCorrected) g2 = (((n + 1) * g2 + 6) * (n - 1)) / ((n - 2) * (n - 3))
  return options.excess === false ? g2 + 3 : g2
}

/** The covariance Σ(xᵢ − x̄)(yᵢ − ȳ) / n, or / (n − 1) with `sample` (`np.cov(x, y, ddof=...)[0, 1]`). */
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

/** Pearson's correlation coefficient r = cov(x, y) / (sd(x) sd(y)); NaN when either input is constant. */
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
 * z-scores (xᵢ − x̄) / s as a rank-1 tensor, with s the population standard deviation by default (`scipy.stats.zscore`, `ddof=0`). All
 * values are NaN when the data are constant (0/0), as in scipy; `standardise` reports that case with a flag.
 */
export function zScores(xs: Data, options: SampleOption = {}): Tensor {
  return standardise(xs, options).values
}

/**
 * Standardises x to zero mean and unit standard deviation, returning the values with the centre and scale used, so
 * that new data can be transformed the same way (`(v − mean) / scale`) and results mapped back. `constant` is true
 * when the scale is 0; the values are then NaN rather than silently set to 0.
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
