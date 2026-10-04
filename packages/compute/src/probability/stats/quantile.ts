import type { Scalar } from 'aifn-compute/foundation/contracts'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { requireNonEmpty, type Along, type Whole } from './descriptive'
import { allValues, reduce, toSequence, vectorOf, type AxisOption, type Data } from './input'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The quantile methods of `numpy.quantile`, which are the nine sample quantiles of Hyndman and Fan (1996) plus four
 * discontinuous numpy extras. `linear` (H&F 7) is numpy's and R's default.
 *
 * - Discontinuous: `inverted-cdf` (H&F 1), `averaged-inverted-cdf` (H&F 2), `closest-observation` (H&F 3),
 *   `lower`, `higher`, `nearest` (round half to even) and `midpoint`.
 * - Continuous, of the form x(k) with k = nq + α + q(1 − α − β) − 1 (0-based) and linear interpolation:
 *   `interpolated-inverted-cdf` (H&F 4, α=0, β=1), `hazen` (H&F 5, ½, ½), `weibull` (H&F 6, 0, 0), `linear` (H&F 7,
 *   1, 1), `median-unbiased` (H&F 8, ⅓, ⅓) and `normal-unbiased` (H&F 9, ⅜, ⅜).
 */
export type QuantileMethod =
  | 'inverted-cdf'
  | 'averaged-inverted-cdf'
  | 'closest-observation'
  | 'interpolated-inverted-cdf'
  | 'hazen'
  | 'weibull'
  | 'linear'
  | 'median-unbiased'
  | 'normal-unbiased'
  | 'lower'
  | 'higher'
  | 'midpoint'
  | 'nearest'

/** Every quantile method, in numpy's documentation order. */
export const quantileMethods: readonly QuantileMethod[] = [
  'inverted-cdf',
  'averaged-inverted-cdf',
  'closest-observation',
  'interpolated-inverted-cdf',
  'hazen',
  'weibull',
  'linear',
  'median-unbiased',
  'normal-unbiased',
  'lower',
  'higher',
  'midpoint',
  'nearest',
]

// α and β of the continuous methods (Hyndman and Fan 1996, §3; numpy's `_QuantileMethods`).
const continuous: Partial<Record<QuantileMethod, [number, number]>> = {
  'interpolated-inverted-cdf': [0, 1],
  hazen: [0.5, 0.5],
  weibull: [0, 0],
  linear: [1, 1],
  'median-unbiased': [1 / 3, 1 / 3],
  'normal-unbiased': [3 / 8, 3 / 8],
}

/** Round half to even, as `np.around`. */
function roundHalfEven(v: number): number {
  const f = Math.floor(v)
  const d = v - f
  if (d < 0.5) return f
  if (d > 0.5) return f + 1
  return f % 2 === 0 ? f : f + 1
}

/** numpy's `_lerp`: a + (b − a)t, computed from the nearer end for accuracy (and exactness at t = 0 and 1). */
function lerp(a: number, b: number, t: number): number {
  const d = b - a
  return t >= 0.5 ? b - d * (1 - t) : a + d * t
}

/** One quantile of sorted data (ascending, no NaN). */
function quantileSorted(sorted: ArrayLike<number>, q: number, method: QuantileMethod): number {
  const n = sorted.length
  const at = (i: number) => sorted[Math.min(n - 1, Math.max(0, i))]
  const ab = continuous[method]
  if (ab) {
    const [alpha, beta] = ab
    const k = n * q + alpha + q * (1 - alpha - beta) - 1
    if (k < 0) return sorted[0]
    if (k >= n - 1) return sorted[n - 1]
    const lo = Math.floor(k)
    return lerp(sorted[lo], sorted[lo + 1], k - lo)
  }
  switch (method) {
    case 'inverted-cdf': {
      // x(⌈nq⌉) in 1-based order statistics: the smallest x with F̂(x) ≥ q.
      const k = n * q - 1
      const lo = Math.floor(k)
      return at(k - lo === 0 ? lo : lo + 1)
    }
    case 'averaged-inverted-cdf': {
      // As inverted-cdf, but averaging the two neighbours where F̂ is flat at q (nq an integer).
      const k = n * q - 1
      if (k < 0) return sorted[0]
      if (k >= n - 1) return sorted[n - 1]
      const lo = Math.floor(k)
      return lerp(sorted[lo], sorted[lo + 1], k - lo === 0 ? 0.5 : 1)
    }
    case 'closest-observation': {
      // The order statistic nearest nq − ½, taking the even one (1-based) on a tie (H&F 1996, p. 362).
      const k = n * q - 1.5
      const lo = Math.floor(k)
      return at(k - lo === 0 && lo % 2 !== 0 ? lo : lo + 1)
    }
    case 'lower':
      return at(Math.floor((n - 1) * q))
    case 'higher':
      return at(Math.ceil((n - 1) * q))
    case 'nearest':
      return at(roundHalfEven((n - 1) * q))
    case 'midpoint': {
      const k = (n - 1) * q
      return (at(Math.floor(k)) + at(Math.ceil(k))) / 2
    }
    default:
      throw new DomainError('stats', `stats: unknown quantile method "${method}"`)
  }
}

/** The values sorted ascending (NaN last), as a Float64Array; private to stats. */
export function sortedValues(x: Data): Float64Array {
  return Float64Array.from(allValues(x)).sort()
}

/** A sorted copy (ascending, NaN last) as a rank-1 tensor; a tensor contributes every element. */
export function sorted(x: Data): Tensor {
  return vectorOf(sortedValues(x))
}

/** Options of `quantile`, `median` and `interquartileRange` when reducing along an axis. */
export type QuantileOptions = { method?: QuantileMethod }

/** Quantiles of one sequence at each probability. */
function quantilesOf(x: ArrayLike<number>, qs: ArrayLike<number>, method: QuantileMethod): Float64Array {
  requireNonEmpty(x, 'quantile')
  for (let i = 0; i < qs.length; i++)
    if (!(qs[i] >= 0 && qs[i] <= 1))
      throw new DomainError('stats', `stats: quantile probability ${qs[i]} is outside [0, 1]`)
  const s = Float64Array.from(x).sort()
  const hasNaN = Number.isNaN(s[s.length - 1]) // typed-array sort puts NaN last
  return Float64Array.from(qs, (p) => (hasNaN ? NaN : quantileSorted(s, p, method)))
}

/**
 * Sample quantiles of x at probabilities q ∈ [0, 1], by the named method (default `linear`, numpy's default). Returns
 * a number for a number and a rank-1 tensor for an array (or rank-1 tensor) of probabilities. Matches
 * `numpy.quantile(x, q, method=...)`, with hyphens in place of underscores in the method names. NaN values in x give
 * NaN, as in numpy. A tensor x contributes every element; with a number q and `{ axis, keepDims, method }` in place of
 * the method, the quantile is taken along one axis of a tensor and returned as a tensor.
 */
export function quantile(x: Data, q: number, method?: QuantileMethod | Whole<QuantileOptions>): number
export function quantile(x: Data, q: Data, method?: QuantileMethod): Tensor
export function quantile(x: Tensor, q: number, options: Along<QuantileOptions>): Tensor
export function quantile(
  x: Data,
  q: number | Data,
  method: QuantileMethod | (QuantileOptions & AxisOption) = 'linear',
): Scalar | Tensor {
  const options = typeof method === 'string' ? { method } : method
  const m = options.method ?? 'linear'
  if (typeof q !== 'number') {
    if (options.axis !== undefined)
      throw new DomainError('stats', 'stats: quantile along an axis takes one probability')
    return vectorOf(quantilesOf(allValues(x), toSequence(q, 'quantile probabilities'), m))
  }
  return reduce(x, options, (v) => quantilesOf(v, [q], m)[0], 'quantile')
}

/**
 * The median: the `linear` quantile at ½, i.e. the middle value or the mean of the two middle values. Over every
 * element, or along `axis` of a tensor.
 */
export function median(x: Data, options?: Whole): number
export function median(x: Tensor, options: Along): Tensor
export function median(x: Data, options: AxisOption = {}): Scalar | Tensor {
  return reduce(x, options, (v) => quantilesOf(v, [0.5], 'linear')[0], 'median')
}

/**
 * The interquartile range q(0.75) − q(0.25) by the given quantile method (default `linear`), over every element, or
 * along an axis of a tensor with `{ axis, keepDims, method }` in place of the method.
 */
export function interquartileRange(x: Data, method?: QuantileMethod | Whole<QuantileOptions>): number
export function interquartileRange(x: Tensor, options: Along<QuantileOptions>): Tensor
export function interquartileRange(
  x: Data,
  method: QuantileMethod | (QuantileOptions & AxisOption) = 'linear',
): Scalar | Tensor {
  const options = typeof method === 'string' ? { method } : method
  return reduce(
    x,
    options,
    (v) => {
      const [a, b] = quantilesOf(v, [0.25, 0.75], options.method ?? 'linear')
      return b - a
    },
    'interquartileRange',
  )
}
