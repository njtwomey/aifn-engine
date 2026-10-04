/**
 * Quality measures of subgroup discovery and their optimistic estimates. A measure scores a cover (the rows a
 * description selects) by how unusual the target is inside it; an optimistic estimate bounds the quality of every
 * subset of the cover, so of every refinement, which lets branch and bound skip whole branches.
 *
 * Binary target (n rows in the subgroup, tp of them positive; N rows and P positives in all; p = tp/n, p₀ = P/N):
 *
 * - the Klösgen family q_a = (n/N)^a (p − p₀) (Klösgen, 1996): a = 1 is weighted relative accuracy (WRAcc; Lavrač,
 *   Flach and Zupan, 1999), a = ½ the binomial-test quality, a = 0 the added value p − p₀. Its optimistic estimate for
 *   a ∈ [0, 1] is the subset of the positives alone, (tp/N)^a (1 − p₀) (Grosskreutz, Rüping and Wrobel, 2008);
 * - the binomial test z = √n (p − p₀)/√(p₀(1 − p₀)), the one-sample z statistic of the subgroup's rate;
 * - lift p/p₀, bounded under a minimum support m by min(1, tp/m)/p₀;
 * - coverage n/N, bounded by itself (it only shrinks);
 * - the χ² statistic of the 2 × 2 table subgroup × target, bounded by the larger of the χ² of the positives alone and
 *   of the negatives alone (χ² is convex in (tp, n − tp); Morishita and Sese, 2000).
 *
 * Numeric target (values yᵢ, population mean μ₀ and standard deviation σ₀): the mean shift
 * q_a = n^a (μ − μ₀)/σ₀ (a = ½: the z-score of the subgroup mean). Its tight optimistic estimate is the best prefix of
 * the cover's values sorted from the largest: maxⱼ j^a (mean of the top j − μ₀)/σ₀, since for every size j the subset
 * with the highest mean is the top j (Lemmerich, Atzmueller and Puppe, 2016).
 *
 * `direction` asks for subgroups where the target is higher than in the population, lower, or either (`both`, the
 * absolute value; its estimate is the larger of the two one-sided estimates).
 */

import type { Tensor } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { bitset, bitsetAndCount, bitsetCount, bitsetIndices, type Bitset } from './cover'

/** A quality measure over covers, with an optional optimistic estimate. */
export interface QualityMeasure {
  readonly key: string
  readonly name: string
  /** Rows in the population. */
  readonly rows: number
  /** The quality of a cover (−∞ for an empty one). */
  quality(cover: Bitset): number
  /** An upper bound on the quality of every non-empty subset of the cover. */
  bound?(cover: Bitset): number
}

/** A binary-target measure: a function of the counts (n, tp) alone, so count-based miners (SD-Map) can use it. */
export interface CountMeasure extends QualityMeasure {
  readonly positives: number
  /** The positives as a cover. */
  readonly target: Bitset
  fromCounts(n: number, tp: number): number
  boundFromCounts?(n: number, tp: number): number
}

/** Which deviation is interesting. */
export type Direction = 'higher' | 'lower' | 'both'

type Target = ArrayLike<number> | Tensor

const flat = (t: Target): ArrayLike<number> => (isTensor(t) ? toFlat(t) : t)

function countMeasure(
  key: string,
  name: string,
  target: Target,
  q: (n: number, tp: number, N: number, P: number) => number,
  b?: (n: number, tp: number, N: number, P: number) => number,
): CountMeasure {
  const y = flat(target)
  const N = y.length
  if (N === 0) throw new DomainError(key, `${key}: the target is empty`)
  const t = bitset(N, (i) => y[i] !== 0 && !Number.isNaN(y[i]))
  const P = bitsetCount(t)
  const fromCounts = (n: number, tp: number) => (n > 0 ? q(n, tp, N, P) : -Infinity)
  const boundFromCounts = b ? (n: number, tp: number) => (n > 0 ? b(n, tp, N, P) : -Infinity) : undefined
  return {
    key,
    name,
    rows: N,
    positives: P,
    target: t,
    fromCounts,
    quality: (c) => fromCounts(bitsetCount(c), bitsetAndCount(c, t)),
    ...(boundFromCounts
      ? { boundFromCounts, bound: (c: Bitset) => boundFromCounts(bitsetCount(c), bitsetAndCount(c, t)) }
      : {}),
  }
}

const signed = (direction: Direction, d: number) =>
  direction === 'higher' ? d : direction === 'lower' ? -d : Math.abs(d)

/** The Klösgen family (n/N)^a (p − p₀), a ∈ [0, 1], of a 0/1 target; optimistic estimate (tp/N)^a (1 − p₀). */
export function standardQuality(target: Target, options: { a?: number; direction?: Direction } = {}): CountMeasure {
  const a = options.a ?? 1
  const direction = options.direction ?? 'higher'
  if (!(a >= 0 && a <= 1)) throw new DomainError('standardQuality', 'standardQuality: a must be in [0, 1]')
  return countMeasure(
    'standard',
    `Klösgen quality (a = ${a})`,
    target,
    (n, tp, N, P) => Math.pow(n / N, a) * signed(direction, tp / n - P / N),
    (n, tp, N, P) => {
      const up = Math.pow(tp / N, a) * (1 - P / N)
      const down = Math.pow((n - tp) / N, a) * (P / N)
      return direction === 'higher' ? up : direction === 'lower' ? down : Math.max(up, down)
    },
  )
}

/** Weighted relative accuracy (n/N)(p − p₀): the Klösgen quality with a = 1. */
export function wraccQuality(target: Target, options: { direction?: Direction } = {}): CountMeasure {
  return { ...standardQuality(target, { a: 1, ...options }), key: 'wracc', name: 'WRAcc' }
}

/** The binomial test z = √n (p − p₀)/√(p₀(1 − p₀)): a = ½ in the Klösgen family, scaled by √N/√(p₀(1 − p₀)). */
export function binomialQuality(target: Target, options: { direction?: Direction } = {}): CountMeasure {
  const direction = options.direction ?? 'higher'
  const scale = (N: number, P: number) => {
    const p0 = P / N
    return p0 > 0 && p0 < 1 ? Math.sqrt(N / (p0 * (1 - p0))) : 0
  }
  const half = standardQuality(target, { a: 0.5, direction })
  const N = half.rows
  const s = scale(N, half.positives)
  return countMeasure(
    'binomial',
    'Binomial test z',
    target,
    (n, tp) => s * half.fromCounts(n, tp),
    (n, tp) => s * half.boundFromCounts!(n, tp),
  )
}

/** Lift p/p₀; under a minimum support m (default 1) its optimistic estimate is min(1, tp/m)/p₀. */
export function liftQuality(target: Target, options: { minSupport?: number } = {}): CountMeasure {
  const m = Math.max(1, options.minSupport ?? 1)
  return countMeasure(
    'lift',
    'Lift',
    target,
    (n, tp, N, P) => (P > 0 ? tp / n / (P / N) : 0),
    (_n, tp, N, P) => (P > 0 ? Math.min(1, tp / m) / (P / N) : 0),
  )
}

/** Coverage n/N, its own optimistic estimate. */
export function coverageQuality(target: Target): CountMeasure {
  return countMeasure(
    'coverage',
    'Coverage',
    target,
    (n, _tp, N) => n / N,
    (n, _tp, N) => n / N,
  )
}

/** χ² of the 2 × 2 table (in the subgroup or not) × (positive or not), without continuity correction. */
function chiSquare2x2(n: number, tp: number, N: number, P: number): number {
  if (n === 0 || n === N || P === 0 || P === N) return 0
  const a = tp
  const b = n - tp
  const c = P - tp
  const d = N - n - c
  const det = a * d - b * c
  return (N * det * det) / (n * (N - n) * P * (N - P))
}

/** The χ² statistic of subgroup × target; optimistic estimate max(χ²(positives alone), χ²(negatives alone)). */
export function chiSquareQuality(target: Target, options: { direction?: Direction } = {}): CountMeasure {
  const direction = options.direction ?? 'both'
  return countMeasure(
    'chiSquare',
    'χ²',
    target,
    (n, tp, N, P) => {
      const v = chiSquare2x2(n, tp, N, P)
      const up = tp / n >= P / N
      return direction === 'both' || (direction === 'higher') === up ? v : -v
    },
    (n, tp, N, P) => {
      const up = tp > 0 ? chiSquare2x2(tp, tp, N, P) : 0
      const down = n - tp > 0 ? chiSquare2x2(n - tp, 0, N, P) : 0
      return direction === 'higher' ? up : direction === 'lower' ? down : Math.max(up, down)
    },
  )
}

/** A numeric-target measure: the mean shift and its estimate also report the cover's values. */
export interface NumericMeasure extends QualityMeasure {
  readonly values: Float64Array
  readonly mean: number
  readonly sd: number
}

/**
 * The mean shift n^a (μ − μ₀)/σ₀ of a numeric target (a ∈ [0, 1], default ½: the z-score of the subgroup mean; with
 * `standardise: false` the σ₀ is dropped, as in pysubgroup's `StandardQFNumeric`). Its optimistic estimate is the best
 * j^a (mean of the j largest values − μ₀)/σ₀ over j.
 */
export function meanShiftQuality(
  target: Target,
  options: { a?: number; direction?: Direction; standardise?: boolean } = {},
): NumericMeasure {
  const a = options.a ?? 0.5
  const direction = options.direction ?? 'higher'
  if (!(a >= 0 && a <= 1)) throw new DomainError('meanShiftQuality', 'meanShiftQuality: a must be in [0, 1]')
  const values = Float64Array.from(flat(target))
  const N = values.length
  if (N === 0) throw new DomainError('meanShiftQuality', 'meanShiftQuality: the target is empty')
  let mean = 0
  for (const v of values) mean += v
  mean /= N
  let ss = 0
  for (const v of values) ss += (v - mean) * (v - mean)
  const sd = Math.sqrt(ss / N)
  const scale = (options.standardise ?? true) ? (sd > 0 ? 1 / sd : 0) : 1
  const quality = (c: Bitset) => {
    const idx = bitsetIndices(c)
    if (!idx.length) return -Infinity
    let s = 0
    for (const i of idx) s += values[i]
    return Math.pow(idx.length, a) * signed(direction, s / idx.length - mean) * scale
  }
  /** max over j of j^a (mean of the j most extreme values on one side − μ₀), signed for that side. */
  const best = (sorted: Float64Array, sign: 1 | -1) => {
    let s = 0
    let out = -Infinity
    for (let j = 1; j <= sorted.length; j++) {
      s += sorted[j - 1]
      out = Math.max(out, Math.pow(j, a) * sign * (s / j - mean))
    }
    return out
  }
  const bound = (c: Bitset) => {
    const idx = bitsetIndices(c)
    if (!idx.length) return -Infinity
    const asc = Float64Array.from(idx, (i) => values[i]).sort()
    const up = direction === 'lower' ? -Infinity : best(asc.slice().reverse(), 1)
    const down = direction === 'higher' ? -Infinity : best(asc, -1)
    return Math.max(up, down) * scale
  }
  return {
    key: 'meanShift',
    name: `Mean shift (a = ${a})`,
    rows: N,
    values,
    mean,
    sd,
    quality,
    bound,
  }
}
