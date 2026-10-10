/**
 * Quality measures of subgroup discovery and their optimistic estimates. A measure scores a cover (the rows a
 * description selects) by how unusual the target is inside it; an optimistic estimate bounds the quality of every
 * subset of the cover, so of every refinement, which lets branch and bound skip whole branches.
 *
 * Binary target ($n$ rows in the subgroup, $\mathit{tp}$ of them positive; $N$ rows and $P$ positives in all;
 * $p = \mathit{tp}/n$, $p_0 = P/N$):
 *
 * - the Klösgen family $q_a = (n/N)^a (p - p_0)$ (Klösgen, 1996): $a = 1$ is weighted relative accuracy (WRAcc;
 *   Lavrač, Flach and Zupan, 1999), $a = 1/2$ the binomial-test quality, $a = 0$ the added value $p - p_0$. Its
 *   optimistic estimate for $a \in [0, 1]$ is the subset of the positives alone, $(\mathit{tp}/N)^a (1 - p_0)$
 *   (Grosskreutz, Rüping and Wrobel, 2008);
 * - the binomial test $z = \sqrt{n} (p - p_0)/\sqrt{p_0(1 - p_0)}$, the one-sample $z$ statistic of the subgroup's
 *   rate;
 * - lift $p/p_0$, bounded under a minimum support $m$ by $\min(1, \mathit{tp}/m)/p_0$;
 * - coverage $n/N$, bounded by itself (it only shrinks);
 * - the $\chi^2$ statistic of the $2 \times 2$ table subgroup $\times$ target, bounded by the larger of the $\chi^2$ of
 *   the positives alone and of the negatives alone ($\chi^2$ is convex in $(\mathit{tp}, n - \mathit{tp})$; Morishita
 *   and Sese, 2000).
 *
 * Numeric target (values $y_i$, population mean $\mu_0$ and standard deviation $\sigma_0$): the mean shift
 * $q_a = n^a (\mu - \mu_0)/\sigma_0$ ($a = 1/2$: the $z$-score of the subgroup mean). Its tight optimistic estimate is
 * the best prefix of the cover's values sorted from the largest: $\max_j j^a (\bar y_{(j)} - \mu_0)/\sigma_0$ with
 * $\bar y_{(j)}$ the mean of the top $j$, since for every size $j$ the subset with the highest mean is the top $j$
 * (Lemmerich, Atzmueller and Puppe, 2016).
 *
 * `direction` asks for subgroups where the target is higher than in the population, lower, or either (`both`, the
 * absolute value; its estimate is the larger of the two one-sided estimates). An empty cover has quality $-\infty$.
 */

import type { Tensor } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { bitset, bitsetAndCount, bitsetCount, bitsetIndices, type Bitset } from './cover'

/** A quality measure over covers, with an optional optimistic estimate. */
export interface QualityMeasure {
  /** A short identifier of the measure (`wracc`, `lift`, `meanShift`, ...). */
  readonly key: string
  /** The measure's name for display. */
  readonly name: string
  /** Rows in the population. */
  readonly rows: number
  /** The quality of a cover ($-\infty$ for an empty one). */
  quality(cover: Bitset): number
  /** An upper bound on the quality of every non-empty subset of the cover. */
  bound?(cover: Bitset): number
}

/**
 * A binary-target measure: a function of the counts $(n, \mathit{tp})$ alone, so count-based miners (SD-Map) can use
 * it.
 */
export interface CountMeasure extends QualityMeasure {
  /** The number of positive rows $P$ in the population. */
  readonly positives: number
  /** The positives as a cover. */
  readonly target: Bitset
  /** The quality of a subgroup of $n$ rows with $\mathit{tp}$ positives ($-\infty$ for $n = 0$). */
  fromCounts(n: number, tp: number): number
  /** The optimistic estimate from the same counts, when the measure has one. */
  boundFromCounts?(n: number, tp: number): number
}

/** Which deviation is interesting. */
export type Direction = 'higher' | 'lower' | 'both'

type Target = ArrayLike<number> | Tensor

/**
 * A target's values as an array: a tensor flattened, an array as it is.
 *
 * @param t The target column.
 * @returns Its values in row order.
 */
const flat = (t: Target): ArrayLike<number> => (isTensor(t) ? toFlat(t) : t)

/**
 * A `CountMeasure` from a quality and an optional bound of the counts. A row is positive when its target is neither 0
 * nor NaN. Throws `DomainError` for an empty target.
 *
 * @param key The measure's `key`, also the name in error messages.
 * @param name The measure's display name.
 * @param target The binary target, one value per row.
 * @param q The quality of a non-empty subgroup from $n$, $\mathit{tp}$, $N$ and $P$.
 * @param b The optimistic estimate from the same counts, if there is one.
 * @returns The measure, with `quality` and `bound` on covers built from `q` and `b`.
 */
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

/**
 * A deviation as the direction scores it.
 *
 * @param direction Which deviation is interesting.
 * @param d The deviation from the population (subgroup minus population).
 * @returns $d$ for `higher`, $-d$ for `lower`, $\lvert d \rvert$ for `both`.
 */
const signed = (direction: Direction, d: number) =>
  direction === 'higher' ? d : direction === 'lower' ? -d : Math.abs(d)

/**
 * The Klösgen family $(n/N)^a (p - p_0)$, $a \in [0, 1]$, of a 0/1 target; optimistic estimate
 * $(\mathit{tp}/N)^a (1 - p_0)$ (for `lower`, $((n - \mathit{tp})/N)^a p_0$). Throws `DomainError` for $a$ outside
 * $[0, 1]$ or an empty target.
 *
 * @param target The binary target, one value per row: positive where it is neither 0 nor NaN.
 * @param options `a`, the weight of the subgroup's size (default 1, WRAcc), and `direction` (default `higher`).
 * @returns The measure.
 *
 * @example Half the rows, with half of them positive against 3/8 overall
 * const target = [1, 1, 0, 0, 1, 0, 0, 0]
 * const cover = bitset(8, (i) => i < 4)
 * for (const a of [0, 0.5, 1]) print(`a = ${a}: quality =`, standardQuality(target, { a }).quality(cover))
 * print('optimistic estimate (a = 1) =', standardQuality(target).bound(cover))
 */
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

/**
 * Weighted relative accuracy $(n/N)(p - p_0)$: the Klösgen quality with $a = 1$.
 *
 * @param target The binary target, one value per row: positive where it is neither 0 nor NaN.
 * @param options `direction` (default `higher`).
 * @returns The measure, with key `wracc`.
 *
 * @example A subgroup with a higher rate than the population
 * const target = [1, 1, 0, 0, 1, 0, 0, 0]
 * const wracc = wraccQuality(target)
 * print('rows 0-3:', wracc.quality(bitset(8, (i) => i < 4)))
 * print('rows 0-1:', wracc.quality(bitset(8, (i) => i < 2)))
 * print('from counts n = 2, tp = 2:', wracc.fromCounts(2, 2))
 */
export function wraccQuality(target: Target, options: { direction?: Direction } = {}): CountMeasure {
  return { ...standardQuality(target, { a: 1, ...options }), key: 'wracc', name: 'WRAcc' }
}

/**
 * The binomial test $z = \sqrt{n} (p - p_0)/\sqrt{p_0(1 - p_0)}$: $a = 1/2$ in the Klösgen family, scaled by
 * $\sqrt{N}/\sqrt{p_0(1 - p_0)}$. Every subgroup scores 0 when the target is all positive or all negative.
 *
 * @param target The binary target, one value per row: positive where it is neither 0 nor NaN.
 * @param options `direction` (default `higher`).
 * @returns The measure.
 *
 * @example The z statistic of a subgroup's rate
 * const target = [1, 1, 0, 0, 1, 0, 0, 0]
 * const z = binomialQuality(target).quality(bitset(8, (i) => i < 4))
 * print('z =', z, ' by hand:', (Math.sqrt(4) * (0.5 - 0.375)) / Math.sqrt(0.375 * 0.625))
 */
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

/**
 * Lift $p/p_0$; under a minimum support $m$ (default 1) its optimistic estimate is $\min(1, \mathit{tp}/m)/p_0$, the
 * best precision a subset of at least $m$ rows can reach. Every subgroup scores 0 when there are no positives.
 *
 * @param target The binary target, one value per row: positive where it is neither 0 nor NaN.
 * @param options `minSupport`, the smallest subgroup the search keeps, $m$ (default 1): pass the search's own.
 * @returns The measure.
 *
 * @example A subgroup's lift, and how far a refinement could raise it
 * const target = [1, 1, 0, 0, 1, 0, 0, 0]
 * const lift = liftQuality(target, { minSupport: 2 })
 * const cover = bitset(8, (i) => i < 4)
 * print('lift =', lift.quality(cover), ' (0.5 / 0.375)')
 * print('optimistic estimate =', lift.bound(cover))
 */
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

/**
 * Coverage $n/N$, its own optimistic estimate.
 *
 * @param target The binary target, one value per row; only its length is used.
 * @returns The measure.
 *
 * @example The share of rows a subgroup covers
 * print(coverageQuality([1, 1, 0, 0, 1, 0, 0, 0]).quality(bitset(8, (i) => i < 2)))
 */
export function coverageQuality(target: Target): CountMeasure {
  return countMeasure(
    'coverage',
    'Coverage',
    target,
    (n, _tp, N) => n / N,
    (n, _tp, N) => n / N,
  )
}

/**
 * $\chi^2$ of the $2 \times 2$ table (in the subgroup or not) $\times$ (positive or not), without continuity
 * correction: $N (ad - bc)^2 / (n (N - n) P (N - P))$ with cells $a = \mathit{tp}$, $b = n - \mathit{tp}$,
 * $c = P - \mathit{tp}$ and $d = N - n - c$; 0 when a margin is empty.
 *
 * @param n The rows in the subgroup.
 * @param tp The positive rows in the subgroup.
 * @param N The rows in all.
 * @param P The positive rows in all.
 * @returns The statistic.
 */
function chiSquare2x2(n: number, tp: number, N: number, P: number): number {
  if (n === 0 || n === N || P === 0 || P === N) return 0
  const a = tp
  const b = n - tp
  const c = P - tp
  const d = N - n - c
  const det = a * d - b * c
  return (N * det * det) / (n * (N - n) * P * (N - P))
}

/**
 * The $\chi^2$ statistic of subgroup $\times$ target; optimistic estimate
 * $\max(\chi^2(\text{positives alone}), \chi^2(\text{negatives alone}))$. With `higher` or `lower`, a subgroup
 * deviating the other way scores $-\chi^2$.
 *
 * @param target The binary target, one value per row: positive where it is neither 0 nor NaN.
 * @param options `direction` (default `both`).
 * @returns The measure.
 *
 * @example The chi-squared of a subgroup, and of one deviating the other way
 * const target = [1, 1, 0, 0, 1, 0, 0, 0]
 * const higher = chiSquareQuality(target, { direction: 'higher' })
 * print('rows 0-3:', higher.quality(bitset(8, (i) => i < 4)))
 * print('rows 4-7:', higher.quality(bitset(8, (i) => i >= 4)))
 */
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

/** A numeric-target measure, with the target's values and their population mean and standard deviation. */
export interface NumericMeasure extends QualityMeasure {
  /** The target's values, one per row. */
  readonly values: Float64Array
  /** The population mean $\mu_0$. */
  readonly mean: number
  /** The population standard deviation $\sigma_0$ (dividing by $N$). */
  readonly sd: number
}

/**
 * The mean shift $n^a (\mu - \mu_0)/\sigma_0$ of a numeric target ($a \in [0, 1]$, default $1/2$: the $z$-score of
 * the subgroup mean; with `standardise: false` the $\sigma_0$ is dropped, as in pysubgroup's `StandardQFNumeric`). Its
 * optimistic estimate is the best $j^a (\bar y_{(j)} - \mu_0)/\sigma_0$ over $j$, with $\bar y_{(j)}$ the mean of the
 * $j$ largest values of the cover (the $j$ smallest for `lower`). A constant target scores 0. Throws `DomainError` for
 * $a$ outside $[0, 1]$ or an empty target.
 *
 * @param target The numeric target, one value per row.
 * @param options `a` (default $1/2$), `direction` (default `higher`) and `standardise` (default true: divide by
 *   $\sigma_0$).
 * @returns The measure, with the target's values, mean and standard deviation.
 *
 * @example A subgroup's mean shift, and the most a subset of it could reach
 * const target = [1, 2, 3, 4, 5, 6, 7, 8]
 * const shift = meanShiftQuality(target)
 * const cover = bitset(8, (i) => i === 0 || i >= 6)
 * print('mean =', shift.mean, 'sd =', shift.sd)
 * print('quality of {1, 7, 8} =', shift.quality(cover))
 * print('optimistic estimate =', shift.bound(cover), ' = quality of {7, 8}:', shift.quality(bitset(8, (i) => i >= 6)))
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
  /** The best $j^a$ times the signed shift of the mean of the $j$ most extreme values on one side, over $j$. */
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
