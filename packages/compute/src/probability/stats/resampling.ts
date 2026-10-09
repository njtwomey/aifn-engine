/**
 * Resampling: bootstrap resamples and the nonparametric bootstrap with its percentile and basic intervals, random
 * permutations and the two-sample permutation test, and Kish's effective sample size of importance weights.
 *
 * Randomness comes from an `aifn-compute/foundation/random` stream, the first argument. `resampleIndices` and
 * `shuffled` draw from it and advance it; `bootstrap` and `permutationTest` draw resample $r$ from a child stream named
 * by $r$, so their results depend only on the stream's key and the stream itself is not advanced.
 */

import { child, integers, permutation, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Size } from 'aifn-compute/foundation/contracts'
import { mean, requireNonEmpty, standardDeviation } from './descriptive'
import { toSequence, vectorOf, type Data } from './input'
import { quantile, type QuantileMethod } from './quantile'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * `size` indices drawn uniformly with replacement from $0, \dots, n - 1$ (default size $n$), as an int32 rank-1
 * tensor: one bootstrap resample.
 *
 * @param s The random stream; advanced by the draws.
 * @param n The number of values to draw indices from.
 * @param size How many indices to draw.
 * @returns The indices.
 *
 * @example One bootstrap resample of five values
 * const x = tensor([10, 20, 30, 40, 50])
 * const idx = resampleIndices(stream(1), 5)
 * print('indices =', idx)
 * print('resample =', take(x, idx))
 */
export function resampleIndices(s: Stream, n: Size, size: Size = n): Tensor {
  return integers(s, n, { shape: [size] })
}

/**
 * The values of $\xvec$ in the order of a uniformly random permutation (`aifn-compute/foundation/random`'s
 * `permutation`).
 *
 * @param s The random stream; advanced by the draw.
 * @param x The values; not modified.
 * @returns A new array of the values in permuted order.
 */
function shuffledValues(s: Stream, x: ArrayLike<number>): Float64Array {
  const order = toFlat(permutation(s, x.length))
  return Float64Array.from(order, (j) => x[j])
}

/**
 * A uniformly random permutation of $\xvec$ (drawn by `aifn-compute/foundation/random`'s `permutation`), as a new
 * rank-1 tensor. (The in-place shuffle of an index array is `aifn-compute/foundation/random`'s `shuffle`.)
 *
 * @param s The random stream; advanced by the draw.
 * @param xData The values: an array or a rank-1 tensor. Not modified.
 * @returns The values in a random order.
 *
 * @example Shuffle twice from one stream
 * const s = stream(1)
 * print('first =', shuffled(s, [1, 2, 3, 4, 5]))
 * print('second =', shuffled(s, [1, 2, 3, 4, 5]))
 */
export function shuffled(s: Stream, xData: Data): Tensor {
  return vectorOf(shuffledValues(s, toSequence(xData, 'shuffled')))
}

/** The result of `bootstrap`. */
export type Bootstrap = {
  /** The statistic on the original sample. */
  estimate: number
  /** The statistic on each resample (rank-1). */
  replicates: Tensor
  /** The standard deviation of the replicates ($n - 1$ divisor); NaN for fewer than two resamples. */
  standardError: number
  /** The mean of the replicates minus the estimate; NaN for no resamples. */
  bias: number
}

/**
 * The nonparametric bootstrap (Efron 1979): the statistic on `resamples` samples drawn with replacement from
 * $\xvec$, with resample $r$ drawn from `child(s, 'resample', r)`. Returns the estimate, the replicates, their
 * standard error and the bias estimate. Throws `DomainError` on an empty sample.
 *
 * @param s The random stream whose children draw the resamples; not advanced.
 * @param xData The sample: an array or a rank-1 tensor.
 * @param statistic The statistic of a sample. It receives one buffer of $n$ values, refilled for each resample, so it
 *   must not keep it.
 * @param resamples The number of bootstrap resamples.
 * @returns The `estimate`, the `replicates`, the `standardError` and the `bias`.
 *
 * @example The standard error of a mean
 * // For the mean it approaches the population sd over the root of n: sqrt(2 / 5).
 * const average = (v) => v.reduce((a, b) => a + b, 0) / v.length
 * const b = bootstrap(stream(0), [1, 2, 3, 4, 5], average, 400)
 * print('estimate =', b.estimate)
 * print('standard error =', b.standardError)
 * print('bias =', b.bias)
 */
export function bootstrap(
  s: Stream,
  xData: Data,
  statistic: (sample: Float64Array) => number,
  resamples: Size,
): Bootstrap {
  const x = toSequence(xData, 'bootstrap')
  requireNonEmpty(x, 'bootstrap')
  const n = x.length
  const sample = new Float64Array(n)
  const replicates = new Float64Array(resamples)
  for (let r = 0; r < resamples; r++) {
    const draw = toFlat(integers(child(s, 'resample', r), n, { shape: [n] }))
    for (let i = 0; i < n; i++) sample[i] = x[draw[i]]
    replicates[r] = statistic(sample)
  }
  const estimate = statistic(Float64Array.from(x))
  return {
    estimate,
    replicates: vectorOf(replicates),
    standardError: resamples > 1 ? standardDeviation(replicates, { sample: true }) : NaN,
    bias: resamples > 0 ? mean(replicates) - estimate : NaN,
  }
}

/**
 * A bootstrap confidence interval at `level` (default 0.95) from the replicates' quantiles (method `linear` by
 * default), with $\alpha = 1 - \text{level}$:
 * - `percentile` (default): $[q(\alpha/2), q(1 - \alpha/2)]$ of the replicates.
 * - `basic`: $[2\hat{\theta} - q(1 - \alpha/2), 2\hat{\theta} - q(\alpha/2)]$, reflecting the percentiles about the
 *   estimate $\hat{\theta}$.
 *
 * (Efron and Tibshirani 1993, §13; Davison and Hinkley 1997, §5.2.)
 *
 * @param result The bootstrap, as `bootstrap` returns it (at least one replicate).
 * @param options The level, the kind of interval and the quantile method.
 * @param options.level The coverage, in $(0, 1)$ (default 0.95).
 * @param options.method `percentile` (default) or `basic`, as above.
 * @param options.quantileMethod The quantile method for $q$ (default `linear`; see `QuantileMethod`).
 * @returns The interval as `[lower, upper]`.
 *
 * @example Percentile and basic intervals for the mean of skewed data
 * // The basic interval reflects the percentiles about the estimate, so the long right tail moves it left.
 * const average = (v) => v.reduce((a, b) => a + b, 0) / v.length
 * const b = bootstrap(stream(0), [1, 2, 3, 4, 5, 6, 7, 8, 9, 30], average, 400)
 * print('estimate =', b.estimate)
 * print('percentile 90% =', bootstrapInterval(b, { level: 0.9 }))
 * print('basic 90% =', bootstrapInterval(b, { level: 0.9, method: 'basic' }))
 */
export function bootstrapInterval(
  result: Bootstrap,
  options: { level?: number; method?: 'percentile' | 'basic'; quantileMethod?: QuantileMethod } = {},
): [number, number] {
  const alpha = 1 - (options.level ?? 0.95)
  const [lo, hi] = toFlat(quantile(result.replicates, [alpha / 2, 1 - alpha / 2], options.quantileMethod ?? 'linear'))
  return (options.method ?? 'percentile') === 'percentile'
    ? [lo, hi]
    : [2 * result.estimate - hi, 2 * result.estimate - lo]
}

/** The result of `permutationTest`. */
export type PermutationTest = {
  /** The statistic on the observed split. */
  observed: number
  /** The statistic on each permuted split (rank-1). */
  null: Tensor
  /**
   * The Monte Carlo p-value with the $+1$ correction, so it is never 0 (Phipson and Smyth 2010).
   */
  pValue: number
}

/**
 * A two-sample permutation test: pools $\xvec$ and $\yvec$, shuffles the pool `resamples` times (shuffle $r$ from
 * `child(s, 'permutation', r)`), and recomputes the statistic on each split into the original group sizes. The p-value
 * is $(1 + \#\{\text{null at least as extreme}\}) / (1 + \text{resamples})$: `greater` counts null values
 * $\ge$ the observed one, `less` those $\le$ it, and `two-sided` (default) is
 * $\min(1, 2 \min(p_\text{less}, p_\text{greater}))$, as in `scipy.stats.permutation_test`. Comparisons allow a
 * relative tolerance of $10^{-14}$ so that ties in exact arithmetic count as ties.
 *
 * @param s The random stream whose children draw the permutations; not advanced.
 * @param xData The first sample: an array or a rank-1 tensor.
 * @param yData The second sample: an array or a rank-1 tensor.
 * @param statistic The statistic of a split, given the two groups (views of a scratch array, so it must not keep
 *   them); large values should point away from the null for `greater`.
 * @param options The number of permutations and the alternative.
 * @param options.resamples The number of random permutations (default 9999).
 * @param options.alternative `two-sided` (default), `greater` or `less`, as above.
 * @returns The `observed` statistic, the `null` statistics and the `pValue`.
 *
 * @example A difference in means
 * // Of the 20 ways to split the six values in two groups of three, only the observed one and its mirror image are
 * // this extreme, so the exact two-sided p-value is 2 / 20.
 * const average = (v) => v.reduce((a, b) => a + b, 0) / v.length
 * const t = permutationTest(stream(0), [1, 2, 3], [7, 8, 9], (x, y) => average(y) - average(x), { resamples: 999 })
 * print('observed =', t.observed)
 * print('p-value =', t.pValue)
 */
export function permutationTest(
  s: Stream,
  xData: Data,
  yData: Data,
  statistic: (x: Float64Array, y: Float64Array) => number,
  options: { resamples?: Size; alternative?: 'two-sided' | 'greater' | 'less' } = {},
): PermutationTest {
  const x = toSequence(xData, 'permutationTest')
  const y = toSequence(yData, 'permutationTest')
  const resamples = options.resamples ?? 9999
  const nx = x.length
  const pooled = Float64Array.from([...Array.from(x), ...Array.from(y)])
  const observed = statistic(Float64Array.from(x), Float64Array.from(y))
  const nullValues = new Float64Array(resamples)
  for (let r = 0; r < resamples; r++) {
    const p = shuffledValues(child(s, 'permutation', r), pooled)
    nullValues[r] = statistic(p.subarray(0, nx), p.subarray(nx))
  }
  const tolerance = 1e-14 * Math.abs(observed)
  let atLeast = 0
  let atMost = 0
  for (const v of nullValues) {
    if (v >= observed - tolerance) atLeast++
    if (v <= observed + tolerance) atMost++
  }
  const greater = (atLeast + 1) / (resamples + 1)
  const less = (atMost + 1) / (resamples + 1)
  const alternative = options.alternative ?? 'two-sided'
  const pValue =
    alternative === 'greater' ? greater : alternative === 'less' ? less : Math.min(1, 2 * Math.min(greater, less))
  return { observed, null: vectorOf(nullValues), pValue }
}

/**
 * Kish's effective sample size of importance weights, $(\sum_i w_i)^2 / \sum_i w_i^2$ (Kong 1992; Kish 1965): $n$
 * for equal weights, 1 when one weight dominates. With `log: true` the inputs are log-weights, shifted by their
 * maximum first so that large log-weights do not overflow. Weights need not be normalised. Throws `DomainError` for no
 * weights or a negative one; NaN when every weight is 0.
 *
 * @param weightsData The weights $w_i$ (or their logarithms): an array or a rank-1 tensor.
 * @param options Whether the weights are given as logarithms.
 * @param options.log The inputs are $\log w_i$ (default false).
 * @returns The effective sample size, between 1 and $n$.
 *
 * @example Equal, uneven and log weights
 * print('equal =', importanceEffectiveSampleSize([1, 1, 1, 1]))
 * print('one dominates =', importanceEffectiveSampleSize([1, 0.001, 0.001, 0.001]))
 * print('uneven =', importanceEffectiveSampleSize([1, 1, 2, 4]))
 * // Weights of exp(1000) each would overflow; as logs they do not.
 * print('log weights =', importanceEffectiveSampleSize([1000, 1000, 1000], { log: true }))
 */
export function importanceEffectiveSampleSize(weightsData: Data, options: { log?: boolean } = {}): number {
  const weights = toSequence(weightsData, 'importanceEffectiveSampleSize')
  requireNonEmpty(weights, 'importanceEffectiveSampleSize')
  let shift = 0
  if (options.log) {
    shift = -Infinity
    for (let i = 0; i < weights.length; i++) shift = Math.max(shift, weights[i])
    if (shift === -Infinity) return NaN // every weight is zero
  }
  let s = 0
  let s2 = 0
  for (let i = 0; i < weights.length; i++) {
    const w = options.log ? Math.exp(weights[i] - shift) : weights[i]
    if (w < 0) throw new DomainError('stats', 'stats: importance weights must be non-negative')
    s += w
    s2 += w * w
  }
  return (s * s) / s2
}
