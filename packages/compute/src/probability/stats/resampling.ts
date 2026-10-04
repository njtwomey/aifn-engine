import { child, integers, permutation, type Stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Size } from 'aifn-compute/foundation/contracts'
import { mean, requireNonEmpty, standardDeviation } from './descriptive'
import { toSequence, vectorOf, type Data } from './input'
import { quantile, type QuantileMethod } from './quantile'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * `size` indices drawn uniformly with replacement from 0 … n − 1 (default size n), as an int32 rank-1 tensor: one
 * bootstrap resample.
 */
export function resampleIndices(s: Stream, n: Size, size: Size = n): Tensor {
  return integers(s, n, { shape: [size] })
}

/** The values of x in the order of a uniformly random permutation (`aifn-compute/foundation/random`'s `permutation`). */
function shuffledValues(s: Stream, x: ArrayLike<number>): Float64Array {
  const order = toFlat(permutation(s, x.length))
  return Float64Array.from(order, (j) => x[j])
}

/**
 * A uniformly random permutation of x (drawn by `aifn-compute/foundation/random`'s `permutation`), as a new rank-1 tensor. (The
 * in-place shuffle of an index array is `aifn-compute/foundation/random`'s `shuffle`.)
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
  /** The standard deviation of the replicates (n − 1). */
  standardError: number
  /** mean(replicates) − estimate. */
  bias: number
}

/**
 * The nonparametric bootstrap (Efron 1979): the statistic on `resamples` samples drawn with replacement from x, with
 * resample r drawn from `child(s, 'resample', r)`. Returns the estimate, the replicates, their standard error and the
 * bias estimate.
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
 * A bootstrap confidence interval at `level` (default 0.95) from the replicates' quantiles (method `linear`):
 * - `percentile` (default): [q(α/2), q(1 − α/2)] of the replicates.
 * - `basic`: [2θ̂ − q(1 − α/2), 2θ̂ − q(α/2)], reflecting the percentiles about the estimate θ̂.
 * (Efron and Tibshirani 1993, §13; Davison and Hinkley 1997, §5.2.)
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
  /** The Monte Carlo p-value with the +1 correction, so it is never 0 (Phipson and Smyth 2010). */
  pValue: number
}

/**
 * A two-sample permutation test: pools x and y, shuffles the pool `resamples` times (shuffle r from `child(s, 'permutation', r)`),
 * and recomputes `statistic(x', y')` on each split into the original group sizes. The p-value is
 * (1 + #{null at least as extreme}) / (1 + resamples): `greater` counts null ≥ observed, `less` null ≤ observed, and
 * `two-sided` (default) is min(1, 2 · min(p_less, p_greater)), as in `scipy.stats.permutation_test`. Comparisons
 * allow a relative tolerance of 1e-14 so that ties in exact arithmetic count as ties.
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
 * Kish's effective sample size of importance weights, (Σwᵢ)² / Σwᵢ² (Kong 1992; Kish 1965): n for equal weights, 1
 * when one weight dominates. With `log: true` the inputs are log-weights, shifted by their maximum first so that
 * large log-weights do not overflow. Weights need not be normalised.
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
