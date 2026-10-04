import { describe, expect, it } from 'vitest'
import {
  autocorrelation,
  autocovariance,
  bootstrap,
  bootstrapInterval,
  correlation,
  covariance,
  crossCorrelation,
  crossCovariance,
  ecdf,
  ecdfAt,
  emptyMoments,
  extent,
  histogram,
  importanceEffectiveSampleSize,
  interquartileRange,
  kde,
  kdeBandwidth,
  kendallTau,
  kurtosis,
  median,
  mode,
  momentsMerge,
  momentsPush,
  momentsVariance,
  permutationTest,
  quantile,
  quantileMethods,
  range,
  ranks,
  resampleIndices,
  runningMean,
  runningVariance,
  shuffled,
  skewness,
  spearman,
  standardDeviation,
  standardise,
  weightedMean,
  weightedVariance,
  zScores,
  type BinRule,
  type QuantileMethod,
  type TiePolicy,
} from 'aifn-compute/probability/stats'
import { stream, uniform } from 'aifn-compute/foundation/random'
import * as T from 'aifn-compute/foundation/tensor'
import { fromData, isTensor, slice, tensor, toFlat, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

// mean and variance are the tensor reductions (one definition, in aifn-compute/foundation/tensor); these read
// them as numbers over a sequence, as the statistics below use them.
const num = (v: T.Value) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])
const seq = (x: ArrayLike<number> | Tensor) => (isTensor(x) ? x : tensor(Array.from(x)))
const mean = (x: ArrayLike<number> | Tensor) => num(T.mean(seq(x)))
const variance = (x: ArrayLike<number> | Tensor, { sample = false } = {}) =>
  num(T.variance(seq(x), null, false, sample ? 1 : 0))

type Hist = {
  x: number[]
  bins: number | string | number[]
  range?: [number, number]
  weights?: number[]
  edges: number[]
  counts: number[]
  density: number[]
}
type Fixture = {
  x: number[]
  y: number[]
  w: number[]
  fweights: number[]
  ties: number[]
  tiesY: number[]
  moments: Record<string, number> & { zscore: number[]; zscoreSample: number[]; mode: { value: number; count: number } }
  ranks: Record<TiePolicy, number[]>
  quantiles: Record<string, { x: number[]; q: number[]; methods: Record<QuantileMethod, number[]> }>
  histograms: Record<string, Hist>
  kde: { grid: number[] } & Record<'scott' | 'silverman' | 'weighted', { bandwidth: number; density: number[] }>
  ecdf: { values: number[]; probabilities: number[] }
  series: {
    x: number[]
    y: number[]
    autocovariance: number[]
    autocovarianceAdjusted: number[]
    autocorrelation: number[]
    crossCovariance: number[]
    crossCorrelation: number[]
  }
}
const f = fixture<Fixture>('probability/stats')
const m = f.moments

/** Elementwise closeness with a relative-and-absolute tolerance. */
/** The values of an array result: a tensor's elements, or the array itself. */
const flat = (x: Tensor | ArrayLike<number>): ArrayLike<number> => (isTensor(x) ? toFlat(x) : x)

function expectClose(actualData: Tensor | ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-12) {
  const actual = flat(actualData)
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i]
    expect(Math.abs(actual[i] - e), `index ${i}: ${actual[i]} vs ${e}`).toBeLessThanOrEqual(
      tolerance * Math.max(1, Math.abs(e)),
    )
  }
}

describe('moments against numpy/scipy', () => {
  it('mean and variance (the tensor reductions) and standard deviation', () => {
    expect(mean(f.x)).toBeCloseTo(m.mean, 13)
    expect(variance(f.x)).toBeCloseTo(m.var, 12)
    expect(variance(f.x, { sample: true })).toBeCloseTo(m.varSample, 12)
    expect(standardDeviation(f.x)).toBeCloseTo(m.std, 12)
    expect(standardDeviation(Float64Array.from(f.x), { sample: true })).toBeCloseTo(m.stdSample, 12)
    expect(standardDeviation([3], { sample: true })).toBeNaN()
    expect(() => standardDeviation([])).toThrow(/empty/)
    expect(mean([1e9 + 0.1, 1e9 + 0.2, 1e9 + 0.3])).toBeCloseTo(1e9 + 0.2, 6)
  })
  it('weighted mean and variance under the three weight meanings', () => {
    expect(weightedMean(f.x, f.w)).toBeCloseTo(m.weightedMean, 12)
    expect(weightedVariance(f.x, f.w)).toBeCloseTo(m.weightedVarPopulation, 12)
    expect(weightedVariance(f.x, f.w, { weights: 'reliability' })).toBeCloseTo(m.weightedVarReliability, 12)
    expect(weightedVariance(f.x, f.fweights, { weights: 'frequency' })).toBeCloseTo(m.weightedVarFrequency, 12)
    expect(() => weightedMean([1, 2], [0, 0])).toThrow(/positive sum/)
    expect(() => weightedMean([1, 2], [1, -1])).toThrow(/non-negative/)
  })
  it('skewness and kurtosis, biased and corrected', () => {
    expect(skewness(f.x)).toBeCloseTo(m.skew, 12)
    expect(skewness(f.x, { biasCorrected: true })).toBeCloseTo(m.skewUnbiased, 12)
    expect(kurtosis(f.x)).toBeCloseTo(m.kurtosis, 12)
    expect(kurtosis(f.x, { excess: false })).toBeCloseTo(m.kurtosisPearson, 12)
    expect(kurtosis(f.x, { biasCorrected: true })).toBeCloseTo(m.kurtosisUnbiased, 12)
    expect(skewness([2, 2, 2])).toBeNaN()
  })
  it('covariance and Pearson correlation', () => {
    expect(covariance(f.x, f.y)).toBeCloseTo(m.cov, 12)
    expect(covariance(f.x, f.y, { sample: true })).toBeCloseTo(m.covSample, 12)
    expect(correlation(f.x, f.y)).toBeCloseTo(m.pearson, 13)
    expect(correlation([1, 2, 3], [2, 4, 6])).toBe(1)
    expect(correlation([1, 1, 1], [2, 4, 6])).toBeNaN()
    expect(() => covariance([1, 2], [1])).toThrow(/equal length/)
  })
  it('min, max, extent, range and mode', () => {
    expect(extent(f.x)).toEqual([Math.min(...f.x), Math.max(...f.x)])
    expect(extent([3, -1, 2])).toEqual([-1, 3])
    expect(range([3, -1, 2])).toBe(4)
    expect(range([1, NaN, 0])).toBeNaN()
    expect(mode(f.ties)).toEqual(m.mode)
    expect(mode([2, 1, 2, 1])).toEqual({ value: 1, count: 2 })
  })
  it('z-scores and standardise', () => {
    expectClose(zScores(f.x), f.moments.zscore, 1e-12)
    expectClose(zScores(f.x, { sample: true }), f.moments.zscoreSample, 1e-12)
    const s = standardise(f.x)
    expect(s.constant).toBe(false)
    expect(mean(s.values)).toBeCloseTo(0, 13)
    expect(standardDeviation(s.values)).toBeCloseTo(1, 13)
    expect(flat(s.values)[3] * s.scale + s.mean).toBeCloseTo(f.x[3], 12)
    const c = standardise([4, 4])
    expect(c.constant).toBe(true)
    expect(flat(c.values)[0]).toBeNaN()
  })
})

describe('quantile', () => {
  it('matches numpy.quantile for all thirteen methods', () => {
    expect(quantileMethods.length).toBe(13)
    for (const [name, c] of Object.entries(f.quantiles)) {
      for (const method of quantileMethods) {
        const got = flat(quantile(c.x, c.q, method))
        const want = c.methods[method]
        for (let i = 0; i < want.length; i++) expect(got[i], `${name} ${method} q=${c.q[i]}`).toBeCloseTo(want[i], 12)
      }
    }
  })
  it('returns a number for a number, defaults to linear, and median and IQR agree with numpy', () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5)
    expect(median(f.x)).toBeCloseTo(m.median, 13)
    expect(interquartileRange(f.x)).toBeCloseTo(m.iqr, 12)
    expect(quantile([1, NaN, 3], 0.5)).toBeNaN()
    expect(() => quantile([1], 1.5)).toThrow(/outside/)
  })
})

describe('ranks and rank correlations', () => {
  it('ranks match scipy.stats.rankdata for every tie policy', () => {
    for (const policy of ['average', 'min', 'max', 'dense', 'ordinal'] as const)
      expect(toFlat(ranks(f.ties, policy)), policy).toEqual(f.ranks[policy])
    expect(toFlat(ranks([10, 20, 20]))).toEqual([1, 2.5, 2.5])
  })
  it('Spearman and Kendall match scipy, with and without ties', () => {
    expect(spearman(f.x, f.y)).toBeCloseTo(m.spearman, 13)
    expect(spearman(f.ties, f.tiesY)).toBeCloseTo(m.spearmanTies, 13)
    expect(kendallTau(f.x, f.y)).toBeCloseTo(m.kendall, 13)
    expect(kendallTau(f.ties, f.tiesY)).toBeCloseTo(m.kendallTies, 13)
    expect(kendallTau([1, 2, 3], [3, 2, 1])).toBe(-1)
    expect(kendallTau([1, 1], [1, 2])).toBeNaN()
  })
})

describe('histogram', () => {
  it('matches numpy.histogram for each bin rule', () => {
    for (const [name, h] of Object.entries(f.histograms)) {
      const got = histogram(h.x, { bins: h.bins as BinRule, range: h.range, weights: h.weights })
      expectClose(got.edges, h.edges, 1e-13)
      expectClose(got.counts, h.counts, 1e-13)
      expectClose(got.density, h.density, 1e-12)
      if (!h.range && typeof h.bins !== 'object') expect(got.dropped, name).toBe(0)
    }
  })
  it('counts the last edge in the last bin and reports dropped values', () => {
    const h = histogram([0, 1, 2, 3, 4, 9, NaN], { bins: [0, 2, 4] })
    expect(toFlat(h.counts)).toEqual([2, 3])
    expect(h.dropped).toBe(2)
  })
  it('bins by width', () => {
    const h = histogram([0, 0.4, 1.1, 2.5], { bins: { width: 1 } })
    const edges = toFlat(h.edges)
    expect(edges).toEqual([0, 1, 2, 3])
    expect(toFlat(h.counts)).toEqual([2, 1, 1])
    expect(toFlat(h.density).reduce((a, d, k) => a + d * (edges[k + 1] - edges[k]), 0)).toBeCloseTo(1, 14)
  })
  it('rejects bad rules', () => {
    expect(() => histogram([1, 2], { bins: [1, 1] })).toThrow(/increase/)
    expect(() => histogram([1, 2], { bins: 0 })).toThrow(/positive integer/)
    expect(() => histogram([1, NaN])).toThrow(/not finite/)
  })
})

describe('empirical CDF', () => {
  it('matches scipy.stats.ecdf', () => {
    const e = ecdf(f.ties)
    expect(toFlat(e.values)).toEqual(f.ecdf.values)
    expectClose(e.probabilities, f.ecdf.probabilities, 1e-15)
  })
  it('evaluates at arbitrary points', () => {
    expect(toFlat(ecdfAt([3, 1, 2, 2], [0, 1, 1.5, 2, 3, 4]))).toEqual([0, 0.25, 0.25, 0.75, 1, 1])
  })
})

describe('kde', () => {
  it('matches scipy.stats.gaussian_kde for Scott, Silverman and weights', () => {
    for (const rule of ['scott', 'silverman'] as const) {
      const got = kde(f.x, f.kde.grid, { bandwidth: rule })
      expect(got.bandwidth).toBeCloseTo(f.kde[rule].bandwidth, 13)
      expectClose(got.density, f.kde[rule].density, 1e-12)
    }
    const weighted = kde(f.x, f.kde.grid, { weights: f.w })
    expect(weighted.bandwidth).toBeCloseTo(f.kde.weighted.bandwidth, 13)
    expectClose(weighted.density, f.kde.weighted.density, 1e-12)
  })
  it('takes a numeric bandwidth and flags constant data', () => {
    expect(kdeBandwidth(f.x, 0.3)).toBe(0.3)
    expect(toFlat(kde([0], [0], { bandwidth: 1 }).density)[0]).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 15)
    const flat = kde([1, 1, 1], [1])
    expect(flat.degenerate).toBe(true)
    expect(toFlat(flat.density)[0]).toBeNaN()
  })
})

describe('running moments (Welford)', () => {
  it('prefix means and variances equal the direct ones', () => {
    const rm = toFlat(runningMean(f.x))
    const rv = toFlat(runningVariance(f.x))
    const rs = toFlat(runningVariance(f.x, { sample: true }))
    for (const i of [0, 1, 5, 36]) {
      const prefix = f.x.slice(0, i + 1)
      expect(rm[i]).toBeCloseTo(mean(prefix), 12)
      expect(rv[i]).toBeCloseTo(variance(prefix), 12)
      if (i > 0) expect(rs[i]).toBeCloseTo(variance(prefix, { sample: true }), 12)
    }
    expect(rs[0]).toBeNaN()
  })
  it('stays accurate with a large offset', () => {
    const shifted = f.x.map((v) => v + 1e9)
    expect(toFlat(runningVariance(shifted)).at(-1)!).toBeCloseTo(m.var, 5)
  })
  it('merging chunks equals one pass', () => {
    const acc = (xs: number[]) => xs.reduce(momentsPush, emptyMoments)
    const merged = momentsMerge(acc(f.x.slice(0, 11)), acc(f.x.slice(11)))
    const whole = acc(f.x)
    expect(merged.count).toBe(37)
    expect(merged.mean).toBeCloseTo(whole.mean, 13)
    expect(momentsVariance(merged, { sample: true })).toBeCloseTo(m.varSample, 12)
    expect(momentsMerge(emptyMoments, whole)).toBe(whole)
  })
})

describe('autocovariance and cross-correlation', () => {
  const s = f.series
  it('direct and FFT agree with direct sums', () => {
    for (const method of ['direct', 'fft', 'auto'] as const) {
      expectClose(autocovariance(s.x, { maxLag: 20, method }), s.autocovariance, 1e-11)
      expectClose(autocorrelation(s.x, { maxLag: 20, method }), s.autocorrelation, 1e-11)
      expectClose(autocovariance(s.x, { maxLag: 20, method, adjusted: true }), s.autocovarianceAdjusted, 1e-11)
    }
    expect(toFlat(autocorrelation(s.x))[0]).toBe(1)
    expect(autocovariance(s.x).shape).toEqual([80])
  })
  it('FFT matches direct on a long series at every lag', () => {
    const long = toFlat(uniform(stream(3), 0, 1, { shape: [1000] }))
    expectClose(autocovariance(long, { method: 'fft' }), toFlat(autocovariance(long, { method: 'direct' })), 1e-10)
  })
  it('cross-covariance and cross-correlation over negative and positive lags', () => {
    for (const method of ['direct', 'fft'] as const) {
      const c = crossCovariance(s.x, s.y, { maxLag: 6, method })
      expect(c.lags.dtype).toBe('int32')
      expect(toFlat(c.lags)).toEqual([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
      expectClose(c.values, s.crossCovariance, 1e-11)
      expectClose(crossCorrelation(s.x, s.y, { maxLag: 6, method }).values, s.crossCorrelation, 1e-11)
    }
    // y is x delayed by 3 steps plus noise, so the peak is at lag +3.
    const r = crossCorrelation(s.x, s.y, { maxLag: 6 })
    const values = toFlat(r.values)
    expect(toFlat(r.lags)[values.indexOf(Math.max(...values))]).toBe(3)
  })
})

describe('resampling', () => {
  it('resample indices and shuffles are in range and deterministic', () => {
    const t = resampleIndices(stream(1), 10, 1000)
    expect(t.dtype).toBe('int32')
    const a = toFlat(t)
    expect(Math.min(...a)).toBe(0)
    expect(Math.max(...a)).toBe(9)
    expect(toFlat(resampleIndices(stream(1), 10, 1000))).toEqual(a)
    const p = shuffled(stream(2), [1, 2, 3, 4, 5])
    expect(toFlat(p).sort()).toEqual([1, 2, 3, 4, 5])
    expect(toFlat(shuffled(stream(2), [1, 2, 3, 4, 5]))).toEqual(toFlat(p))
  })
  it('bootstrap standard error of the mean is near s/√n and is reproducible', () => {
    const b = bootstrap(stream(11), f.x, mean, 4000)
    expect(b.estimate).toBeCloseTo(m.mean, 13)
    expect(b.replicates.shape).toEqual([4000])
    const expected = m.std / Math.sqrt(f.x.length)
    expect(Math.abs(b.standardError - expected) / expected).toBeLessThan(0.05)
    expect(Math.abs(b.bias)).toBeLessThan(0.1 * expected)
    expect(toFlat(bootstrap(stream(11), f.x, mean, 50).replicates)).toEqual(toFlat(b.replicates).slice(0, 50))
  })
  it('percentile and basic intervals', () => {
    const b = bootstrap(stream(5), f.x, median, 2000)
    const [lo, hi] = bootstrapInterval(b)
    expect(lo).toBe(quantile(b.replicates, 0.025))
    expect(hi).toBe(quantile(b.replicates, 0.975))
    const [blo, bhi] = bootstrapInterval(b, { method: 'basic', level: 0.9 })
    expect(blo).toBeCloseTo(2 * b.estimate - quantile(b.replicates, 0.95), 14)
    expect(bhi).toBeCloseTo(2 * b.estimate - quantile(b.replicates, 0.05), 14)
  })
  it('permutation p-values approach the exact enumeration', () => {
    const x = [1.2, 2.9, 3.1]
    const y = [2.5, 4.4, 5.0, 6.1]
    const diff = (a: ArrayLike<number>, b: ArrayLike<number>) => mean(b) - mean(a)
    // Exact null: all C(7, 3) = 35 ways to choose x's positions.
    const pool = [...x, ...y]
    const exact: number[] = []
    for (let i = 0; i < 7; i++)
      for (let j = i + 1; j < 7; j++)
        for (let k = j + 1; k < 7; k++) {
          const xs = [pool[i], pool[j], pool[k]]
          exact.push(
            diff(
              xs,
              pool.filter((_, q) => q !== i && q !== j && q !== k),
            ),
          )
        }
    const observed = diff(x, y)
    const pGreater = exact.filter((v) => v >= observed - 1e-12).length / exact.length
    const pLess = exact.filter((v) => v <= observed + 1e-12).length / exact.length
    const greater = permutationTest(stream(9), x, y, diff, { resamples: 20000, alternative: 'greater' })
    expect(greater.observed).toBeCloseTo(observed, 14)
    expect(Math.abs(greater.pValue - pGreater)).toBeLessThan(0.01)
    const less = permutationTest(stream(9), x, y, diff, { resamples: 20000, alternative: 'less' })
    expect(Math.abs(less.pValue - pLess)).toBeLessThan(0.01)
    const two = permutationTest(stream(9), x, y, diff, { resamples: 20000 })
    expect(two.pValue).toBeCloseTo(Math.min(1, 2 * Math.min(greater.pValue, less.pValue)), 12)
    expect(permutationTest(stream(1), x, y, diff, { resamples: 10 }).pValue).toBeGreaterThan(0)
  })
  it('importance effective sample size', () => {
    expect(importanceEffectiveSampleSize([1, 1, 1, 1])).toBe(4)
    expect(importanceEffectiveSampleSize([0, 5, 0])).toBe(1)
    expect(importanceEffectiveSampleSize([1, 2, 3])).toBeCloseTo(36 / 14, 14)
    expect(importanceEffectiveSampleSize([0, Math.log(2), Math.log(3)], { log: true })).toBeCloseTo(36 / 14, 13)
    expect(importanceEffectiveSampleSize([1000, 1000, 1000 + Math.log(2)], { log: true })).toBeCloseTo(16 / 6, 12)
    expect(importanceEffectiveSampleSize([-Infinity, -Infinity], { log: true })).toBeNaN()
  })
})

describe('tensor inputs', () => {
  // Deterministic data with no ties.
  const flat = (n: number, seed = 1) => Array.from({ length: n }, (_, k) => Math.sin(12.9898 * (k + seed)) * 3 + k / n)
  const x = flat(24)
  const y = flat(24, 7)
  const tx = tensor(x)
  const ty = tensor(y)

  it('rank-1 tensors give the same results as arrays', () => {
    expect(standardDeviation(tx)).toBe(standardDeviation(x))
    expect(range(tx)).toEqual(range(x))
    expect(extent(tx)).toEqual(extent(x))
    expect(skewness(tx, { biasCorrected: true })).toBe(skewness(x, { biasCorrected: true }))
    expect(kurtosis(tx)).toBe(kurtosis(x))
    expect(median(tx)).toBe(median(x))
    expect(quantile(tx, tensor([0.1, 0.9]))).toEqual(quantile(x, [0.1, 0.9]))
    expect(interquartileRange(tx, 'hazen')).toBe(interquartileRange(x, 'hazen'))
    expect(weightedMean(tx, tensor(y.map(Math.abs)))).toBe(weightedMean(x, y.map(Math.abs)))
    expect(covariance(tx, ty)).toBe(covariance(x, y))
    expect(correlation(tx, ty)).toBe(correlation(x, y))
    expect(zScores(tx)).toEqual(zScores(x))
    expect(ranks(tx)).toEqual(ranks(x))
    expect(spearman(tx, ty)).toBe(spearman(x, y))
    expect(kendallTau(tx, ty)).toBe(kendallTau(x, y))
    expect(runningVariance(tx)).toEqual(runningVariance(x))
    expect(autocorrelation(tx, { maxLag: 5 })).toEqual(autocorrelation(x, { maxLag: 5 }))
    expect(crossCorrelation(tx, ty, { maxLag: 3 })).toEqual(crossCorrelation(x, y, { maxLag: 3 }))
    expect(histogram(tx, { bins: tensor([-4, 0, 4]) })).toEqual(histogram(x, { bins: [-4, 0, 4] }))
    expect(ecdfAt(tx, tensor([0, 1]))).toEqual(ecdfAt(x, [0, 1]))
    expect(kde(tx, tensor([0, 1]))).toEqual(kde(x, [0, 1]))
    expect(importanceEffectiveSampleSize(tensor(y.map(Math.abs)))).toBe(importanceEffectiveSampleSize(y.map(Math.abs)))
  })

  it('int32 tensors are read as numbers', () => {
    const counts = tensor([3, 1, 4, 1, 5])
    const ints = fromData(Int32Array.of(3, 1, 4, 1, 5))
    expect(standardDeviation(ints)).toBe(standardDeviation(counts))
    expect(mode(ints)).toEqual({ value: 1, count: 2 })
  })

  it('reductions without an axis cover every element of any rank', () => {
    const m = tensor(x, [2, 3, 4])
    expect(skewness(m)).toBe(skewness(x))
    expect(median(m)).toBe(median(x))
    expect(quantile(m, 0.3, 'weibull')).toBe(quantile(x, 0.3, 'weibull'))
  })

  it('sequence functions need rank-1 tensors', () => {
    expect(() => ranks(tensor(x, [4, 6]))).toThrow(/rank-1/)
    expect(() => correlation(tensor(x, [4, 6]), ty)).toThrow(/rank-1/)
    expect(() => median(x, { axis: 0 } as never)).toThrow(/tensor/)
  })

  /** Every lane along `axis`, as arrays, in row-major order of the other axes. */
  function lanes(values: number[], shape: number[], axis: number): number[][] {
    const a = axis < 0 ? axis + shape.length : axis
    const strides = shape.map((_, k) => shape.slice(k + 1).reduce((p, d) => p * d, 1))
    const others = shape.map((_, k) => k).filter((k) => k !== a)
    const count = others.reduce((p, k) => p * shape[k], 1)
    const out: number[][] = []
    for (let i = 0; i < count; i++) {
      let rest = i
      let base = 0
      for (let j = others.length - 1; j >= 0; j--) {
        const k = others[j]
        base += (rest % shape[k]) * strides[k]
        rest = Math.floor(rest / shape[k])
      }
      out.push(Array.from({ length: shape[a] }, (_, t) => values[base + t * strides[a]]))
    }
    return out
  }

  const reductions: [string, (v: number[]) => number, (t: Tensor, axis: number, keepDims: boolean) => Tensor][] = [
    ['sd', (v) => standardDeviation(v), (t, axis, keepDims) => standardDeviation(t, { axis, keepDims })],
    ['range', (v) => range(v), (t, axis, keepDims) => range(t, { axis, keepDims })],
    ['skewness', (v) => skewness(v), (t, axis, keepDims) => skewness(t, { axis, keepDims })],
    [
      'kurtosis',
      (v) => kurtosis(v, { excess: false }),
      (t, axis, keepDims) => kurtosis(t, { axis, keepDims, excess: false }),
    ],
    ['median', (v) => median(v), (t, axis, keepDims) => median(t, { axis, keepDims })],
    [
      'quantile',
      (v) => quantile(v, 0.3, 'hazen'),
      (t, axis, keepDims) => quantile(t, 0.3, { axis, keepDims, method: 'hazen' }),
    ],
    [
      'interquartileRange',
      (v) => interquartileRange(v),
      (t, axis, keepDims) => interquartileRange(t, { axis, keepDims }),
    ],
  ]

  const shapes = [[24], [4, 6], [2, 3, 4]]
  for (const [name, one, along] of reductions) {
    it(`${name} along an axis equals the statistic of each lane`, () => {
      for (const shape of shapes) {
        const t = tensor(x, shape)
        for (let axis = -shape.length; axis < shape.length; axis++) {
          const a = axis < 0 ? axis + shape.length : axis
          const want = lanes(x, shape, axis).map(one)
          const got = along(t, axis, false)
          expect(got.shape).toEqual(shape.filter((_, k) => k !== a))
          expect(toFlat(got)).toEqual(want)
          const kept = along(t, axis, true)
          expect(kept.shape).toEqual(shape.map((d, k) => (k === a ? 1 : d)))
          expect(toFlat(kept)).toEqual(want)
        }
      }
    })
  }

  it('reads strided (transposed) tensors through their strides', () => {
    const m = tensor(x, [4, 6])
    const t = transpose(m) // [6, 4], not contiguous
    const rows = lanes(x, [4, 6], 0) // columns of m = rows of t
    expect(toFlat(skewness(t, { axis: 1 }))).toEqual(rows.map((r) => skewness(r)))
    expect(toFlat(median(t, { axis: -1 }))).toEqual(rows.map((r) => median(r)))
    expect(ranks(slice(m, null, 2))).toEqual(ranks(lanes(x, [4, 6], 0)[2]))
    expect(median(t)).toBe(median(x))
    expect(() => median(t, { axis: 2 })).toThrow(/out of range/)
  })

  it('resampling takes tensor data and draws only from the stream it is given', () => {
    expect(bootstrap(stream('boot'), tx, (v) => mean(v), 50)).toEqual(bootstrap(stream('boot'), x, (v) => mean(v), 50))
    expect(resampleIndices(stream(3), 10)).toEqual(resampleIndices(stream(3), 10))
    expect(shuffled(stream('shuffle'), tx)).toEqual(shuffled(stream('shuffle'), x))
    const diff = (u: ArrayLike<number>, v: ArrayLike<number>) => mean(u) - mean(v)
    expect(permutationTest(stream('p'), tx, ty, diff, { resamples: 99 })).toEqual(
      permutationTest(stream('p'), x, y, diff, { resamples: 99 }),
    )
  })
})
