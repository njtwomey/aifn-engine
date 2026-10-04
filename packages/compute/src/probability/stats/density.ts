import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { allValues, toSequence, vectorOf, type Data } from './input'
import { requireNonEmpty, requireSameLength, weightedVariance, variance } from './descriptive'
import { interquartileRange, sortedValues } from './quantile'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * How `histogram` chooses its bins. Every rule gives equal-width bins over [first, last] (the data's min and max, or
 * `range`), except explicit edges.
 * - a number: that many bins.
 * - `{ width }`: bins of that width starting at `first`; the last edge is the first multiple at or past `last`.
 * - `sturges`: width = (last − first) / (log₂ n + 1) (Sturges 1926), then ⌈(last − first) / width⌉ bins.
 * - `freedman-diaconis`: width = 2·IQR·n^{−1/3} (Freedman and Diaconis 1981); one bin when the IQR is 0.
 * - an array or rank-1 tensor: explicit, strictly increasing edges.
 * The number, `sturges` and `freedman-diaconis` rules match `numpy.histogram_bin_edges` (`bins=k`, `'sturges'`,
 * `'fd'`).
 */
export type BinRule = number | { width: number } | 'sturges' | 'freedman-diaconis' | Data

/** A histogram: `edges` (rank 1) has one more entry than `counts` and `density`. */
export type Histogram = {
  edges: Tensor
  /** Values (or summed weights) per bin. Bin i is [edges[i], edges[i+1]), except the last, which includes its right edge. */
  counts: Tensor
  /** counts / (total counted × bin width): integrates to 1 over the bins. */
  density: Tensor
  /** Values (or weight) outside the edges, or NaN, which were not counted. */
  dropped: number
}

/** numpy's `linspace(first, last, count + 1)`: i·step + first, with the last edge exactly `last`. */
function evenEdges(first: number, last: number, count: number): Float64Array {
  const step = (last - first) / count
  const edges = Float64Array.from({ length: count + 1 }, (_, i) => i * step + first)
  edges[count] = last
  return edges
}

/**
 * Bins the values x and returns `{ edges, counts, density, dropped }`. Values equal to the last edge fall in the last
 * bin (the aifn convention, as in numpy); values outside `range` or the explicit edges, and NaN, are dropped and
 * reported in `dropped`, never clipped into the end bins. `weights` replace unit counts. The default rule is 10 bins,
 * as in numpy. Matches `numpy.histogram`.
 */
export function histogram(
  xData: Data,
  options: { bins?: BinRule; range?: [number, number]; weights?: Data } = {},
): Histogram {
  const x = allValues(xData)
  const weights = options.weights === undefined ? undefined : toSequence(options.weights, 'histogram')
  const { bins = 10 } = options
  if (weights) requireSameLength(x, weights, 'histogram')
  let edges: Float64Array
  let uniform = true
  if (typeof bins === 'object' && !('width' in bins)) {
    edges = Float64Array.from(toSequence(bins, 'histogram edges'))
    if (edges.length < 2) throw new DomainError('stats', 'stats: histogram needs at least two edges')
    for (let i = 1; i < edges.length; i++)
      if (!(edges[i] > edges[i - 1])) throw new DomainError('stats', 'stats: histogram edges must increase strictly')
    uniform = false
  } else {
    let [first, last] = options.range ?? [Infinity, -Infinity]
    if (!options.range) {
      for (let i = 0; i < x.length; i++) {
        // As numpy: without an explicit range, NaN makes the range undefined.
        if (Number.isNaN(x[i]))
          throw new DomainError('stats', 'stats: histogram range is not finite (the data contain NaN)')
        if (x[i] < first) first = x[i]
        if (x[i] > last) last = x[i]
      }
      if (x.length === 0) [first, last] = [0, 1]
    }
    if (!Number.isFinite(first) || !Number.isFinite(last) || first > last)
      throw new DomainError('stats', `stats: histogram range [${first}, ${last}] is not finite and increasing`)
    // Rules estimate a width from the data inside the range, as numpy does.
    const inside = options.range ? Array.from(x).filter((v) => v >= first && v <= last) : x
    let lowest = Infinity
    let highest = -Infinity
    for (let i = 0; i < inside.length; i++) {
      if (inside[i] < lowest) lowest = inside[i]
      if (inside[i] > highest) highest = inside[i]
    }
    const spread = inside.length ? highest - lowest : 0
    if (first === last) {
      first -= 0.5
      last += 0.5
    }
    let count: number
    if (typeof bins === 'number') {
      if (!(bins >= 1) || !Number.isInteger(bins))
        throw new DomainError('stats', 'stats: the number of bins must be a positive integer')
      count = bins
    } else if (typeof bins === 'object') {
      if (!(bins.width > 0)) throw new DomainError('stats', 'stats: the bin width must be positive')
      count = Math.max(1, Math.ceil((last - first) / bins.width))
      last = first + count * bins.width
    } else {
      const width =
        inside.length === 0
          ? 0
          : bins === 'sturges'
            ? spread / (Math.log2(inside.length) + 1)
            : 2 * interquartileRange(inside) * inside.length ** (-1 / 3)
      count = width > 0 ? Math.ceil((last - first) / width) : 1
    }
    edges = evenEdges(first, last, count)
  }

  const nBins = edges.length - 1
  const first = edges[0]
  const last = edges[nBins]
  const counts = new Float64Array(nBins)
  let dropped = 0
  for (let i = 0; i < x.length; i++) {
    const v = x[i]
    const w = weights ? weights[i] : 1
    if (!(v >= first && v <= last)) {
      dropped += w
      continue
    }
    let k: number
    if (uniform) {
      // numpy's fast path: compute the bin, then correct it against the edges for rounding.
      k = Math.floor(((v - first) / (last - first)) * nBins)
      if (k === nBins) k--
      if (v < edges[k]) k--
      else if (k !== nBins - 1 && v >= edges[k + 1]) k++
    } else {
      // Binary search for the last edge ≤ v; the last edge itself belongs to the last bin.
      let lo = 0
      let hi = nBins
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1
        if (edges[mid] <= v) lo = mid
        else hi = mid
      }
      k = lo
    }
    counts[k] += w
  }
  let total = 0
  for (let k = 0; k < nBins; k++) total += counts[k]
  const density = Float64Array.from(counts, (c, k) => c / total / (edges[k + 1] - edges[k]))
  return { edges: vectorOf(edges), counts: vectorOf(counts), density: vectorOf(density), dropped }
}

/**
 * The empirical CDF as a step function: the distinct sorted values and F̂(v) = #{xᵢ ≤ v} / n at each. Matches
 * `scipy.stats.ecdf(x).cdf` (`quantiles`, `probabilities`).
 */
export function ecdf(xData: Data): { values: Tensor; probabilities: Tensor } {
  const x = allValues(xData)
  requireNonEmpty(x, 'ecdf')
  const s = sortedValues(x)
  const values: number[] = []
  const probabilities: number[] = []
  for (let i = 0; i < s.length; i++) {
    if (i + 1 < s.length && s[i + 1] === s[i]) continue
    values.push(s[i])
    probabilities.push((i + 1) / s.length)
  }
  return { values: vectorOf(values), probabilities: vectorOf(probabilities) }
}

/** The empirical CDF of x evaluated at each point t: #{xᵢ ≤ t} / n. */
export function ecdfAt(xData: Data, tData: Data): Tensor {
  const x = toSequence(xData, 'ecdfAt')
  const t = toSequence(tData, 'ecdfAt')
  requireNonEmpty(x, 'ecdfAt')
  const s = sortedValues(x)
  return vectorOf(
    Float64Array.from(t, (v) => {
      // Count of sorted values ≤ v by binary search (upper bound).
      let lo = 0
      let hi = s.length
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (s[mid] <= v) lo = mid + 1
        else hi = mid
      }
      return lo / s.length
    }),
  )
}

/**
 * A bandwidth rule for the Gaussian KDE, in scipy's form h = factor × σ̂, with σ̂ the sample standard deviation
 * (n − 1) and n the (effective) sample size:
 * - `scott`: factor n^{−1/5} (Scott 1992).
 * - `silverman`: factor (3n/4)^{−1/5} (Silverman 1986, eq. 3.28 in scipy's form).
 * - a number: the bandwidth h itself (the kernel's standard deviation).
 */
export type BandwidthRule = 'scott' | 'silverman' | number

/**
 * The Gaussian KDE bandwidth h (the kernel's standard deviation) for a sample, as `scipy.stats.gaussian_kde`
 * computes `sqrt(covariance)`. With weights, n is Kish's effective size (Σw)²/Σw² and σ̂² the reliability-weighted
 * variance, as in scipy.
 */
export function kdeBandwidth(xData: Data, rule: BandwidthRule = 'scott', weightsData?: Data): number {
  const x = toSequence(xData, 'kdeBandwidth')
  const weights = weightsData === undefined ? undefined : toSequence(weightsData, 'kdeBandwidth')
  requireNonEmpty(x, 'kdeBandwidth')
  if (typeof rule === 'number') {
    if (!(rule > 0)) throw new DomainError('stats', 'stats: a KDE bandwidth must be positive')
    return rule
  }
  const n = weights ? importanceSize(weights) : x.length
  const sd = Math.sqrt(
    weights ? weightedVariance(x, weights, { weights: 'reliability' }) : variance(x, { sample: true }),
  )
  const factor = rule === 'scott' ? n ** (-1 / 5) : ((n * 3) / 4) ** (-1 / 5)
  return factor * sd
}

function importanceSize(w: ArrayLike<number>): number {
  let s = 0
  let s2 = 0
  for (let i = 0; i < w.length; i++) {
    s += w[i]
    s2 += w[i] * w[i]
  }
  return (s * s) / s2
}

/**
 * A Gaussian kernel density estimate of the sample x, evaluated at the points `at`:
 * f̂(t) = Σᵢ wᵢ φ((t − xᵢ)/h) / h with weights normalised to sum to 1 (uniform by default) and h from `bandwidth`
 * (default `scott`). Returns the densities (a rank-1 tensor) and the bandwidth used. Matches `scipy.stats.gaussian_kde(x, weights=w)`
 * evaluated at `at`. `degenerate` is true when h is 0 (constant data); the densities are then NaN.
 */
export function kde(
  xData: Data,
  atData: Data,
  options: { bandwidth?: BandwidthRule; weights?: Data } = {},
): { density: Tensor; bandwidth: number; degenerate: boolean } {
  const x = toSequence(xData, 'kde')
  const at = toSequence(atData, 'kde')
  const weights = options.weights === undefined ? undefined : toSequence(options.weights, 'kde')
  if (weights) requireSameLength(x, weights, 'kde')
  const h = kdeBandwidth(x, options.bandwidth ?? 'scott', weights)
  let total = 0
  if (weights) for (let i = 0; i < weights.length; i++) total += weights[i]
  const norm = 1 / (h * Math.sqrt(2 * Math.PI))
  const density = Float64Array.from(at, (t) => {
    let s = 0
    for (let i = 0; i < x.length; i++) {
      const z = (t - x[i]) / h
      s += (weights ? weights[i] / total : 1 / x.length) * Math.exp(-0.5 * z * z)
    }
    return s * norm
  })
  return { density: vectorOf(density), bandwidth: h, degenerate: !(h > 0) }
}

/** The sample covariance (n − 1) of row-major points [n × d]. */
function sampleCovariance(x: ArrayLike<number>, n: number, d: number): Float64Array {
  const mu = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += x[i * d + j] / n
  const cov = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++)
      for (let b = 0; b <= a; b++) cov[a * d + b] += ((x[i * d + a] - mu[a]) * (x[i * d + b] - mu[b])) / (n - 1)
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) cov[b * d + a] = cov[a * d + b]
  return cov
}

/**
 * The kernel N(0, c²Σ̂) of a sample: its covariance, log of 1/(n (2π)^{d/2} |c²Σ̂|^{1/2}), the whitened sample L⁻¹xᵢ
 * (LLᵀ = c²Σ̂) and the whitening map, so a quadratic form is a squared distance between whitened points.
 */
function whitened(x: ArrayLike<number>, n: number, d: number, factor: number) {
  const cov = sampleCovariance(x, n, d).map((v) => v * factor * factor)
  const L = new Float64Array(d * d)
  for (let a = 0; a < d; a++)
    for (let b = 0; b <= a; b++) {
      let s = cov[a * d + b]
      for (let k = 0; k < b; k++) s -= L[a * d + k] * L[b * d + k]
      if (a === b) {
        if (!(s > 0))
          throw new NumericalError(
            'multivariateKde',
            'multivariateKde: the sample covariance is singular (points on a subspace)',
            'singular',
          )
        L[a * d + a] = Math.sqrt(s)
      } else L[a * d + b] = s / L[b * d + b]
    }
  let logDet = 0
  for (let a = 0; a < d; a++) logDet += 2 * Math.log(L[a * d + a])
  const logNorm = -0.5 * (d * Math.log(2 * Math.PI) + logDet) - Math.log(n)
  const whiten = (src: ArrayLike<number>, off: number, out: Float64Array, o: number) => {
    for (let a = 0; a < d; a++) {
      let s = src[off + a]
      for (let k = 0; k < a; k++) s -= L[a * d + k] * out[o + k]
      out[o + a] = s / L[a * d + a]
    }
  }
  const wx = new Float64Array(n * d)
  for (let i = 0; i < n; i++) whiten(x, i * d, wx, i * d)
  return { cov, logNorm, wx, whiten }
}

/**
 * The bandwidth factor maximising the leave-one-out log-likelihood Σᵢ log f̂₋ᵢ(xᵢ) over 25 log-spaced factors in
 * [0.01, 2] (likelihood cross-validation; Duin, 1976; Silverman, 1986, §3.4.4). Unlike Scott's rule, which assumes one
 * Gaussian bump, it follows the scale of narrow, separated modes.
 */
function crossValidatedFactor(x: ArrayLike<number>, n: number, d: number): number {
  // Squared distances under the unscaled covariance (factor 1); a factor c divides them by c².
  const { wx } = whitened(x, n, d, 1)
  const d2 = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < i; j++) {
      let s = 0
      for (let a = 0; a < d; a++) s += (wx[i * d + a] - wx[j * d + a]) ** 2
      d2[i * n + j] = d2[j * n + i] = s
    }
  let best = 1
  let bestScore = -Infinity
  for (let k = 0; k < 25; k++) {
    const c = 0.01 * 200 ** (k / 24)
    let score = -n * d * Math.log(c)
    for (let i = 0; i < n; i++) {
      let top = -Infinity
      for (let j = 0; j < n; j++) if (j !== i) top = Math.max(top, -d2[i * n + j] / (2 * c * c))
      let acc = 0
      for (let j = 0; j < n; j++) if (j !== i) acc += Math.exp(-d2[i * n + j] / (2 * c * c) - top)
      score += top + Math.log(acc)
    }
    if (score > bestScore) {
      bestScore = score
      best = c
    }
  }
  return best
}

/** What `multivariateKde` returns. */
export type MultivariateKde = {
  /** f̂ at each query row (rank 1). */
  density: Tensor
  /** log f̂ at each query row (rank 1), computed stably (finite far from the sample). */
  logDensity: Tensor
  /** The bandwidth factor c: the kernel covariance is c²Σ̂. */
  factor: number
  /** The kernel covariance c²Σ̂ (d × d, row-major). */
  covariance: Tensor
}

/**
 * A Gaussian kernel density estimate in d dimensions, as `scipy.stats.gaussian_kde`: the kernel is N(0, c²Σ̂) with Σ̂
 * the sample covariance (n − 1 denominator) and c the bandwidth factor, Scott's n^{−1/(d+4)} (default), Silverman's
 * (n(d + 2)/4)^{−1/(d+4)}, `cross-validation` (the leave-one-out likelihood's best factor) or a given number; f̂(t) = (1/n) Σᵢ N(t; xᵢ, c²Σ̂). `sample` is [n, d] and `at` is [m, d].
 */
export function multivariateKde(
  sample: Tensor,
  at: Tensor,
  options: { bandwidth?: BandwidthRule | 'cross-validation' } = {},
): MultivariateKde {
  const [n, d] = sample.shape
  const [m, dq] = at.shape
  if (!(n >= 2) || d !== dq)
    throw new ShapeError('multivariateKde', 'multivariateKde: need an [n, d] sample (n ≥ 2) and [m, d] queries')
  const rule = options.bandwidth ?? 'scott'
  const x = allValues(sample)
  const q = allValues(at)
  const factor =
    typeof rule === 'number'
      ? rule
      : rule === 'scott'
        ? n ** (-1 / (d + 4))
        : rule === 'silverman'
          ? ((n * (d + 2)) / 4) ** (-1 / (d + 4))
          : crossValidatedFactor(x, n, d)
  if (!(factor > 0)) throw new DomainError('multivariateKde', 'multivariateKde: the bandwidth factor must be positive')
  const { cov, logNorm, wx, whiten } = whitened(x, n, d, factor)
  const wq = new Float64Array(d)
  const terms = new Float64Array(n)
  const logDensity = new Float64Array(m)
  for (let r = 0; r < m; r++) {
    whiten(q, r * d, wq, 0)
    let top = -Infinity
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let a = 0; a < d; a++) s += (wq[a] - wx[i * d + a]) ** 2
      terms[i] = -0.5 * s
      if (terms[i] > top) top = terms[i]
    }
    let acc = 0
    for (let i = 0; i < n; i++) acc += Math.exp(terms[i] - top)
    logDensity[r] = logNorm + top + Math.log(acc)
  }
  return {
    density: vectorOf(logDensity.map(Math.exp)),
    logDensity: vectorOf(logDensity),
    factor,
    covariance: fromData(cov, [d, d]),
  }
}
