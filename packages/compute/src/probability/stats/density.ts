/**
 * Estimates of a sample's distribution: histograms with an explicit bin rule, the empirical CDF, and Gaussian kernel
 * density estimates in one dimension and in $d$, with the bandwidth rules of Scott and Silverman (and likelihood
 * cross-validation in $d$ dimensions).
 *
 * Each matches its numpy or scipy counterpart (`numpy.histogram`, `scipy.stats.ecdf`, `scipy.stats.gaussian_kde`).
 * Values a histogram cannot place are reported in `dropped`, never clipped into the end bins, and a KDE of constant
 * data is flagged rather than hidden.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { allValues, toSequence, vectorOf, type Data } from './input'
import { requireNonEmpty, requireSameLength, weightedVariance, variance } from './descriptive'
import { interquartileRange, sortedValues } from './quantile'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * How `histogram` chooses its bins. Every rule gives equal-width bins over $[\text{first}, \text{last}]$ (the data's
 * min and max, or `range`; widened by $\tfrac12$ each way when they are equal), except explicit edges.
 * - a number: that many bins.
 * - `{ width }`: bins of that width starting at first; the last edge is the first one at or past last.
 * - `sturges`: width $= s / (\log_2 n + 1)$ (Sturges 1926), then $\lceil (\text{last} - \text{first}) /
 *   \text{width} \rceil$ bins.
 * - `freedman-diaconis`: width $= 2 \cdot \text{IQR} \cdot n^{-1/3}$ (Freedman and Diaconis 1981), then bins as for
 *   `sturges`; one bin when the IQR is 0.
 * - an array or rank-1 tensor: explicit, strictly increasing edges.
 *
 * Here $n$, the spread $s$ (max minus min) and the IQR are those of the data inside the range. The number, `sturges`
 * and `freedman-diaconis` rules match `numpy.histogram_bin_edges` (`bins=k`, `'sturges'`, `'fd'`).
 */
export type BinRule = number | { width: number } | 'sturges' | 'freedman-diaconis' | Data

/** A histogram: `edges` (rank 1) has one more entry than `counts` and `density`. */
export type Histogram = {
  /** The bin edges, ascending (rank 1, $k + 1$ entries for $k$ bins). */
  edges: Tensor
  /**
   * Values (or summed weights) per bin. Bin $i$ is $[e_i, e_{i+1})$, except the last, which includes its right edge.
   */
  counts: Tensor
  /** counts / (total counted $\times$ bin width): integrates to 1 over the bins (NaN when nothing was counted). */
  density: Tensor
  /** Values (or weight) outside the edges, or NaN, which were not counted. */
  dropped: number
}

/**
 * numpy's `linspace(first, last, count + 1)`: $i \cdot \text{step} + \text{first}$, with the last edge exactly
 * `last`.
 *
 * @param first The first edge.
 * @param last The last edge.
 * @param count The number of bins (one fewer than the edges).
 * @returns The $\text{count} + 1$ edges.
 */
function evenEdges(first: number, last: number, count: number): Float64Array {
  const step = (last - first) / count
  const edges = Float64Array.from({ length: count + 1 }, (_, i) => i * step + first)
  edges[count] = last
  return edges
}

/**
 * Bins the values $\xvec$ and returns `{ edges, counts, density, dropped }`. Values equal to the last edge fall in the
 * last bin (the aifn convention, as in numpy); values outside `range` or the explicit edges, and NaN, are dropped and
 * reported in `dropped`, never clipped into the end bins. `weights` replace unit counts. The default rule is 10 bins,
 * as in numpy. Matches `numpy.histogram`. Throws `DomainError` for a bad rule, range or edges, and (as numpy) for NaN
 * in the data when the range comes from the data; `ShapeError` when the weights do not match the values.
 *
 * @param xData The values: an array, or a tensor of any rank (every element).
 * @param options The bins, the range and the weights.
 * @param options.bins The bin rule (see `BinRule`; default 10 bins).
 * @param options.range The interval $[\text{first}, \text{last}]$ the equal-width bins cover (default: the data's min
 *   and max, or $[0, 1]$ for no data). Ignored with explicit edges.
 * @param options.weights One weight per value (an array or rank-1 tensor), counted in place of 1.
 * @returns The `edges`, the `counts` and `density` per bin, and the count (or weight) `dropped`.
 *
 * @example Three bins, as np.histogram(x, bins=3)
 * const h = histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], { bins: 3 })
 * print('edges =', h.edges)
 * print('counts =', h.counts)
 * print('density =', h.density)
 *
 * @example A range drops what lies outside it
 * const h = histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], { bins: 2, range: [1, 3] })
 * print('edges =', h.edges)
 * print('counts =', h.counts)
 * print('dropped =', h.dropped)
 *
 * @example Explicit edges, and weights
 * print('counts =', histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], { bins: [0, 2, 3, 10] }).counts)
 * print('weighted =', histogram([1, 2, 3, 4], { bins: 2, weights: [1, 1, 1, 3] }).counts)
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
 * The empirical CDF as a step function: the distinct sorted values and $\hat{F}(v) = \#\{i : x_i \le v\} / n$ at
 * each. Matches `scipy.stats.ecdf(x).cdf` (`quantiles`, `probabilities`). Throws `DomainError` on empty data.
 *
 * @param xData The sample: an array, or a tensor of any rank (every element).
 * @returns The distinct `values` in ascending order and the `probabilities` $\hat{F}$ at each (rank-1 tensors).
 *
 * @example Steps at the distinct values
 * const { values, probabilities } = ecdf([3, 1, 2, 2])
 * print('values =', values)
 * print('F =', probabilities)
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

/**
 * The empirical CDF of $\xvec$ evaluated at each point $t$: $\#\{i : x_i \le t\} / n$. Throws `DomainError` on an
 * empty sample.
 *
 * @param xData The sample: an array or a rank-1 tensor.
 * @param tData The points at which to evaluate it: an array or a rank-1 tensor, in any order.
 * @returns $\hat{F}(t)$ at each point, in the order of the points.
 *
 * @example Between, at and beyond the sample
 * print('F =', ecdfAt([1, 2, 3, 4], [0, 2, 2.5, 4, 9]))
 */
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
 * A bandwidth rule for the Gaussian KDE, in scipy's form $h = \text{factor} \times \hat{\sigma}$, with
 * $\hat{\sigma}$ the sample standard deviation ($n - 1$ divisor) and $n$ the (effective) sample size:
 * - `scott`: factor $n^{-1/5}$ (Scott 1992).
 * - `silverman`: factor $(3n/4)^{-1/5}$ (Silverman 1986, eq. 3.28 in scipy's form).
 * - a number: the bandwidth $h$ itself (the kernel's standard deviation) for `kde`; for `multivariateKde`, the factor.
 */
export type BandwidthRule = 'scott' | 'silverman' | number

/**
 * The Gaussian KDE bandwidth $h$ (the kernel's standard deviation) for a sample, as `scipy.stats.gaussian_kde`
 * computes `sqrt(covariance)`. With weights, $n$ is Kish's effective size $(\sum_i w_i)^2 / \sum_i w_i^2$ and
 * $\hat{\sigma}^2$ the reliability-weighted variance, as in scipy. Throws `DomainError` on an empty sample or a
 * bandwidth that is not positive. Constant data give $h = 0$.
 *
 * @param xData The sample: an array or a rank-1 tensor.
 * @param rule The bandwidth rule (see `BandwidthRule`); a number is returned as it is.
 * @param weightsData One non-negative weight per value (an array or rank-1 tensor); left out, the values count
 *   equally.
 * @returns The bandwidth $h$.
 *
 * @example Scott and Silverman, as scipy's gaussian_kde
 * const x = [1, 2, 3, 4, 5]
 * print('Scott h =', kdeBandwidth(x))
 * print('Silverman h =', kdeBandwidth(x, 'silverman'))
 * // Weight on one value lowers the effective size, here to 3.2.
 * print('weighted Scott h =', kdeBandwidth(x, 'scott', [1, 1, 1, 1, 4]))
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

/**
 * Kish's effective sample size $(\sum_i w_i)^2 / \sum_i w_i^2$ of weights.
 *
 * @param w The weights.
 * @returns The effective size (NaN when every weight is 0).
 */
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
 * A Gaussian kernel density estimate of the sample $\xvec$, evaluated at the points `at`:
 * $\hat{f}(t) = \sum_i w_i \varphi((t - x_i)/h) / h$ with weights normalised to sum to 1 (uniform by default) and $h$
 * from `bandwidth` (default `scott`). Returns the densities (a rank-1 tensor) and the bandwidth used. Matches
 * `scipy.stats.gaussian_kde(x, weights=w)` evaluated at `at`. `degenerate` is true when $h$ is 0 (constant data); the
 * densities are then NaN.
 *
 * @param xData The sample $x_i$: an array or a rank-1 tensor.
 * @param atData The points $t$ at which to evaluate the density: an array or a rank-1 tensor.
 * @param options The bandwidth and the weights.
 * @param options.bandwidth The bandwidth rule (see `BandwidthRule`; default `scott`), or $h$ itself.
 * @param options.weights One non-negative weight per sample value (an array or rank-1 tensor), normalised to sum to 1.
 * @returns The `density` at each point, the `bandwidth` $h$ used, and whether it is `degenerate`.
 *
 * @example As scipy's gaussian_kde([0, 1, 2]) at four points
 * const { density, bandwidth } = kde([0, 1, 2], [0, 1, 2, 5])
 * print('density =', density)
 * print('h =', bandwidth)
 *
 * @example A fixed bandwidth, and constant data
 * print('h = 0.5:', kde([0, 1, 2], [0, 1], { bandwidth: 0.5 }).density)
 * print('degenerate =', kde([3, 3, 3], [3]).degenerate)
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

/**
 * The sample covariance ($n - 1$ divisor) of row-major points ($n \times d$).
 *
 * @param x The points: row $i$ occupies entries `i * d` to `i * d + d - 1`.
 * @param n The number of points.
 * @param d The dimension.
 * @returns The $d \times d$ covariance, row-major.
 */
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
 * The kernel $\Gauss(\zeros, c^2\hat{\Sigmamat})$ of a sample: its covariance, the log of
 * $1/(n (2\pi)^{d/2} \lvert c^2\hat{\Sigmamat} \rvert^{1/2})$, the whitened sample $\Lmat^{-1}\xvec_i$
 * ($\Lmat\Lmat^\top = c^2\hat{\Sigmamat}$) and the whitening map, so a quadratic form is a squared distance between
 * whitened points. Throws `NumericalError` 'singular' when the covariance is not positive definite (the points lie on a
 * subspace).
 *
 * @param x The sample points, row-major: row $i$ occupies entries `i * d` to `i * d + d - 1`.
 * @param n The number of points (at least 2).
 * @param d The dimension.
 * @param factor The bandwidth factor $c$.
 * @returns `cov` ($c^2\hat{\Sigmamat}$, row-major), `logNorm` (the log normaliser above), `wx` (the whitened points,
 *   row-major) and `whiten(src, off, out, o)`, which whitens the $d$ values of `src` from `off` into `out` from `o`.
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
 * The bandwidth factor maximising the leave-one-out log-likelihood $\sum_i \log \hat{f}_{-i}(\xvec_i)$ over 25
 * log-spaced factors in $[0.01, 2]$ (likelihood cross-validation; Duin, 1976; Silverman, 1986, §3.4.4). Unlike Scott's
 * rule, which assumes one Gaussian bump, it follows the scale of narrow, separated modes. It costs $O(n^2)$ memory and
 * $O(25 n^2)$ time.
 *
 * @param x The sample points, row-major: row $i$ occupies entries `i * d` to `i * d + d - 1`.
 * @param n The number of points (at least 2).
 * @param d The dimension.
 * @returns The best factor $c$ of the 25.
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
  /** $\hat{f}$ at each query row (rank 1). */
  density: Tensor
  /** $\log \hat{f}$ at each query row (rank 1), computed stably (finite far from the sample). */
  logDensity: Tensor
  /** The bandwidth factor $c$: the kernel covariance is $c^2\hat{\Sigmamat}$. */
  factor: number
  /** The kernel covariance $c^2\hat{\Sigmamat}$ ($d \times d$). */
  covariance: Tensor
}

/**
 * A Gaussian kernel density estimate in $d$ dimensions, as `scipy.stats.gaussian_kde`: the kernel is
 * $\Gauss(\zeros, c^2\hat{\Sigmamat})$ with $\hat{\Sigmamat}$ the sample covariance ($n - 1$ denominator) and $c$
 * the bandwidth factor, Scott's $n^{-1/(d+4)}$ (default), Silverman's $(n(d + 2)/4)^{-1/(d+4)}$, `cross-validation`
 * (the leave-one-out likelihood's best factor) or a given number;
 * $\hat{f}(\tvec) = \frac{1}{n} \sum_i \Gauss(\tvec; \xvec_i, c^2\hat{\Sigmamat})$. `sample` is $[n, d]$ and
 * `at` is $[m, d]$. Throws `ShapeError` for fewer than two points or mismatched dimensions, `DomainError` for a factor
 * that is not positive, and `NumericalError` 'singular' when the points lie on a subspace.
 *
 * @param sample The sample, an $n \times d$ tensor with one point per row ($n \ge 2$).
 * @param at The query points, an $m \times d$ tensor with one point per row.
 * @param options The bandwidth.
 * @param options.bandwidth `scott` (default), `silverman`, `cross-validation`, or the factor $c$ itself (not $h$, as in
 *   `kde`).
 * @returns The `density` and `logDensity` at each query row, the `factor` $c$ and the kernel `covariance`.
 *
 * @example As scipy's gaussian_kde on five points in the plane
 * const sample = tensor([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
 * const fit = multivariateKde(sample, tensor([[0.5, 0], [0.5, 3]]))
 * print('density =', fit.density)
 * print('log density =', fit.logDensity)
 * print('factor =', fit.factor)
 * print('kernel covariance =', fit.covariance)
 *
 * @example Cross-validation picks a narrow kernel for three tight clusters
 * const sample = tensor([
 *   [0, 0], [0.2, 0.1], [0.1, 0.3], [-0.1, 0.1],
 *   [4, 0], [4.2, 0.2], [3.9, 0.1], [4.1, -0.2],
 *   [0, 4], [0.1, 4.2], [-0.2, 3.9], [0.2, 4.1],
 * ])
 * // A point in a cluster, and one in the gap between two.
 * const at = tensor([[0, 0], [2, 0]])
 * const scott = multivariateKde(sample, at)
 * const cv = multivariateKde(sample, at, { bandwidth: 'cross-validation' })
 * print('Scott: factor', scott.factor, ' density', scott.density)
 * print('cross-validated: factor', cv.factor, ' density', cv.density)
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
