/**
 * Transport on the line, where the optimal plan is monotone (it matches quantiles) for any cost $h(\abs{x - y})$ with
 * $h$ convex: Wasserstein distances from quantile functions, the monotone plan between histograms, sliced Wasserstein
 * distances in higher dimensions (Rabin, Peyré, Delon and Bernot, 2011, SSVM), and 1-D Wasserstein barycentres
 * (quantile averaging; Agueh and Carlier, 2011, SIAM J. Math. Anal. 43(2)).
 *
 * A distribution here is a sample of values with optional weights, normalised to sum to 1, and its quantile function
 * $F^{-1}(q) = \min\set{x : F(x) \ge q}$ is a step function. Everything is computed exactly from those steps, with no
 * grid, except the barycentre, which is returned on a grid of levels.
 */

import { normals, type Stream } from 'aifn-compute/foundation/random'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { readPoints, readVector, type PointsInput, type WeightsInput } from './discrete'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A weighted sample sorted by value, with its weights normalised to sum to 1. Throws `ShapeError` (named for
 * `wasserstein1d`) when the weights do not match the values in length.
 *
 * @param values The sample's values, in any order; not modified.
 * @param weights The weight of each value, non-negative and not all zero; equal weights when left out.
 * @returns The values `x` in increasing order and their weights `w`, normalised, in the same order.
 */
function sortedWithWeights(values: Float64Array, weights?: Float64Array): { x: number[]; w: number[] } {
  const n = values.length
  const w = weights ?? new Float64Array(n).fill(1)
  if (w.length !== n) throw new ShapeError('wasserstein1d', 'wasserstein1d: weights must match the values')
  const total = w.reduce((s, v) => s + v, 0)
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => values[i] - values[j])
  return { x: order.map((i) => values[i]), w: order.map((i) => w[i] / total) }
}

/**
 * The $p$-Wasserstein distance between two weighted samples on the line,
 * $W_p = \left(\int_0^1 \abs{F^{-1}(q) - G^{-1}(q)}^p \, dq\right)^{1/p}$ with $F$ and $G$ their CDFs, computed
 * exactly from the step quantile functions by merging their breakpoints (after sorting, linear in the sample sizes).
 * For $p = 1$ this equals `scipy.stats.wasserstein_distance` (the area between the CDFs). Weights are normalised, so
 * the samples may differ in size and total weight.
 *
 * @param u The values of the first sample.
 * @param v The values of the second sample.
 * @param options The order and the weights.
 * @param options.p The order $p$ of the distance (1 by default).
 * @param options.uWeights The weight of each value of `u` (same length, non-negative); equal weights when left out.
 * @param options.vWeights The weight of each value of `v` (same length, non-negative); equal weights when left out.
 * @returns $W_p$.
 *
 * @example Shifting a sample moves it by the shift
 * // Every quantile of the second sample is 3 more than the first's, so W_p = 3 for every p.
 * print('W1 =', wasserstein1d([0, 1, 2], [3, 4, 5]))
 * print('W2 =', wasserstein1d([0, 1, 2], [3, 4, 5], { p: 2 }))
 *
 * @example Weights move mass without moving points
 * // Three quarters of the mass at 0 against half: a quarter of the mass moves a distance of 1.
 * print('W1 =', wasserstein1d([0, 1], [0, 1], { uWeights: [3, 1] }))
 */
export function wasserstein1d(
  u: WeightsInput,
  v: WeightsInput,
  { p = 1, uWeights, vWeights }: { p?: Scalar; uWeights?: WeightsInput; vWeights?: WeightsInput } = {},
): Scalar {
  const U = sortedWithWeights(readVector(u, 'wasserstein1d u'), uWeights && readVector(uWeights, 'wasserstein1d'))
  const V = sortedWithWeights(readVector(v, 'wasserstein1d v'), vWeights && readVector(vWeights, 'wasserstein1d'))
  let i = 0
  let j = 0
  let ru = U.w[0]
  let rv = V.w[0]
  let total = 0
  // Walk both quantile functions together: each piece of mass pairs the current atoms of u and v.
  while (i < U.x.length && j < V.x.length) {
    const mass = Math.min(ru, rv)
    total += mass * Math.abs(U.x[i] - V.x[j]) ** p
    ru -= mass
    rv -= mass
    if (ru <= 1e-15) {
      i++
      ru = U.w[i] ?? 0
    }
    if (rv <= 1e-15) {
      j++
      rv = V.w[j] ?? 0
    }
  }
  return total ** (1 / p)
}

/** The monotone (north-west corner) plan between two histograms on sorted grids, as its non-zero entries. */
export interface MonotonePlan {
  /** The source bin of each entry (int32). */
  i: Tensor
  /** The target bin of each entry (int32). */
  j: Tensor
  /** The mass moved from source bin `i` to target bin `j` by each entry. */
  mass: Tensor
}

/**
 * The exact optimal plan between two histograms $\avec$ and $\bvec$ with equal totals, each over bins in increasing
 * order of position on the line, for any cost $h(\abs{x - y})$ with $h$ convex: the monotone coupling that matches
 * quantiles, built by the north-west corner rule. Only the bin indices are used, not their positions. Entries with mass
 * at most $10^{-12} \max(1, \sum_i a_i)$ are dropped; the totals are not checked.
 *
 * @param a The source histogram: the mass in each bin, bins in increasing order of position.
 * @param b The target histogram: the mass in each bin, bins in increasing order of position.
 * @returns The non-zero entries of the plan, in order: source bins, target bins and masses.
 *
 * @example Mass moves one bin to the right
 * const { i, j, mass } = monotonePlan([0.5, 0.5, 0], [0, 0.5, 0.5])
 * print('from', i, 'to', j, 'mass', mass)
 *
 * @example The north-west corner rule splits a bin
 * const { i, j, mass } = monotonePlan([0.5, 0.25, 0.25], [0.25, 0.25, 0.5])
 * print('from', i, 'to', j, 'mass', mass)
 */
export function monotonePlan(a: WeightsInput, b: WeightsInput): MonotonePlan {
  const ra = Array.from(readVector(a, 'monotonePlan a'))
  const rb = Array.from(readVector(b, 'monotonePlan b'))
  const is: number[] = []
  const js: number[] = []
  const ms: number[] = []
  let i = 0
  let j = 0
  const tiny =
    1e-12 *
    Math.max(
      1,
      ra.reduce((s, v) => s + v, 0),
    )
  while (i < ra.length && j < rb.length) {
    const mass = Math.min(ra[i], rb[j])
    if (mass > tiny) {
      is.push(i)
      js.push(j)
      ms.push(mass)
    }
    ra[i] -= mass
    rb[j] -= mass
    if (ra[i] <= tiny) i++
    else j++
  }
  return { i: fromData(Int32Array.from(is)), j: fromData(Int32Array.from(js)), mass: fromData(Float64Array.from(ms)) }
}

/** The result of `slicedWasserstein`. */
export interface SlicedWasserstein {
  /**
   * $SW_p = \left(\frac{1}{k} \sum_{l=1}^k W_p^p(\thetavec_l)\right)^{1/p}$, with $W_p(\thetavec_l)$ the distance
   * between the projections on the direction $\thetavec_l$.
   */
  distance: Scalar
  /** The unit directions $\thetavec_l$, $k \times d$, one per row. */
  directions: Tensor
  /** $W_p$ of the projections on each direction ($k$ values). */
  perDirection: Tensor
}

/**
 * The sliced $p$-Wasserstein distance between two point clouds in $d$ dimensions, as POT's
 * `ot.sliced_wasserstein_distance`: the average of the 1-D $W_p^p$ between their projections on $k$ random unit
 * directions (drawn uniformly on the sphere from `s`), to the power $1/p$. It is a Monte Carlo estimate of the
 * average over all directions. Throws `ShapeError` when the clouds differ in dimension.
 *
 * @param s The random stream the directions are drawn from: the same stream gives the same directions.
 * @param x The first point cloud, $n \times d$ (a 1-D array is $n$ points on a line).
 * @param y The second point cloud, $m \times d$.
 * @param options The number of directions, the order and the weights.
 * @param options.projections The number $k$ of random directions (default 50).
 * @param options.p The order $p$ (default 2).
 * @param options.xWeights The weight of each point of `x`; equal weights when left out.
 * @param options.yWeights The weight of each point of `y`; equal weights when left out.
 * @returns The distance, with the directions and the 1-D distance along each.
 *
 * @example A translated cloud
 * // Shifting by t gives W2 = |t . theta| along theta, so SW2 approaches |t| / sqrt(d) = 5 / sqrt(2) here.
 * const x = [[0, 0], [1, 0]]
 * const y = [[3, 4], [4, 4]]
 * const { distance } = slicedWasserstein(stream(0), x, y, { projections: 200 })
 * print('SW2 =', distance)
 * print('5 / sqrt(2) =', 5 / Math.sqrt(2))
 *
 * @example On the line every slice is the 1-D distance
 * const r = slicedWasserstein(stream(0), [0, 1, 2], [3, 4, 5], { projections: 3, p: 1 })
 * print('directions =', r.directions)
 * print('per direction =', r.perDirection)
 * print('SW1 =', r.distance, ' W1 =', wasserstein1d([0, 1, 2], [3, 4, 5]))
 */
export function slicedWasserstein(
  s: Stream,
  x: PointsInput,
  y: PointsInput,
  options: { projections?: Size; p?: Scalar; xWeights?: WeightsInput; yWeights?: WeightsInput } = {},
): SlicedWasserstein {
  const { projections = 50, p = 2 } = options
  const X = readPoints(x, 'slicedWasserstein')
  const Y = readPoints(y, 'slicedWasserstein')
  if (X.d !== Y.d) throw new ShapeError('slicedWasserstein', 'slicedWasserstein: point sets differ in dimension')
  const d = X.d
  // One block of standard normals, normalised row by row: uniform directions on the sphere.
  const dirs = Float64Array.from(toFlat(normals(s, [projections, d])))
  const per = new Float64Array(projections)
  let sum = 0
  for (let k = 0; k < projections; k++) {
    let norm = 0
    for (let c = 0; c < d; c++) norm += dirs[k * d + c] ** 2
    norm = Math.sqrt(norm)
    for (let c = 0; c < d; c++) dirs[k * d + c] /= norm
    const project = (P: { v: Float64Array; n: number }) =>
      Float64Array.from({ length: P.n }, (_, i) => {
        let t = 0
        for (let c = 0; c < d; c++) t += P.v[i * d + c] * dirs[k * d + c]
        return t
      })
    const w = wasserstein1d(project(X), project(Y), { p, uWeights: options.xWeights, vWeights: options.yWeights })
    per[k] = w
    sum += w ** p
  }
  return {
    distance: (sum / projections) ** (1 / p),
    directions: fromData(dirs, [projections, d]),
    perDirection: fromData(per),
  }
}

/**
 * The quantile function of a weighted sample at level $q \in (0, 1)$: the left-continuous inverse CDF, the first value
 * whose cumulative weight reaches $q$ (to within $10^{-15}$), or the last value when none does.
 *
 * @param xs The sample's values in increasing order.
 * @param ws Their weights, in the same order, summing to 1.
 * @param q The level $q$.
 * @returns $F^{-1}(q)$.
 */
function quantileAt(xs: number[], ws: number[], q: number): number {
  let acc = 0
  for (let i = 0; i < xs.length; i++) {
    acc += ws[i]
    if (q <= acc + 1e-15) return xs[i]
  }
  return xs[xs.length - 1]
}

/** A 1-D Wasserstein barycentre, as its quantile function on a grid of levels. */
export interface Barycenter1d {
  /** The levels $q_k = (k + 1/2)/K$, $k = 0, \dots, K - 1$. */
  levels: Tensor
  /**
   * The barycentre's quantiles $\sum_s \lambda_s F_s^{-1}(q_k)$; as a sample of $K$ equally weighted points, the
   * barycentre itself.
   */
  quantiles: Tensor
}

/**
 * The $W_2$ barycentre of distributions on the line with weights $\lambda_s$ (default equal), the distribution that
 * minimises $\sum_s \lambda_s W_2^2(\cdot, \mu_s)$: its quantile function is the $\lambda$-weighted average of theirs.
 * Each distribution is a sample (optionally weighted), and the result is the quantile function on $K$ levels.
 *
 * @param samples The distributions $\mu_s$, each a sample of values in any order.
 * @param options The weights and the number of levels.
 * @param options.weights The weight $\lambda_s$ of each distribution, one per sample, normalised to sum to 1; equal
 *   weights when left out.
 * @param options.sampleWeights For each sample, the weights of its values, or `undefined` for equal weights; equal
 *   weights for every sample when left out.
 * @param options.levels The number $K$ of quantile levels (default 100).
 * @returns The levels and the barycentre's quantiles at them.
 *
 * @example The barycentre of two shifted samples sits halfway
 * const r = barycenter1d([[0, 1, 2], [10, 11, 12]], { levels: 3 })
 * print('levels =', r.levels)
 * print('quantiles =', r.quantiles)
 *
 * @example Weights pull it towards one sample
 * // Weights 3 : 1, so each quantile is a quarter of the way from the first sample to the second.
 * print('quantiles =', barycenter1d([[0, 1, 2], [10, 11, 12]], { levels: 3, weights: [3, 1] }).quantiles)
 */
export function barycenter1d(
  samples: readonly WeightsInput[],
  options: { weights?: WeightsInput; sampleWeights?: readonly (WeightsInput | undefined)[]; levels?: Size } = {},
): Barycenter1d {
  const k = options.levels ?? 100
  const lambda = options.weights
    ? Array.from(readVector(options.weights, 'barycenter1d weights'))
    : samples.map(() => 1 / samples.length)
  const total = lambda.reduce((s, v) => s + v, 0)
  const sorted = samples.map((x, i) => {
    const w = options.sampleWeights?.[i]
    return sortedWithWeights(readVector(x, 'barycenter1d'), w && readVector(w, 'barycenter1d'))
  })
  const levels = Float64Array.from({ length: k }, (_, i) => (i + 0.5) / k)
  const q = levels.map((l) => sorted.reduce((s, d, i) => s + (lambda[i] / total) * quantileAt(d.x, d.w, l), 0))
  return { levels: fromData(levels), quantiles: fromData(q) }
}
