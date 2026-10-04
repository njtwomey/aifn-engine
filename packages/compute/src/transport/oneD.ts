/**
 * Transport on the line, where the optimal plan is monotone (it matches quantiles) for any convex cost of |x − y|:
 * Wasserstein distances from quantile functions, the monotone plan between histograms, sliced Wasserstein distances in
 * higher dimensions (Rabin, Peyré, Delon and Bernot, 2011, SSVM), and 1-D Wasserstein barycentres (quantile averaging;
 * Agueh and Carlier, 2011, SIAM J. Math. Anal. 43(2)).
 */

import { normals, type Stream } from 'aifn-compute/foundation/random'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { readPoints, readVector, type PointsInput, type WeightsInput } from './discrete'
import { ShapeError } from 'aifn-compute/foundation/errors'

function sortedWithWeights(values: Float64Array, weights?: Float64Array): { x: number[]; w: number[] } {
  const n = values.length
  const w = weights ?? new Float64Array(n).fill(1)
  if (w.length !== n) throw new ShapeError('wasserstein1d', 'wasserstein1d: weights must match the values')
  const total = w.reduce((s, v) => s + v, 0)
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => values[i] - values[j])
  return { x: order.map((i) => values[i]), w: order.map((i) => w[i] / total) }
}

/**
 * The p-Wasserstein distance between two weighted samples on the line, W_p = (∫₀¹ |F⁻¹(q) − G⁻¹(q)|^p dq)^{1/p},
 * computed exactly from the step quantile functions by merging their breakpoints. For p = 1 this equals
 * `scipy.stats.wasserstein_distance` (the area between the CDFs).
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
  /** Source bin, target bin (int32) and mass of each entry. */
  i: Tensor
  j: Tensor
  mass: Tensor
}

/**
 * The exact optimal plan between two histograms a and b (equal totals) on the same sorted 1-D grid, for any convex cost
 * of |x − y|: the monotone coupling that matches quantiles.
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
  /** SW_p = (mean over directions of W_p^p of the projections)^{1/p}. */
  distance: Scalar
  /** The unit directions, k × d. */
  directions: Tensor
  /** W_p of the projections on each direction. */
  perDirection: Tensor
}

/**
 * The sliced p-Wasserstein distance between two point clouds in d dimensions: the average of 1-D W_p^p between their
 * projections on `projections` random unit directions (drawn uniformly on the sphere from `s`), to the power 1/p.
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

/** The quantile function of a weighted sample at level q ∈ (0, 1) (the left-continuous inverse CDF). */
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
  /** Levels q_k = (k + 1/2)/K. */
  levels: Tensor
  /** The barycentre's quantiles Σ λ_s F_s⁻¹(q_k); as a sample of K equally weighted points, the barycentre itself. */
  quantiles: Tensor
}

/**
 * The W₂ barycentre of distributions on the line with weights λ (default equal): its quantile function is the
 * λ-weighted average of theirs. Each distribution is a sample (optionally weighted).
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
