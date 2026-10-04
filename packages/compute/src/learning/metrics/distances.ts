/**
 * Distances and similarities: Minkowski distances (Manhattan, Euclidean, Chebyshev), cosine similarity and distance,
 * angular and Mahalanobis distances, pairwise distance matrices, orthogonal Procrustes alignment and the Procrustes
 * disparity, and the Hausdorff family of distances between point sets (Hausdorff, HD95, average symmetric surface
 * distance).
 */

import { det, solve, squaredDistances, svd } from 'aifn-compute/numerics/linalg'
import { quantile } from 'aifn-compute/probability/stats'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { defineMetric, dense, divide, matrix, sameLength, values, vector, type Data, type Rows } from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

function pairOf(x: Data, y: Data, what: string) {
  const a = values(x)
  const b = values(y)
  sameLength(a, b, what)
  return { a, b }
}

const distanceInfo = (key: string, name: string, range: readonly [number, number] = [0, Infinity]) =>
  ({
    key,
    stability: 'stable',
    name,
    inputs: 'vectors',
    direction: 'lower',
    range,
    notes: ['pairwise-distances-and-cosine-similarity'],
  }) as const

/** The Minkowski distance (Σ|xₖ − yₖ|^p)^{1/p} for p ≥ 1 (default 2); p = ∞ gives the Chebyshev distance. */
export const minkowskiDistance = defineMetric(
  distanceInfo('minkowskiDistance', 'Minkowski distance'),
  (x: Data, y: Data, options: { p?: number } = {}): number => {
    const { a, b } = pairOf(x, y, 'minkowskiDistance')
    const p = options.p ?? 2
    if (p === Infinity) return a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0)
    return a.reduce((s, v, i) => s + Math.abs(v - b[i]) ** p, 0) ** (1 / p)
  },
)

/** The Euclidean distance ‖x − y‖₂. */
export const euclideanDistance = defineMetric(
  distanceInfo('euclideanDistance', 'Euclidean distance'),
  (x: Data, y: Data) => minkowskiDistance(x, y, { p: 2 }),
)

/** The Manhattan distance Σ|xₖ − yₖ|. */
export const manhattanDistance = defineMetric(
  distanceInfo('manhattanDistance', 'Manhattan distance'),
  (x: Data, y: Data) => minkowskiDistance(x, y, { p: 1 }),
)

/** The Chebyshev distance maxₖ |xₖ − yₖ|. */
export const chebyshevDistance = defineMetric(
  distanceInfo('chebyshevDistance', 'Chebyshev distance'),
  (x: Data, y: Data) => minkowskiDistance(x, y, { p: Infinity }),
)

/** Cosine similarity xᵀy/(‖x‖‖y‖); NaN when either vector is zero. */
export const cosineSimilarity = defineMetric(
  { ...distanceInfo('cosineSimilarity', 'Cosine similarity', [-1, 1]), direction: 'higher' },
  (x: Data, y: Data): number => {
    const { a, b } = pairOf(x, y, 'cosineSimilarity')
    let dot = 0
    let na = 0
    let nb = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      na += a[i] * a[i]
      nb += b[i] * b[i]
    }
    return divide(dot, Math.sqrt(na * nb))
  },
)

/** Cosine distance 1 − cos θ (not a metric: it can break the triangle inequality). */
export const cosineDistance = defineMetric(
  distanceInfo('cosineDistance', 'Cosine distance', [0, 2]),
  (x: Data, y: Data) => 1 - cosineSimilarity(x, y),
)

/** Angular distance arccos(cos θ)/π, a true metric on directions, in [0, 1]. */
export const angularDistance = defineMetric(
  distanceInfo('angularDistance', 'Angular distance', [0, 1]),
  (x: Data, y: Data) => Math.acos(Math.max(-1, Math.min(1, cosineSimilarity(x, y)))) / Math.PI,
)

/** The Mahalanobis distance √((x − y)ᵀΣ⁻¹(x − y)) for a covariance matrix Σ (by a linear solve, not an inverse). */
export const mahalanobisDistance = defineMetric(
  distanceInfo('mahalanobisDistance', 'Mahalanobis distance'),
  (x: Data, y: Data, options: { covariance: Rows }): number => {
    const { a, b } = pairOf(x, y, 'mahalanobisDistance')
    const d = Float64Array.from(a, (v, i) => v - b[i])
    const S = dense(options.covariance, 'mahalanobis covariance')
    const z = solve(matrix(S.data, S.rows, S.cols), vector(d)).data
    return Math.sqrt(d.reduce((s, v, i) => s + v * z[i], 0))
  },
)

// ── Procrustes ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** The result of `orthogonalProcrustes`. */
export type Procrustes = {
  /** d × d orthogonal R with R xᵢ ≈ yᵢ (so X Rᵀ ≈ Y for points as rows). */
  rotation: Tensor
  /** The scale s (1 unless `scale: true`). */
  scale: number
  /** The translation t (zero unless `centre: true`): yᵢ ≈ s R xᵢ + t. */
  translation: Tensor
  /** The residual ‖s X Rᵀ + t − Y‖²_F. */
  residual: number
  /** True when R is a reflection (det R = −1). */
  reflection: boolean
}

/**
 * Orthogonal Procrustes (Schönemann 1966; procrustes-analysis): the orthogonal R minimising ‖RA − B‖_F for matched
 * points as the rows of X and Y (A = Xᵀ, B = Yᵀ), R = VUᵀ from the SVD AB⊤ = UΣVᵀ. With `reflection: false` a
 * reflection is turned into the best proper rotation (Kabsch 1976). Optionally also fits a translation (`centre`) and
 * a scale s = Σσₖ/‖A‖²_F (`scale`).
 */
export function orthogonalProcrustes(
  x: Rows,
  y: Rows,
  options: { reflection?: boolean; centre?: boolean; scale?: boolean } = {},
): Procrustes {
  const X = dense(x, 'orthogonalProcrustes')
  const Y = dense(y, 'orthogonalProcrustes')
  if (X.rows !== Y.rows || X.cols !== Y.cols)
    throw new ShapeError('metrics', 'metrics: orthogonalProcrustes: point sets differ in shape')
  const { rows: n, cols: d } = X
  const mx = new Float64Array(d)
  const my = new Float64Array(d)
  if (options.centre)
    for (let i = 0; i < n; i++)
      for (let c = 0; c < d; c++) {
        mx[c] += X.data[i * d + c] / n
        my[c] += Y.data[i * d + c] / n
      }
  const A = Float64Array.from(X.data, (v, k) => v - mx[k % d])
  const B = Float64Array.from(Y.data, (v, k) => v - my[k % d])
  // M = Xᵀ Y (d × d) = A Bᵀ.
  const M = new Float64Array(d * d)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) M[a * d + b] += A[i * d + a] * B[i * d + b]
  const { U, S, V } = svd(matrix(M, d, d))
  const u = U.data
  const v = V.data
  const s = Float64Array.from(S.data)
  const flip = new Float64Array(d).fill(1)
  const rotationOf = () => {
    const R = new Float64Array(d * d)
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) R[i * d + j] += v[i * d + k] * flip[k] * u[j * d + k]
    return R
  }
  let R = rotationOf()
  if (options.reflection === false && det(matrix(R, d, d)) < 0) {
    flip[d - 1] = -1 // the direction of the smallest singular value
    R = rotationOf()
  }
  let normA = 0
  for (const w of A) normA += w * w
  const traceRM = s.reduce((acc, sv, k) => acc + flip[k] * sv, 0)
  const scale = options.scale ? traceRM / normA : 1
  const t = new Float64Array(d)
  for (let c = 0; c < d; c++) {
    let rx = 0
    for (let k = 0; k < d; k++) rx += R[c * d + k] * mx[k]
    t[c] = my[c] - scale * rx
  }
  let residual = 0
  for (let i = 0; i < n; i++)
    for (let c = 0; c < d; c++) {
      let rx = 0
      for (let k = 0; k < d; k++) rx += R[c * d + k] * X.data[i * d + k]
      residual += (scale * rx + t[c] - Y.data[i * d + c]) ** 2
    }
  return {
    rotation: fromData(R, [d, d]),
    scale,
    translation: vector(t),
    residual,
    reflection: det(matrix(R, d, d)) < 0,
  }
}

/**
 * The Procrustes disparity of two matched point sets (rows as points): after centring both and scaling each to unit
 * Frobenius norm, the least squared error over rotations (reflections allowed) and scale, 1 − (Σσₖ)², in [0, 1]
 * (Gower 1975; as SciPy's `procrustes`).
 */
export const procrustesDisparity = defineMetric(
  {
    key: 'procrustesDisparity',
    stability: 'stable',
    name: 'Procrustes disparity',
    inputs: 'points',
    direction: 'lower',
    range: [0, 1],
    notes: ['procrustes-analysis'],
  },
  (x: Rows, y: Rows): number => {
    const normalised = (m: Rows) => {
      const D = dense(m, 'procrustesDisparity')
      const mean = new Float64Array(D.cols)
      for (let i = 0; i < D.rows; i++) for (let c = 0; c < D.cols; c++) mean[c] += D.data[i * D.cols + c] / D.rows
      const out = Float64Array.from(D.data, (v, k) => v - mean[k % D.cols])
      const norm = Math.sqrt(out.reduce((s, v) => s + v * v, 0))
      if (norm === 0) throw new DomainError('metrics', 'metrics: procrustesDisparity: a point set has no spread')
      return matrix(
        out.map((v) => v / norm),
        D.rows,
        D.cols,
      )
    }
    const X = normalised(x)
    const Y = normalised(y)
    const d = X.shape[1]
    const n = X.shape[0]
    const M = new Float64Array(d * d)
    for (let i = 0; i < n; i++)
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) M[a * d + b] += X.data[i * d + a] * Y.data[i * d + b]
    const sum = (svd(matrix(M, d, d)).S.data as Float64Array).reduce((s, v) => s + v, 0)
    return 1 - sum * sum
  },
)

// ── Hausdorff ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Each point of X's Euclidean distance to its nearest point of Y. */
function nearestDistances(X: ReturnType<typeof dense>, Y: ReturnType<typeof dense>): Float64Array {
  if (X.cols !== Y.cols) throw new ShapeError('metrics', 'metrics: point sets differ in dimension')
  if (Y.rows === 0) throw new DomainError('metrics', 'metrics: a point set is empty')
  const D = toFlat(squaredDistances(matrix(X.data, X.rows, X.cols), matrix(Y.data, Y.rows, Y.cols)))
  return Float64Array.from({ length: X.rows }, (_, i) => {
    let best = Infinity
    for (let j = 0; j < Y.rows; j++) best = Math.min(best, D[i * Y.rows + j])
    return Math.sqrt(best)
  })
}

/** The Hausdorff distances between point sets X and Y (rows as points). */
export type HausdorffDistances = {
  /** H(X, Y) = max(h(X, Y), h(Y, X)). */
  hausdorff: number
  /** h(X, Y) = max over x of the distance to the nearest y. */
  directedXY: number
  /** h(Y, X). */
  directedYX: number
  /** The 95th percentile of the pooled nearest distances of both directions (linear quantile). */
  hd95: number
  /** Average symmetric surface distance: the mean of the pooled nearest distances. */
  averageSymmetric: number
}

/**
 * The Hausdorff distance and its robust relatives between two finite point sets (Huttenlocher et al. 1993;
 * hausdorff-distance). For segmentation, pass the boundary pixels of each region (see `maskBoundary`). HD95 pools both
 * directions before taking the percentile; implementations differ, so state it.
 */
export function hausdorffDistances(x: Rows, y: Rows): HausdorffDistances {
  const X = dense(x, 'hausdorffDistances')
  const Y = dense(y, 'hausdorffDistances')
  const xy = nearestDistances(X, Y)
  const yx = nearestDistances(Y, X)
  const pooled = Float64Array.from([...xy, ...yx])
  // A loop, not Math.max(...xy): spreading a data-sized array overflows the call stack past ~10⁵ entries.
  const directedXY = xy.reduce((m, v) => Math.max(m, v), -Infinity)
  const directedYX = yx.reduce((m, v) => Math.max(m, v), -Infinity)
  return {
    hausdorff: Math.max(directedXY, directedYX),
    directedXY,
    directedYX,
    hd95: quantile(pooled, 0.95),
    averageSymmetric: pooled.reduce((s, v) => s + v, 0) / pooled.length,
  }
}

/** The Hausdorff distance H(X, Y) as a metric. */
export const hausdorffDistance = defineMetric(
  {
    key: 'hausdorffDistance',
    stability: 'stable',
    name: 'Hausdorff distance',
    inputs: 'points',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['hausdorff-distance'],
  },
  (x: Rows, y: Rows): number => hausdorffDistances(x, y).hausdorff,
)
