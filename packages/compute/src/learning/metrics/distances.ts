/**
 * Distances and similarities: Minkowski distances (Manhattan, Euclidean, Chebyshev), cosine similarity and distance,
 * angular and Mahalanobis distances, orthogonal Procrustes alignment and the Procrustes disparity, and the Hausdorff
 * family of distances between point sets (Hausdorff, HD95, average symmetric surface distance).
 *
 * The vector distances take two vectors $\xvec$, $\yvec$ of the same length (arrays or tensors, read flattened) and
 * throw `ShapeError` otherwise, as scipy.spatial.distance. The point-set functions take matrices with one point per
 * row. Matrices of distances between every pair of points are `pairwiseDistances` and `squaredDistances` of
 * `aifn-compute/numerics/linalg`.
 */

import { det, solve, squaredDistances, svd } from 'aifn-compute/numerics/linalg'
import { quantile } from 'aifn-compute/probability/stats'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { defineMetric, dense, divide, matrix, sameLength, values, vector, type Data, type Rows } from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * Two vectors as new arrays, checked to have the same length.
 *
 * @param x The first vector (an array or a tensor, read flattened).
 * @param y The second vector.
 * @param what The caller's name for error messages.
 * @returns `a` and `b`, the two vectors as Float64Arrays. Throws `ShapeError` when their lengths differ.
 */
function pairOf(x: Data, y: Data, what: string) {
  const a = values(x)
  const b = values(y)
  sameLength(a, b, what)
  return { a, b }
}

/**
 * The registry metadata of a vector distance: stable, read from two `vectors`, lower is better.
 *
 * @param key The metric's registry key (its export name).
 * @param name The metric's display name.
 * @param range The metric's range of values.
 * @returns The metadata, with its literal fields kept.
 */
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

/**
 * The Minkowski distance $(\sum_k \lvert x_k - y_k \rvert^p)^{1/p}$ for $p \ge 1$ (default 2); $p = \infty$ gives the
 * Chebyshev distance. A $p$ below 1 is not checked, and gives a value that breaks the triangle inequality.
 *
 * @param x The first vector.
 * @param y The second vector, of the same length.
 * @param options `p`, the order of the norm (default 2; `Infinity` for the largest difference).
 * @returns The distance $\lVert \xvec - \yvec \rVert_p$.
 *
 * @example The 3-4-5 triangle at four orders
 * for (const p of [1, 2, 3, Infinity]) print('p =', p, minkowskiDistance([0, 0], [3, 4], { p }))
 */
export const minkowskiDistance = defineMetric(
  distanceInfo('minkowskiDistance', 'Minkowski distance'),
  (x: Data, y: Data, options: { p?: number } = {}): number => {
    const { a, b } = pairOf(x, y, 'minkowskiDistance')
    const p = options.p ?? 2
    if (p === Infinity) return a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0)
    return a.reduce((s, v, i) => s + Math.abs(v - b[i]) ** p, 0) ** (1 / p)
  },
)

/**
 * The Euclidean distance $\lVert \xvec - \yvec \rVert_2$.
 *
 * @param x The first vector.
 * @param y The second vector, of the same length.
 * @returns The distance.
 *
 * @example The 3-4-5 triangle
 * print(euclideanDistance([0, 0], [3, 4]))
 */
export const euclideanDistance = defineMetric(
  distanceInfo('euclideanDistance', 'Euclidean distance'),
  (x: Data, y: Data) => minkowskiDistance(x, y, { p: 2 }),
)

/**
 * The Manhattan distance $\sum_k \lvert x_k - y_k \rvert$.
 *
 * @param x The first vector.
 * @param y The second vector, of the same length.
 * @returns The distance.
 *
 * @example Three along and four up
 * print(manhattanDistance([0, 0], [3, 4]))
 */
export const manhattanDistance = defineMetric(
  distanceInfo('manhattanDistance', 'Manhattan distance'),
  (x: Data, y: Data) => minkowskiDistance(x, y, { p: 1 }),
)

/**
 * The Chebyshev distance $\max_k \lvert x_k - y_k \rvert$.
 *
 * @param x The first vector.
 * @param y The second vector, of the same length.
 * @returns The distance.
 *
 * @example The larger of the two differences
 * print(chebyshevDistance([0, 0], [3, 4]))
 */
export const chebyshevDistance = defineMetric(
  distanceInfo('chebyshevDistance', 'Chebyshev distance'),
  (x: Data, y: Data) => minkowskiDistance(x, y, { p: Infinity }),
)

/**
 * Cosine similarity $\xvec^\top\yvec/(\lVert \xvec \rVert \lVert \yvec \rVert)$; NaN when either vector is zero.
 *
 * @param x The first vector.
 * @param y The second vector, of the same length.
 * @returns The cosine of the angle between them, in $[-1, 1]$.
 *
 * @example At 45 degrees, opposite, and with a zero vector
 * print('45 degrees', cosineSimilarity([1, 0], [1, 1]))
 * print('opposite', cosineSimilarity([1, 0], [-2, 0]))
 * print('zero', cosineSimilarity([1, 0], [0, 0]))
 */
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

/**
 * Cosine distance $1 - \cos\theta$ (not a metric: it can break the triangle inequality); NaN when either vector is
 * zero. As `scipy.spatial.distance.cosine`.
 *
 * @param x The first vector.
 * @param y The second vector, of the same length.
 * @returns The distance, in $[0, 2]$.
 *
 * @example Orthogonal and opposite vectors
 * print('orthogonal', cosineDistance([1, 0], [0, 1]))
 * print('opposite', cosineDistance([1, 0], [-1, 0]))
 */
export const cosineDistance = defineMetric(
  distanceInfo('cosineDistance', 'Cosine distance', [0, 2]),
  (x: Data, y: Data) => 1 - cosineSimilarity(x, y),
)

/**
 * Angular distance $\arccos(\cos\theta)/\pi$, a true metric on directions, in $[0, 1]$; NaN when either vector is
 * zero.
 *
 * @param x The first vector.
 * @param y The second vector, of the same length.
 * @returns The angle between them as a fraction of $\pi$.
 *
 * @example A quarter turn is half the largest angle
 * print('45 degrees', angularDistance([1, 0], [1, 1]))
 * print('90 degrees', angularDistance([1, 0], [0, 1]))
 */
export const angularDistance = defineMetric(
  distanceInfo('angularDistance', 'Angular distance', [0, 1]),
  (x: Data, y: Data) => Math.acos(Math.max(-1, Math.min(1, cosineSimilarity(x, y)))) / Math.PI,
)

/**
 * The Mahalanobis distance $\sqrt{(\xvec - \yvec)^\top \Sigmamat^{-1} (\xvec - \yvec)}$ for a covariance matrix
 * $\Sigmamat$ (by a linear solve, not an inverse). As `scipy.spatial.distance.mahalanobis`, which takes
 * $\Sigmamat^{-1}$ instead. Throws `LinAlgError` for a singular $\Sigmamat$.
 *
 * @param x The first vector, of length $d$.
 * @param y The second vector, of the same length.
 * @param options `covariance`, the $d \times d$ covariance matrix $\Sigmamat$ (symmetric positive definite).
 * @returns The distance.
 *
 * @example Two steps along a direction of standard deviation 2 is one unit
 * const covariance = [
 *   [4, 0],
 *   [0, 1],
 * ]
 * print('along x', mahalanobisDistance([2, 0], [0, 0], { covariance }))
 * print('along y', mahalanobisDistance([0, 2], [0, 0], { covariance }))
 */
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
  /**
   * The $d \times d$ orthogonal $\Rmat$ with $\Rmat\xvec_i \approx \yvec_i$ (so $\Xmat\Rmat^\top \approx \Ymat$ for
   * points as rows; scipy's `orthogonal_procrustes` returns $\Rmat^\top$).
   */
  rotation: Tensor
  /** The scale $s$ (1 unless `scale: true`). */
  scale: number
  /** The translation $\tvec$ (zero unless `centre: true`): $\yvec_i \approx s\Rmat\xvec_i + \tvec$. */
  translation: Tensor
  /** The residual $\lVert s\Xmat\Rmat^\top + \ones\tvec^\top - \Ymat \rVert_F^2$. */
  residual: number
  /** True when $\Rmat$ is a reflection ($\det\Rmat = -1$). */
  reflection: boolean
}

/**
 * Orthogonal Procrustes (Schönemann 1966; procrustes-analysis): the orthogonal $\Rmat$ minimising
 * $\lVert \Rmat\Amat - \Bmat \rVert_F$ for matched points as the rows of $\Xmat$ and $\Ymat$
 * ($\Amat = \Xmat^\top$, $\Bmat = \Ymat^\top$), $\Rmat = \Vmat\Umat^\top$ from the SVD
 * $\Amat\Bmat^\top = \Umat\Sigmamat\Vmat^\top$. With `reflection: false` a reflection is turned into the best proper
 * rotation (Kabsch 1976). Optionally also fits a translation (`centre`, which centres both sets first) and a scale
 * $s = \sum_k \sigma_k/\lVert \Amat \rVert_F^2$ (`scale`). Throws `ShapeError` when the two sets differ in shape.
 *
 * @param x The points to move: an $n \times d$ matrix, one point per row.
 * @param y The target points, matched row by row: an $n \times d$ matrix.
 * @param options `reflection` (default allowed): `false` restricts $\Rmat$ to proper rotations; `centre` also fits a
 *   translation; `scale` also fits a scale.
 * @returns The fitted rotation, scale and translation, the residual and whether $\Rmat$ is a reflection.
 *
 * @example Recover a quarter turn, then a turn with a scale and a shift
 * const x = [
 *   [1, 0],
 *   [0, 1],
 *   [-1, 0],
 * ]
 * const turned = [
 *   [0, 1],
 *   [-1, 0],
 *   [0, -1],
 * ]
 * const fit = orthogonalProcrustes(x, turned)
 * print('R', fit.rotation, 'residual', fit.residual)
 * const moved = turned.map(([a, b]) => [2 * a + 1, 2 * b + 1])
 * const full = orthogonalProcrustes(x, moved, { centre: true, scale: true })
 * print('scale', full.scale, 'translation', full.translation, 'residual', full.residual)
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
 * Frobenius norm, the least squared error over rotations (reflections allowed) and scale, $1 - (\sum_k \sigma_k)^2$,
 * in $[0, 1]$ (Gower 1975; as SciPy's `procrustes`). Throws `DomainError` when a set has all its points equal. The
 * two sets' shapes are not checked against each other: pass the same $n \times d$.
 *
 * @param x The first point set: an $n \times d$ matrix, one point per row.
 * @param y The second point set, matched row by row.
 * @returns The disparity: 0 when one set is a similarity transform of the other.
 *
 * @example The scipy example, a rotated, scaled and shifted copy
 * const a = [
 *   [1, 3],
 *   [1, 2],
 *   [1, 1],
 *   [2, 1],
 * ]
 * const b = [
 *   [4, -2],
 *   [4, -4],
 *   [4, -6],
 *   [2, -6],
 * ]
 * print('similar shapes', procrustesDisparity(a, b))
 * print('different shapes', procrustesDisparity(a, [[0, 0], [1, 0], [2, 0], [0, 3]]))
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

/**
 * Each point of $\Xmat$'s Euclidean distance to its nearest point of $\Ymat$. Throws `ShapeError` when the sets
 * differ in dimension and `DomainError` when $\Ymat$ is empty.
 *
 * @param X The query points, row-major with their dimensions.
 * @param Y The reference points, row-major with their dimensions.
 * @returns One distance per row of `X`.
 */
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

/** The Hausdorff distances between point sets $\Xmat$ and $\Ymat$ (rows as points), as `hausdorffDistances` returns. */
export type HausdorffDistances = {
  /** $H(\Xmat, \Ymat) = \max(h(\Xmat, \Ymat), h(\Ymat, \Xmat))$. */
  hausdorff: number
  /** $h(\Xmat, \Ymat) = \max_{\xvec} \min_{\yvec} \lVert \xvec - \yvec \rVert$, as scipy's `directed_hausdorff`. */
  directedXY: number
  /** $h(\Ymat, \Xmat)$. */
  directedYX: number
  /** The 95th percentile of the pooled nearest distances of both directions (linear quantile). */
  hd95: number
  /** Average symmetric surface distance: the mean of the pooled nearest distances. */
  averageSymmetric: number
}

/**
 * The Hausdorff distance and its robust relatives between two finite point sets (Huttenlocher et al. 1993;
 * hausdorff-distance). For segmentation, pass the boundary pixels of each region (see `maskBoundary` in
 * `aifn-methods/evaluation`). HD95 pools both directions before taking the percentile; implementations differ, so
 * state it. Throws `ShapeError` when the sets differ in dimension and `DomainError` when either is empty.
 *
 * @param x The first point set: a matrix with one point per row.
 * @param y The second point set, in the same dimension (the number of points may differ).
 * @returns The symmetric and both directed Hausdorff distances, HD95 and the average symmetric surface distance.
 *
 * @example A point set and one with a far point
 * print(hausdorffDistances([[0, 0], [1, 0]], [[0, 0], [3, 0]]))
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

/**
 * The Hausdorff distance $H(\Xmat, \Ymat)$ as a metric: the largest distance from a point of either set to the other
 * set.
 *
 * @param x The first point set: a matrix with one point per row.
 * @param y The second point set, in the same dimension.
 * @returns The Hausdorff distance.
 *
 * @example Decided by the far point of the second set
 * print(hausdorffDistance([[0, 0], [1, 0]], [[0, 0], [3, 0]]))
 */
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
