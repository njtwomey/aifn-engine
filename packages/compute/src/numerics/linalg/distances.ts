/**
 * Distance matrices between two sets of points (rows): the one definition that `metrics`, `kernels`, `classify`,
 * `cluster` and `embed` build on. Differences are formed directly, not through ‖x‖² + ‖y‖² − 2xᵀy, which loses
 * precision far from the origin. The metrics are scipy's `cdist` ones (Virtanen et al., 2020, "SciPy 1.0", Nature
 * Methods 17), the reference the tests compare against.
 */

import { fromData, isTensor, type MatrixLike, type Tensor, toFlat } from 'aifn-compute/foundation/tensor'
import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A distance for `pairwiseDistances`. */
export type PairwiseMetric = 'euclidean' | 'sqeuclidean' | 'manhattan' | 'chebyshev' | 'minkowski' | 'cosine'

type Points = { rows: number; cols: number; data: Float64Array }

/** Points as rows: an n × d matrix, or a rank-1 tensor or array of numbers read as n points in one dimension. */
function points(x: MatrixLike | VectorLike, where: string): Points {
  if (isTensor(x)) {
    const data = Float64Array.from(toFlat(x))
    if (x.shape.length === 2) return { rows: x.shape[0], cols: x.shape[1], data }
    if (x.shape.length === 1) return { rows: x.shape[0], cols: 1, data }
    throw new ShapeError(where, `${where}: expected points as rows of a matrix, got shape [${x.shape.join(', ')}]`)
  }
  const n = x.length
  if (n === 0) return { rows: 0, cols: 0, data: new Float64Array(0) }
  if (typeof x[0] === 'number') return { rows: n, cols: 1, data: Float64Array.from(x as ArrayLike<number>) }
  const rows = x as ArrayLike<ArrayLike<number>>
  const cols = rows[0].length
  const data = new Float64Array(n * cols)
  for (let i = 0; i < n; i++) {
    const r = rows[i]
    if (r.length !== cols) throw new ShapeError(where, `${where}: row ${i} has ${r.length} values, expected ${cols}`)
    for (let j = 0; j < cols; j++) data[i * cols + j] = r[j]
  }
  return { rows: n, cols, data }
}

/**
 * The distance between row i of `a` and row j of `b` (row-major, d columns each): the scalar kernel of
 * `pairwiseDistances`, for loops that need one pair at a time (k-means++ seeding, neighbour scans). `metric` and `p` as
 * there (Minkowski order p, default 2); cosine distance is NaN when either row is zero. No checks: the caller owns the
 * shapes.
 */
export function rowDistance(
  a: ArrayLike<number>,
  i: number,
  b: ArrayLike<number>,
  j: number,
  d: number,
  metric: PairwiseMetric = 'euclidean',
  p = 2,
): number {
  let s = 0
  let na = 0
  let nb = 0
  const oa = i * d
  const ob = j * d
  for (let c = 0; c < d; c++) {
    const u = a[oa + c]
    const v = b[ob + c]
    const diff = Math.abs(u - v)
    if (metric === 'manhattan') s += diff
    else if (metric === 'chebyshev') s = Math.max(s, diff)
    else if (metric === 'minkowski') s += diff ** p
    else if (metric === 'cosine') {
      s += u * v
      na += u * u
      nb += v * v
    } else s += diff * diff
  }
  if (metric === 'euclidean') return Math.sqrt(s)
  if (metric === 'minkowski') return s ** (1 / p)
  if (metric === 'cosine') {
    const norms = Math.sqrt(na * nb)
    return norms > 0 ? 1 - s / norms : NaN
  }
  return s
}

/** ‖aᵢ − bⱼ‖² between row i of `a` and row j of `b` (d columns): `rowDistance` with the squared Euclidean metric. */
export const squaredRowDistance = (
  a: ArrayLike<number>,
  i: number,
  b: ArrayLike<number>,
  j: number,
  d: number,
): number => rowDistance(a, i, b, j, d, 'sqeuclidean')

/**
 * The m × n matrix of distances between the rows of X (m × d) and Y (n × d; default X): Euclidean (default), squared
 * Euclidean, Manhattan, Chebyshev, Minkowski of order `p` (default 2) or cosine distance 1 − cos θ (NaN when either
 * point is zero). A rank-1 input is n points on a line.
 */
export function pairwiseDistances(
  x: MatrixLike | VectorLike,
  y?: MatrixLike | VectorLike,
  options: { metric?: PairwiseMetric; p?: number } = {},
): Tensor {
  const X = points(x, 'pairwiseDistances')
  const Y = y === undefined ? X : points(y, 'pairwiseDistances')
  if (X.rows > 0 && Y.rows > 0 && X.cols !== Y.cols)
    throw new ShapeError('pairwiseDistances', 'pairwiseDistances: the two sets differ in dimension')
  const d = X.cols
  const metric = options.metric ?? 'euclidean'
  const p = options.p ?? 2
  const out = new Float64Array(X.rows * Y.rows)
  for (let i = 0; i < X.rows; i++)
    for (let j = 0; j < Y.rows; j++) out[i * Y.rows + j] = rowDistance(X.data, i, Y.data, j, d, metric, p)
  return fromData(out, [X.rows, Y.rows])
}

/** The m × n matrix of squared Euclidean distances ‖xᵢ − yⱼ‖² between the rows of X and Y (default X). */
export function squaredDistances(x: MatrixLike | VectorLike, y?: MatrixLike | VectorLike): Tensor {
  return pairwiseDistances(x, y, { metric: 'sqeuclidean' })
}
