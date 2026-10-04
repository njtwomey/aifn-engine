/**
 * Distance matrices between two sets of points (rows): the one definition that `metrics`, `kernels`, `classify`,
 * `cluster` and `embed` build on. Differences are formed directly, not through
 * $\lVert \xvec \rVert^2 + \lVert \yvec \rVert^2 - 2\xvec^\top\yvec$, which loses precision far from the origin. The
 * metrics are scipy's `cdist` ones (Virtanen et al., 2020, "SciPy 1.0", Nature Methods 17), the reference the tests
 * compare against.
 */

import { fromData, isTensor, type MatrixLike, type Tensor, toFlat } from 'aifn-compute/foundation/tensor'
import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A distance for `pairwiseDistances`. */
export type PairwiseMetric = 'euclidean' | 'sqeuclidean' | 'manhattan' | 'chebyshev' | 'minkowski' | 'cosine'

type Points = { rows: number; cols: number; data: Float64Array }

/**
 * Points as rows: an $n \times d$ matrix, or a rank-1 tensor or array of numbers read as $n$ points in one dimension.
 *
 * @param x The points: a matrix (a rank-2 tensor or an array of equal-length rows) with one point per row and one
 *   coordinate per column, or a rank-1 tensor or plain array of $n$ numbers, read as $n$ points on a line. It is
 *   copied, not modified.
 * @param where The name of the calling function, used as the prefix of error messages (a tensor of rank above 2, or
 *   rows of unequal length, throws `ShapeError`).
 * @returns `rows`, the number of points $n$; `cols`, their dimension $d$; and `data`, a new flat row-major array of
 *   $n \cdot d$ values in which point $i$ occupies entries $i \cdot d$ to $i \cdot d + d - 1$. An empty array gives 0
 *   rows and 0 columns.
 */
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
 * The distance between row $i$ of `a` and row $j$ of `b` (row-major, $d$ columns each): the scalar kernel of
 * `pairwiseDistances`, for loops that need one pair at a time (k-means++ seeding, neighbour scans). `metric` and `p` as
 * there (Minkowski order $p$, default 2); cosine distance is NaN when either row is zero. No checks: the caller owns
 * the shapes.
 *
 * @param a The first set of points, as a flat row-major array: one point per row, $d$ coordinates per point, so point
 *   $i$ occupies entries $i \cdot d$ to $i \cdot d + d - 1$. Only that one row is read; nothing is modified.
 * @param i The index (from 0) of the point of `a` to measure from: a row number, not an offset into the array.
 * @param b The second set of points, laid out as `a` with the same $d$: point $j$ occupies entries $j \cdot d$ to
 *   $j \cdot d + d - 1$. It may be the same array as `a`, for distances within one set. Only that one row is read.
 * @param j The index (from 0) of the point of `b` to measure to: a row number, not an offset into the array.
 * @param d The number of coordinates of each point (the number of columns of both `a` and `b`). With $d = 0$ the
 *   result is 0, or NaN for the cosine metric.
 * @param metric Which distance to compute between the two rows $\uvec$ and $\vvec$ (default `'euclidean'`):
 *   `'euclidean'` is $\sqrt{\sum_c (u_c - v_c)^2}$, `'sqeuclidean'` its square $\sum_c (u_c - v_c)^2$, `'manhattan'`
 *   $\sum_c \lvert u_c - v_c \rvert$, `'chebyshev'` $\max_c \lvert u_c - v_c \rvert$, `'minkowski'`
 *   $(\sum_c \lvert u_c - v_c \rvert^p)^{1/p}$, and `'cosine'`
 *   $1 - \uvec^\top\vvec / (\lVert \uvec \rVert \lVert \vvec \rVert)$.
 * @param p The order $p$ of the Minkowski distance (default 2, which is the Euclidean distance; 1 is Manhattan). Used
 *   only when `metric` is `'minkowski'`, and ignored otherwise.
 * @returns The distance between the two points, in the units of the coordinates (squared units for `'sqeuclidean'`;
 *   a number in $[0, 2]$ for `'cosine'`, or NaN when either point is the zero vector).
 *
 * @example The distance between two rows of flat row-major arrays
 * // Two points in the plane, stored as rows of a flat array.
 * const points = [0, 0, 3, 4]
 * print('euclidean =', rowDistance(points, 0, points, 1, 2))
 * print('manhattan =', rowDistance(points, 0, points, 1, 2, 'manhattan'))
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

/**
 * $\lVert \avec_i - \bvec_j \rVert^2$ between row $i$ of `a` and row $j$ of `b` ($d$ columns): `rowDistance` with the
 * squared Euclidean metric.
 *
 * @param a The first set of points, as a flat row-major array: point $i$ occupies entries $i \cdot d$ to
 *   $i \cdot d + d - 1$. Read, not modified.
 * @param i The index (from 0) of the point of `a`: a row number, not an offset into the array.
 * @param b The second set of points, laid out as `a` with the same $d$ (it may be the same array). Read, not modified.
 * @param j The index (from 0) of the point of `b`: a row number, not an offset into the array.
 * @param d The number of coordinates of each point (the number of columns of both `a` and `b`).
 * @returns The sum of squared coordinate differences $\sum_c (a_{ic} - b_{jc})^2$, in squared units of the
 *   coordinates.
 *
 * @example The squared Euclidean distance between two rows
 * const points = [0, 0, 3, 4]
 * print(squaredRowDistance(points, 0, points, 1, 2))
 */
export const squaredRowDistance = (
  a: ArrayLike<number>,
  i: number,
  b: ArrayLike<number>,
  j: number,
  d: number,
): number => rowDistance(a, i, b, j, d, 'sqeuclidean')

/**
 * The $m \times n$ matrix of distances between the rows of $\Xmat$ ($m \times d$) and $\Ymat$ ($n \times d$; default
 * $\Xmat$): Euclidean (default), squared Euclidean, Manhattan, Chebyshev, Minkowski of order `p` (default 2) or cosine
 * distance $1 - \cos\theta$ (NaN when either point is zero). A rank-1 input is $n$ points on a line.
 *
 * @param x The first set of points $\Xmat$: an $m \times d$ matrix (tensor or array of rows) with one point per row,
 *   or a rank-1 tensor or array of $m$ numbers read as $m$ points in one dimension. Read, not modified.
 * @param y The second set of points $\Ymat$, $n \times d$ with the same $d$ as `x` (else `ShapeError`). When omitted,
 *   the distances are between the rows of `x` themselves ($n = m$, zero diagonal).
 * @param options Which distance to compute; the default is the Euclidean distance.
 * @param options.metric The distance between two rows, as in `rowDistance`: `'euclidean'` (default), `'sqeuclidean'`,
 *   `'manhattan'`, `'chebyshev'`, `'minkowski'` or `'cosine'`.
 * @param options.p The order $p$ of the Minkowski distance (default 2); used only with `metric: 'minkowski'`.
 * @returns An $m \times n$ tensor whose entry $(i, j)$ is the distance from row $i$ of `x` to row $j$ of `y`.
 *
 * @example Distances between every pair of rows
 * const X = tensor([[0, 0], [3, 4], [6, 8]])
 * print('euclidean =', pairwiseDistances(X))
 * print('manhattan =', pairwiseDistances(X, undefined, { metric: 'manhattan' }))
 *
 * @example Between two different sets of points
 * const X = tensor([[0, 0], [1, 1]])
 * const Y = tensor([[3, 4]])
 * print(pairwiseDistances(X, Y))
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

/**
 * The $m \times n$ matrix of squared Euclidean distances $\lVert \xvec_i - \yvec_j \rVert^2$ between the rows of
 * $\Xmat$ and $\Ymat$ (default $\Xmat$).
 *
 * @param x The first set of points $\Xmat$: an $m \times d$ matrix with one point per row, or a rank-1 tensor or
 *   array of $m$ numbers read as $m$ points in one dimension. Read, not modified.
 * @param y The second set of points $\Ymat$, $n \times d$ with the same $d$ as `x`. When omitted, the distances are
 *   between the rows of `x` themselves.
 * @returns An $m \times n$ tensor whose entry $(i, j)$ is the squared distance from row $i$ of `x` to row $j$ of `y`.
 *
 * @example Squared Euclidean distances between rows
 * const X = tensor([[0, 0], [3, 4], [6, 8]])
 * print(squaredDistances(X))
 */
export function squaredDistances(x: MatrixLike | VectorLike, y?: MatrixLike | VectorLike): Tensor {
  return pairwiseDistances(x, y, { metric: 'sqeuclidean' })
}
