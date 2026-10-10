/**
 * Private helpers for `aifn-methods/unsupervised/clustering`: float64 views (tensor's `dense.data`), the nearest-row
 * search Lloyd's steps use, pairwise distances (`aifn-compute/numerics/linalg`) and small tensors.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrixShape } from 'aifn-compute/learning/estimators'
import { pairwiseDistances, squaredRowDistance } from 'aifn-compute/numerics/linalg'

/**
 * The elements of a tensor in row-major order as float64 (`dense.data`: shared when already dense; do not mutate).
 *
 * @param t The tensor to read.
 * @returns Its elements, row-major; the tensor's own storage when it is already dense.
 */
export const values = (t: Tensor): Float64Array => dense.data(t)

/**
 * Rows, columns and values of a matrix ($n \times d$), or throw naming the caller.
 *
 * @param x The matrix, one row per point. Anything not two-dimensional throws a `ShapeError`.
 * @param where The caller's name for error messages.
 * @returns `n` rows, `d` columns, and `v`, the $nd$ values row-major (shared with `x`; do not mutate).
 */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/** Squared Euclidean distance between row $i$ of `a` and row $j$ of `b` (both of width $d$): linalg's row kernel. */
export const sq = squaredRowDistance

/**
 * All pairwise Euclidean distances between the rows of a matrix (`pairwiseDistances`).
 *
 * @param x The points, $n \times d$, one per row.
 * @returns The $n \times n$ distances as a flat row-major array of $n^2$ values.
 */
export function pairwise(x: Tensor): Float64Array {
  return dense.data(pairwiseDistances(x) as Tensor)
}

/**
 * Nearest row of `c` to row $i$ of `v`: its index and squared distance (ties to the lower index).
 *
 * @param v The points as a row-major array of width $d$; only row `i` is read.
 * @param i The row of `v` to look up.
 * @param c The candidate centres, $k \times d$, row-major.
 * @param k The number of rows of `c`.
 * @param d The number of columns of `v` and `c`.
 * @returns `[index, squared distance]` of the nearest row of `c`; `[0, Infinity]` when $k = 0$.
 */
export function nearest(v: Float64Array, i: number, c: Float64Array, k: number, d: number): [number, number] {
  let best = 0
  let bestD = Infinity
  for (let j = 0; j < k; j++) {
    const t = sq(v, i, c, j, d)
    if (t < bestD) {
      bestD = t
      best = j
    }
  }
  return [best, bestD]
}

/**
 * Wrap a row-major array as an $n \times d$ tensor (not copied).
 *
 * @param v The $nd$ values, row-major.
 * @param n The number of rows.
 * @param d The number of columns.
 * @returns The tensor of shape `[n, d]`.
 */
export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])
/**
 * Copy numbers into a one-dimensional int32 tensor.
 *
 * @param v The values (truncated to 32-bit integers).
 * @returns An int32 tensor of their length.
 */
export const ints = (v: ArrayLike<number>): Tensor => fromData(Int32Array.from(v), [v.length])
/**
 * Copy numbers into a one-dimensional float64 tensor.
 *
 * @param v The values.
 * @returns A float64 tensor of their length.
 */
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

/**
 * Labels renumbered $0, 1, \dots$ in order of first appearance; negative labels (noise) are kept. Two clusterings that
 * differ only in how their clusters are numbered come out the same.
 *
 * @param labels One cluster label per point, any integers; a negative one marks noise.
 * @returns The relabelled points as int32, with each negative label unchanged.
 *
 * @example Two numberings of one partition agree
 * print('a:', canonicalLabels([2, 2, -1, 0, 2, 0]))
 * print('b:', canonicalLabels([5, 5, -1, 1, 5, 1]))
 */
export function canonical(labels: ArrayLike<number>): Int32Array {
  const map = new Map<number, number>()
  return Int32Array.from(labels, (l) => {
    if (l < 0) return l
    if (!map.has(l)) map.set(l, map.size)
    return map.get(l)!
  })
}
