/**
 * Private helpers for `aifn-methods/unsupervised/clustering`: float64 views (tensor's `dense.data`), the nearest-row
 * search Lloyd's steps use, pairwise distances (`aifn-compute/numerics/linalg`) and small tensors.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrixShape } from 'aifn-compute/learning/estimators'
import { pairwiseDistances, squaredRowDistance } from 'aifn-compute/numerics/linalg'

/** The elements of `t` in row-major order as float64 (`dense.data`: shared when already dense; do not mutate). */
export const values = (t: Tensor): Float64Array => dense.data(t)

/** Rows, columns and values of a matrix [n, d], or throw naming the caller. */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/** Squared Euclidean distance between row i of `a` and row j of `b` (both of width d): linalg's row kernel. */
export const sq = squaredRowDistance

/** All pairwise Euclidean distances between the rows of x [n, d], as a flat [n, n] array (`pairwiseDistances`). */
export function pairwise(x: Tensor): Float64Array {
  return dense.data(pairwiseDistances(x) as Tensor)
}

/** Nearest row of `c` [k, d] to row i of `v`: its index and squared distance (ties to the lower index). */
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

export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])
export const ints = (v: ArrayLike<number>): Tensor => fromData(Int32Array.from(v), [v.length])
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

/** Labels renumbered 0, 1, … in order of first appearance; negative labels (noise) are kept. */
export function canonical(labels: ArrayLike<number>): Int32Array {
  const map = new Map<number, number>()
  return Int32Array.from(labels, (l) => {
    if (l < 0) return l
    if (!map.has(l)) map.set(l, map.size)
    return map.get(l)!
  })
}
