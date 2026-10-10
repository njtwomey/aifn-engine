/**
 * Shared by the embeddings of `aifn-methods/unsupervised/embedding`: squared distances between points (by
 * `aifn-compute/numerics/linalg`) and their $k$-nearest-neighbour lists.
 *
 * Both work on flat row-major `Float64Array`s, as the embeddings hold their data.
 */

import { dense, fromData } from 'aifn-compute/foundation/tensor'
import { squaredDistances as squaredDistanceMatrix } from 'aifn-compute/numerics/linalg'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Squared Euclidean distances $\lVert \xvec_i - \xvec_j \rVert^2$ between all rows of a matrix, by
 * `aifn-compute/numerics/linalg`'s `squaredDistances`.
 *
 * @param v The points as a row-major array of $n \times d$ values, one point per row (read, not modified).
 * @param n The number of points (rows).
 * @param d The number of features (columns).
 * @returns The squared distances as a row-major array of $n^2$ values; entry `i * n + j` is between rows `i` and `j`.
 */
export function squaredDistances(v: Float64Array, n: number, d: number): Float64Array {
  return dense.data(squaredDistanceMatrix(fromData(v, [n, d])))
}

/**
 * The $k$ nearest other points of each point under a distance matrix, nearest first; ties go to the lower index.
 * Throws `DomainError` unless $1 \le k \le n - 1$.
 *
 * @param D The distances (or squared distances: only their order matters) as a row-major array of $n^2$ values.
 * @param n The number of points.
 * @param k The number of neighbours per point, excluding the point itself.
 * @returns One array per point of its $k$ neighbours' indices, nearest first.
 */
export function nearestNeighbours(D: Float64Array, n: number, k: number): number[][] {
  if (!(k >= 1 && k < n)) throw new DomainError('nearest neighbours', `nearest neighbours: k must lie in 1 … ${n - 1}`)
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D[i * n + a] - D[i * n + b] || a - b)
      .slice(0, k),
  )
}
