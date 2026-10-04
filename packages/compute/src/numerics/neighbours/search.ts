/**
 * What every index of `aifn-compute/numerics/neighbours` shares: points read as rows, the distance between two rows, a bounded
 * list of the k best candidates, the result of a search and the exact search by brute force that every other method is
 * checked against.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { rowDistance } from 'aifn-compute/numerics/linalg'

/** The distances the indexes support: those of scipy's `cdist` that are metrics on rows, and the cosine distance. */
export type NeighbourMetric = 'euclidean' | 'sqeuclidean' | 'manhattan' | 'chebyshev' | 'cosine'

/** Points as rows: n × d values, row-major. */
export interface Rows {
  readonly n: Size
  readonly d: Size
  readonly data: Float64Array
}

/** The rows of a matrix (a rank-1 input is one point). */
export function rowsOf(x: MatrixLike | VectorLike, where: string): Rows {
  const vector = isTensor(x)
    ? x.shape.length === 1
    : (x as ArrayLike<unknown>).length > 0 && typeof (x as ArrayLike<unknown>)[0] === 'number'
  if (vector) {
    const data = dense.toF64(x as VectorLike, where)
    return { n: 1, d: data.length, data }
  }
  const { data, m, n } = dense.toMatrixF64(x as MatrixLike, where)
  return { n: m, d: n, data }
}

/** One query point as a float64 vector of width d. */
export function queryOf(q: VectorLike, d: Size, where: string): Float64Array {
  const v = dense.toF64(q, where)
  if (v.length !== d) throw new ShapeError(where, `${where}: the query has ${v.length} values, the index ${d}`)
  return v
}

/** The distance between row i of `a` and row j of `b` (both of width d) under a metric. */
export const distanceOf = (
  metric: NeighbourMetric,
  a: ArrayLike<number>,
  i: number,
  b: ArrayLike<number>,
  j: number,
  d: number,
): number => rowDistance(a, i, b, j, d, metric)

/** Throws unless k lies in 1 … n. */
export function checkK(k: Size, n: Size, where: string): void {
  if (!(Number.isInteger(k) && k >= 1 && k <= n))
    throw new DomainError(where, `${where}: k = ${k} must lie in 1 … ${n}`)
}

/**
 * The k best candidates seen so far, nearest first (ties to the smaller index): the bounded priority list every search
 * keeps. `worst()` is the distance a new candidate must beat (Infinity until k are held). A NaN distance (the cosine
 * distance of a zero row) ranks as +∞: it is held only when nothing better fills the list, and it is reported as NaN.
 */
export interface KBest {
  readonly k: Size
  readonly indices: number[]
  readonly distances: number[]
  worst(): number
  /** Offer a candidate; true when it entered the list. Each index is held at most once. */
  offer(index: number, distance: number): boolean
}

/** An empty {@link KBest} list of capacity k. */
export function kBest(k: Size): KBest {
  const indices: number[] = []
  const distances: number[] = []
  const key = (d: number) => (Number.isNaN(d) ? Infinity : d)
  return {
    k,
    indices,
    distances,
    worst: () => (indices.length < k ? Infinity : key(distances[k - 1])),
    offer(index, distance) {
      const d = key(distance)
      if (indices.length >= k) {
        const w = key(distances[k - 1])
        if (d > w || (d === w && index > indices[k - 1])) return false
      }
      if (indices.includes(index)) return false
      let r = indices.length
      while (r > 0 && (key(distances[r - 1]) > d || (key(distances[r - 1]) === d && indices[r - 1] > index))) r--
      indices.splice(r, 0, index)
      distances.splice(r, 0, distance)
      if (indices.length > k) {
        indices.pop()
        distances.pop()
      }
      return true
    },
  }
}

/** The answer to one query: the k nearest found, nearest first, and the number of distances computed to find them. */
export interface QueryResult {
  readonly indices: readonly number[]
  readonly distances: readonly number[]
  /** Distances between the query and stored points evaluated (the cost every index tries to cut). */
  readonly distanceEvaluations: number
}

/** The answers to m queries: indices (int32) and distances, both m × k, nearest first. */
export interface Neighbours {
  readonly kind: 'neighbours'
  readonly indices: Tensor
  readonly distances: Tensor
  /** Distances evaluated over all queries. */
  readonly distanceEvaluations: number
}

/** Stack per-query results (each of length k) into a `Neighbours`. */
export function stackResults(results: readonly QueryResult[], k: Size): Neighbours {
  const m = results.length
  const idx = new Int32Array(m * k).fill(-1)
  const dist = new Float64Array(m * k).fill(Infinity)
  let evaluations = 0
  results.forEach((r, i) => {
    for (let j = 0; j < Math.min(k, r.indices.length); j++) {
      idx[i * k + j] = r.indices[j]
      dist[i * k + j] = r.distances[j]
    }
    evaluations += r.distanceEvaluations
  })
  return {
    kind: 'neighbours',
    indices: fromData(idx, [m, k]),
    distances: fromData(dist, [m, k]),
    distanceEvaluations: evaluations,
  }
}

/** Options of {@link bruteForceNeighbours}. */
export interface BruteForceOptions {
  /** Default `euclidean`. */
  metric?: NeighbourMetric
  /** Leave each query's own row out (queries are the data, as in a k-NN graph). Default false. */
  excludeSelf?: boolean
}

/**
 * The exact k nearest rows of `data` (n × d) to each row of `queries` (m × d) by scanning all n: O(nmd), the reference
 * for every index. Nearest first, ties to the smaller index. With `excludeSelf` the queries must be the data and row i
 * never answers query i.
 */
export function bruteForceNeighbours(
  data: MatrixLike,
  queries: MatrixLike,
  k: Size,
  options: BruteForceOptions = {},
): Neighbours {
  const { metric = 'euclidean', excludeSelf = false } = options
  const X = rowsOf(data, 'bruteForceNeighbours')
  const Q = rowsOf(queries, 'bruteForceNeighbours')
  if (Q.d !== X.d)
    throw new ShapeError('bruteForceNeighbours', `bruteForceNeighbours: queries have ${Q.d} columns, data ${X.d}`)
  if (excludeSelf && Q.n !== X.n)
    throw new ShapeError('bruteForceNeighbours', 'bruteForceNeighbours: excludeSelf needs the queries to be the data')
  checkK(k, excludeSelf ? X.n - 1 : X.n, 'bruteForceNeighbours')
  const results: QueryResult[] = []
  for (let i = 0; i < Q.n; i++) {
    const best = kBest(k)
    for (let j = 0; j < X.n; j++) {
      if (excludeSelf && i === j) continue
      best.offer(j, distanceOf(metric, Q.data, i, X.data, j, X.d))
    }
    results.push({ indices: best.indices, distances: best.distances, distanceEvaluations: X.n })
  }
  return stackResults(results, k)
}

/** The exact k nearest rows of `data` to one query, by scanning (see {@link bruteForceNeighbours}). */
export function bruteForceQuery(
  data: MatrixLike,
  query: VectorLike,
  k: Size,
  options: { metric?: NeighbourMetric } = {},
): QueryResult {
  const X = rowsOf(data, 'bruteForceQuery')
  const q = queryOf(query, X.d, 'bruteForceQuery')
  checkK(k, X.n, 'bruteForceQuery')
  const best = kBest(k)
  for (let j = 0; j < X.n; j++) best.offer(j, distanceOf(options.metric ?? 'euclidean', q, 0, X.data, j, X.d))
  return { indices: best.indices, distances: best.distances, distanceEvaluations: X.n }
}
