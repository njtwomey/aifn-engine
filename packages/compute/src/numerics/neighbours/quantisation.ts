/**
 * Indexes by quantisation:
 *
 * - **Inverted file (IVF)** (Sivic and Zisserman 2003, "Video Google", ICCV; Jégou, Douze and Schmid 2011): a coarse
 *   k-means codebook of `lists` centroids splits the points into cells; a query scans only the points of its `probes`
 *   nearest cells.
 * - **Product quantisation (PQ)** (Jégou, Douze and Schmid 2011, "Product quantization for nearest neighbor search",
 *   IEEE TPAMI 33(1)): each vector is cut into M sub-vectors and each sub-vector is replaced by the index of its nearest
 *   of K sub-codewords, so a vector costs M·log₂K bits. Asymmetric distance computation (ADC) answers a query from a
 *   table of its squared distances to every sub-codeword: d̂(q, x)² = Σₘ T[m, codeₘ(x)].
 * - **Optimised PQ (OPQ)** (Ge, He, Ke and Sun 2014, "Optimized product quantization", IEEE TPAMI 36(4), the
 *   non-parametric method): learn an orthogonal rotation R with the codebooks by alternating PQ on XR with the
 *   orthogonal Procrustes update R = UVᵀ, where UΣVᵀ is the SVD of Xᵀ Ŷ and Ŷ the decoded XR.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { squaredRowDistance, svd } from 'aifn-compute/numerics/linalg'
import { trainCodebook } from './codebook'
import { checkK, distanceOf, kBest, queryOf, rowsOf, stackResults, type Neighbours, type QueryResult } from './search'

// ── Inverted file ────────────────────────────────────────────────────────────────────────────────────────────────────

/** An inverted-file index. */
export interface IvfIndex {
  readonly kind: 'ivf-index'
  readonly n: Size
  readonly d: Size
  readonly data: Float64Array
  /** The coarse centroids [lists, d]. */
  readonly centroids: Tensor
  /** The cell of every point [n] (int32). */
  readonly assignment: Tensor
  /** The points of each cell, in index order. */
  readonly lists: readonly (readonly number[])[]
}

/** Options of {@link ivfIndex}. */
export interface IvfOptions {
  /** Number of cells (coarse centroids). */
  lists: Size
  stream: Stream
  /** Lloyd iterations of the coarse quantiser (default 25). */
  iterations?: number
}

/** Build an inverted file over the rows of x: k-means cells, each with the list of its points. */
export function ivfIndex(x: MatrixLike, options: IvfOptions): IvfIndex {
  const X = rowsOf(x, 'ivfIndex')
  const code = trainCodebook(fromData(X.data, [X.n, X.d]), options.lists, {
    stream: child(options.stream, 'coarse'),
    iterations: options.iterations,
  })
  const labels = dense.data(code.labels)
  const lists: number[][] = Array.from({ length: code.centroids.shape[0] }, () => [])
  for (let i = 0; i < X.n; i++) lists[labels[i]].push(i)
  return {
    kind: 'ivf-index',
    n: X.n,
    d: X.d,
    data: Float64Array.from(X.data),
    centroids: code.centroids,
    assignment: code.labels,
    lists,
  }
}

/** The answer to one IVF query with the cells probed. */
export interface IvfQueryResult extends QueryResult {
  /** The cells scanned, nearest centroid first. */
  readonly probed: readonly number[]
}

/**
 * The k nearest points (Euclidean) among the cells of the `probes` centroids nearest the query. The distance count
 * includes the query's distances to the centroids.
 */
export function ivfQuery(index: IvfIndex, query: VectorLike, k: Size, options: { probes?: Size } = {}): IvfQueryResult {
  const q = queryOf(query, index.d, 'ivfQuery')
  checkK(k, index.n, 'ivfQuery')
  const C = dense.data(index.centroids)
  const L = index.lists.length
  const probes = Math.min(L, options.probes ?? 1)
  const cells = Array.from({ length: L }, (_, c) => ({ c, d: squaredRowDistance(q, 0, C, c, index.d) }))
    .sort((a, b) => a.d - b.d || a.c - b.c)
    .slice(0, probes)
    .map((e) => e.c)
  const best = kBest(k)
  let evaluations = L
  for (const c of cells)
    for (const j of index.lists[c]) {
      best.offer(j, distanceOf('euclidean', q, 0, index.data, j, index.d))
      evaluations++
    }
  return { indices: best.indices, distances: best.distances, distanceEvaluations: evaluations, probed: cells }
}

/** {@link ivfQuery} for every row of `queries`. */
export function ivfSearch(index: IvfIndex, queries: MatrixLike, k: Size, options: { probes?: Size } = {}): Neighbours {
  const Q = rowsOf(queries, 'ivfSearch')
  return stackResults(
    Array.from({ length: Q.n }, (_, i) => ivfQuery(index, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k, options)),
    k,
  )
}

// ── Product quantisation ─────────────────────────────────────────────────────────────────────────────────────────────

/** A trained product quantiser, optionally with an OPQ rotation applied first. */
export interface ProductQuantiser {
  readonly kind: 'product-quantiser'
  readonly d: Size
  /** Sub-spaces M (d must be a multiple of M). */
  readonly subspaces: Size
  /** Sub-codewords per sub-space K. */
  readonly codewords: Size
  /** The sub-codebooks [M, K, d/M]. */
  readonly codebooks: Tensor
  /** OPQ: the orthogonal rotation R [d, d] (vectors are coded as xR); absent for plain PQ. */
  readonly rotation?: Tensor
  /** Mean squared reconstruction error ‖x − x̂‖² over the training rows. */
  readonly distortion: number
}

/** Options of {@link productQuantiser}. */
export interface ProductQuantiserOptions {
  subspaces: Size
  codewords: Size
  stream: Stream
  /** Lloyd iterations per sub-codebook (default 25). */
  iterations?: number
}

function rotate(X: Float64Array, n: number, d: number, R: Tensor | undefined): Float64Array {
  return R ? dense.matMul(X, dense.data(R), n, d, d) : X
}

function trainSub(
  Y: Float64Array,
  n: number,
  d: number,
  options: ProductQuantiserOptions,
  initial?: Float64Array,
): Float64Array {
  const { subspaces: M, codewords: K } = options
  const ds = d / M
  const books = new Float64Array(M * K * ds)
  for (let m = 0; m < M; m++) {
    const sub = new Float64Array(n * ds)
    for (let i = 0; i < n; i++) for (let c = 0; c < ds; c++) sub[i * ds + c] = Y[i * d + m * ds + c]
    const code = trainCodebook(fromData(sub, [n, ds]), K, {
      stream: child(options.stream, 'subspace', m),
      iterations: options.iterations,
      ...(initial && K <= n ? { initial: fromData(initial.slice(m * K * ds, (m + 1) * K * ds), [K, ds]) } : {}),
    })
    const cw = dense.data(code.centroids)
    // Fewer rows than codewords: the spare codewords repeat the last one (never chosen before it).
    for (let k = 0; k < K; k++) {
      const from = Math.min(k, code.centroids.shape[0] - 1)
      books.set(cw.subarray(from * ds, (from + 1) * ds), (m * K + k) * ds)
    }
  }
  return books
}

function encodeRows(Y: Float64Array, n: number, d: number, M: number, K: number, books: Float64Array): Int32Array {
  const ds = d / M
  const codes = new Int32Array(n * M)
  for (let i = 0; i < n; i++)
    for (let m = 0; m < M; m++) {
      let best = 0
      let bestD = Infinity
      for (let k = 0; k < K; k++) {
        let s = 0
        for (let c = 0; c < ds; c++) {
          const t = Y[i * d + m * ds + c] - books[(m * K + k) * ds + c]
          s += t * t
        }
        if (s < bestD) {
          bestD = s
          best = k
        }
      }
      codes[i * M + m] = best
    }
  return codes
}

function decodeRows(codes: Int32Array, n: number, d: number, M: number, K: number, books: Float64Array): Float64Array {
  const ds = d / M
  const out = new Float64Array(n * d)
  for (let i = 0; i < n; i++)
    for (let m = 0; m < M; m++)
      out.set(books.subarray((m * K + codes[i * M + m]) * ds, (m * K + codes[i * M + m] + 1) * ds), i * d + m * ds)
  return out
}

function checkPq(d: number, M: number, K: number, op: string): void {
  if (!(Number.isInteger(M) && M >= 1 && d % M === 0))
    throw new DomainError(op, `${op}: the width ${d} must be a multiple of the number of sub-spaces ${M}`)
  if (!(Number.isInteger(K) && K >= 1)) throw new DomainError(op, `${op}: codewords must be a positive integer`)
}

const meanSquaredError = (a: Float64Array, b: Float64Array, n: number) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
  return s / n
}

/** Train a product quantiser on the rows of x (n × d): one k-means codebook of K codewords per sub-space. */
export function productQuantiser(x: MatrixLike, options: ProductQuantiserOptions): ProductQuantiser {
  const X = rowsOf(x, 'productQuantiser')
  const { subspaces: M, codewords: K } = options
  checkPq(X.d, M, K, 'productQuantiser')
  const books = trainSub(X.data, X.n, X.d, options)
  const decoded = decodeRows(encodeRows(X.data, X.n, X.d, M, K, books), X.n, X.d, M, K, books)
  return {
    kind: 'product-quantiser',
    d: X.d,
    subspaces: M,
    codewords: K,
    codebooks: fromData(books, [M, K, X.d / M]),
    distortion: meanSquaredError(X.data, decoded, X.n),
  }
}

/** Options of {@link optimisedProductQuantiser}. */
export interface OpqOptions extends ProductQuantiserOptions {
  /** Alternations of PQ training and the Procrustes rotation update (default 10). */
  rounds?: Size
}

/** An OPQ quantiser and the distortion after each round (round 0 is plain PQ, R = I). */
export interface OpqResult extends ProductQuantiser {
  readonly distortions: readonly number[]
}

/**
 * Optimised product quantisation, non-parametric (Ge et al. 2014, Algorithm 1): start from R = I; each round runs
 * Lloyd's iterations on XR from the last round's sub-codebooks, decodes Ŷ, and sets R = UVᵀ from the SVD
 * Xᵀ Ŷ = UΣVᵀ, the rotation that best maps X onto Ŷ. Every step lowers ‖XR − Ŷ‖², so the distortion never rises.
 */
export function optimisedProductQuantiser(x: MatrixLike, options: OpqOptions): OpqResult {
  const X = rowsOf(x, 'optimisedProductQuantiser')
  const { subspaces: M, codewords: K, rounds = 10 } = options
  checkPq(X.d, M, K, 'optimisedProductQuantiser')
  const { n, d } = X
  let R = fromData(dense.identity(d), [d, d])
  const distortions: number[] = []
  let books: Float64Array = new Float64Array(0)
  for (let r = 0; r <= rounds; r++) {
    const Y = rotate(X.data, n, d, R)
    // Warm start from the last round's codebooks, so every round lowers the distortion (Ge et al. 2014, §3.2).
    books = trainSub(Y, n, d, options, r > 0 ? books : undefined)
    const Yhat = decodeRows(encodeRows(Y, n, d, M, K, books), n, d, M, K, books)
    distortions.push(meanSquaredError(Y, Yhat, n))
    if (r === rounds) break
    // Procrustes: maximise tr(Rᵀ Xᵀ Ŷ) over orthogonal R.
    const XtY = dense.matMul(dense.transpose(X.data, n, d), Yhat, d, n, d)
    const { U, V } = svd(fromData(XtY, [d, d]))
    R = fromData(dense.matMul(dense.data(U), dense.transpose(dense.data(V), d, d), d, d, d), [d, d])
  }
  return {
    kind: 'product-quantiser',
    d,
    subspaces: M,
    codewords: K,
    codebooks: fromData(books, [M, K, d / M]),
    rotation: R,
    distortion: distortions[distortions.length - 1],
    distortions,
  }
}

/** The PQ codes of the rows of x: [n, M] int32 sub-codeword indices. */
export function pqEncode(pq: ProductQuantiser, x: MatrixLike): Tensor {
  const X = rowsOf(x, 'pqEncode')
  if (X.d !== pq.d) throw new ShapeError('pqEncode', `pqEncode: the quantiser codes width ${pq.d}, data ${X.d}`)
  const Y = rotate(X.data, X.n, X.d, pq.rotation)
  return fromData(encodeRows(Y, X.n, X.d, pq.subspaces, pq.codewords, dense.data(pq.codebooks)), [X.n, pq.subspaces])
}

/** The reconstructions of codes [n, M], mapped back by Rᵀ when the quantiser has a rotation: [n, d]. */
export function pqDecode(pq: ProductQuantiser, codes: Tensor): Tensor {
  const n = codes.shape[0]
  const Yhat = decodeRows(
    Int32Array.from(dense.data(codes)),
    n,
    pq.d,
    pq.subspaces,
    pq.codewords,
    dense.data(pq.codebooks),
  )
  const out = pq.rotation
    ? dense.matMul(Yhat, dense.transpose(dense.data(pq.rotation), pq.d, pq.d), n, pq.d, pq.d)
    : Yhat
  return fromData(out, [n, pq.d])
}

/** The ADC table of a query: its squared distances to every sub-codeword [M, K] (after the rotation, if any). */
export function pqDistanceTable(pq: ProductQuantiser, query: VectorLike): Tensor {
  const q0 = queryOf(query, pq.d, 'pqDistanceTable')
  const q = rotate(q0, 1, pq.d, pq.rotation)
  const { subspaces: M, codewords: K } = pq
  const ds = pq.d / M
  const books = dense.data(pq.codebooks)
  const T = new Float64Array(M * K)
  for (let m = 0; m < M; m++)
    for (let k = 0; k < K; k++) {
      let s = 0
      for (let c = 0; c < ds; c++) s += (q[m * ds + c] - books[(m * K + k) * ds + c]) ** 2
      T[m * K + k] = s
    }
  return fromData(T, [M, K])
}

/**
 * The k nearest coded points to the query by asymmetric distance: d̂² = Σₘ T[m, codeₘ] from the query's table, so a
 * point costs M lookups instead of d multiplications. Distances are the estimates d̂ (not squared).
 */
export function pqQuery(pq: ProductQuantiser, codes: Tensor, query: VectorLike, k: Size): QueryResult {
  const T = dense.data(pqDistanceTable(pq, query))
  const n = codes.shape[0]
  checkK(k, n, 'pqQuery')
  const C = dense.data(codes)
  const { subspaces: M, codewords: K } = pq
  const best = kBest(k)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let m = 0; m < M; m++) s += T[m * K + C[i * M + m]]
    best.offer(i, Math.sqrt(s))
  }
  return { indices: best.indices, distances: best.distances, distanceEvaluations: M * K }
}

/** {@link pqQuery} for every row of `queries`. */
export function pqSearch(pq: ProductQuantiser, codes: Tensor, queries: MatrixLike, k: Size): Neighbours {
  const Q = rowsOf(queries, 'pqSearch')
  return stackResults(
    Array.from({ length: Q.n }, (_, i) => pqQuery(pq, codes, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k)),
    k,
  )
}
