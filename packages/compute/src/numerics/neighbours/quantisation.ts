/**
 * Indexes by quantisation:
 *
 * - **Inverted file (IVF)** (Sivic and Zisserman 2003, "Video Google", ICCV; Jégou, Douze and Schmid 2011): a coarse
 *   $k$-means codebook of `lists` centroids splits the points into cells; a query scans only the points of its `probes`
 *   nearest cells.
 * - **Product quantisation (PQ)** (Jégou, Douze and Schmid 2011, "Product quantization for nearest neighbor search",
 *   IEEE TPAMI 33(1)): each vector is cut into $M$ sub-vectors and each sub-vector is replaced by the index of its nearest
 *   of $K$ sub-codewords, so a vector costs $M \log_2 K$ bits. Asymmetric distance computation (ADC) answers a query from a
 *   table of its squared distances to every sub-codeword: $\hat{d}(q, \xvec)^2 = \sum_m T[m, \mathrm{code}_m(\xvec)]$.
 * - **Optimised PQ (OPQ)** (Ge, He, Ke and Sun 2014, "Optimized product quantization", IEEE TPAMI 36(4), the
 *   non-parametric method): learn an orthogonal rotation $\mathbf{R}$ with the codebooks by alternating PQ on $\mathbf{X}\mathbf{R}$ with the
 *   orthogonal Procrustes update $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$, where $\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ is the SVD of $\mathbf{X}^\top \hat{\mathbf{Y}}$ and $\hat{\mathbf{Y}}$ the decoded $\mathbf{X}\mathbf{R}$.
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
  /** Discriminator kind. */
  readonly kind: 'ivf-index'
  /** Total number of indexed points $n$. */
  readonly n: Size
  /** Dimensionality $d$. */
  readonly d: Size
  /** Flattened point data array ($n \times d$). */
  readonly data: Float64Array
  /** The coarse centroids tensor $[lists, d]$. */
  readonly centroids: Tensor
  /** The cell of every point $[n]$ (int32). */
  readonly assignment: Tensor
  /** The points of each cell, in index order. */
  readonly lists: readonly (readonly number[])[]
}

/** Options of {@link ivfIndex}. */
export interface IvfOptions {
  /** Number of cells (coarse centroids). */
  lists: Size
  /** Random stream. */
  stream: Stream
  /** Lloyd iterations of the coarse quantiser (default 25). */
  iterations?: number
}

/**
 * Build an inverted file over the rows of $x$: $k$-means cells, each with the list of its points.
 *
 * @param x - Input data matrix ($n \times d$).
 * @param options - IVF construction options.
 * @param options.lists - Number of cells (coarse centroids).
 * @param options.stream - Random stream.
 * @param options.iterations - Lloyd iterations for coarse quantiser (default 25).
 * @returns Built inverted-file index.
 *
 * @example Build an inverted-file index
 * const s = stream(42)
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const index = ivfIndex(data, { lists: 2, stream: s })
 * print('Lists:', index.lists.length)
 */
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
 * The $k$ nearest points (Euclidean) among the cells of the `probes` centroids nearest the query. The distance count
 * includes the query's distances to the centroids.
 *
 * @param index - IVF index.
 * @param query - Query vector of length $d$.
 * @param k - Number of nearest neighbours $k$ to return.
 * @param options - Query options.
 * @param options.probes - Number of coarse cells to probe (default 1).
 * @returns Query result containing $k$ nearest indices, distances, and probed cell list.
 *
 * @example Query nearest neighbours in IVF index
 * const s = stream(42)
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const index = ivfIndex(data, { lists: 2, stream: s })
 * const res = ivfQuery(index, [0.1, 0.1], 2, { probes: 2 })
 * print('Nearest index:', res.indices[0])
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

/**
 * Search nearest neighbours across multiple queries using an IVF index.
 *
 * @param index - IVF index.
 * @param queries - Query points matrix ($m \times d$).
 * @param k - Number of nearest neighbours $k$ to return per query.
 * @param options - Search options.
 * @param options.probes - Number of coarse cells to probe (default 1).
 * @returns Stacked `Neighbours` object.
 *
 * @example Search IVF index with multiple queries
 * const s = stream(42)
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const queries = [[0.1, 0.1], [2.1, 2.1]]
 * const index = ivfIndex(data, { lists: 2, stream: s })
 * const res = ivfSearch(index, queries, 2)
 * print('Nearest indices:\n' + res.indices)
 */
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
  /** Discriminator kind. */
  readonly kind: 'product-quantiser'
  /** Vector dimensionality $d$. */
  readonly d: Size
  /** Sub-spaces $M$ ($d$ must be a multiple of $M$). */
  readonly subspaces: Size
  /** Sub-codewords per sub-space $K$. */
  readonly codewords: Size
  /** The sub-codebooks tensor $[M, K, d/M]$. */
  readonly codebooks: Tensor
  /** OPQ: the orthogonal rotation $\mathbf{R}$ $[d, d]$ (vectors are coded as $\xvec\mathbf{R}$); absent for plain PQ. */
  readonly rotation?: Tensor
  /** Mean squared reconstruction error $\|\xvec - \hat{\xvec}\|^2$ over the training rows. */
  readonly distortion: number
}

/** Options of {@link productQuantiser}. */
export interface ProductQuantiserOptions {
  /** Number of sub-spaces $M$. */
  subspaces: Size
  /** Number of sub-codewords per sub-space $K$. */
  codewords: Size
  /** Random stream. */
  stream: Stream
  /** Lloyd iterations per sub-codebook (default 25). */
  iterations?: number
}

/**
 * Rotate a dataset matrix $X$ ($n \times d$) by an orthogonal rotation matrix $R$ ($d \times d$).
 *
 * @param X - Data array of size $n \times d$.
 * @param n - Number of rows $n$.
 * @param d - Dimensionality $d$.
 * @param R - Optional rotation tensor of shape $[d, d]$.
 * @returns Rotated data array ($n \times d$).
 */
function rotate(X: Float64Array, n: number, d: number, R: Tensor | undefined): Float64Array {
  return R ? dense.matMul(X, dense.data(R), n, d, d) : X
}

/**
 * Train sub-space codebooks for product quantisation.
 *
 * @param Y - Rotated data array ($n \times d$).
 * @param n - Number of rows $n$.
 * @param d - Dimensionality $d$.
 * @param options - Product quantiser options.
 * @param initial - Optional initial sub-codebooks array.
 * @returns Flattened array of trained sub-codebooks.
 */
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

/**
 * Encode rows into sub-codeword indices.
 *
 * @param Y - Rotated data array ($n \times d$).
 * @param n - Number of rows $n$.
 * @param d - Dimensionality $d$.
 * @param M - Number of sub-spaces $M$.
 * @param K - Number of codewords per sub-space $K$.
 * @param books - Sub-codebooks array.
 * @returns Flattened int32 array of codes ($n \times M$).
 */
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

/**
 * Decode sub-codeword indices back into approximate continuous vectors.
 *
 * @param codes - Int32 array of codes ($n \times M$).
 * @param n - Number of rows $n$.
 * @param d - Dimensionality $d$.
 * @param M - Number of sub-spaces $M$.
 * @param K - Number of codewords per sub-space $K$.
 * @param books - Sub-codebooks array.
 * @returns Flattened float64 array of reconstructed points ($n \times d$).
 */
function decodeRows(codes: Int32Array, n: number, d: number, M: number, K: number, books: Float64Array): Float64Array {
  const ds = d / M
  const out = new Float64Array(n * d)
  for (let i = 0; i < n; i++)
    for (let m = 0; m < M; m++)
      out.set(books.subarray((m * K + codes[i * M + m]) * ds, (m * K + codes[i * M + m] + 1) * ds), i * d + m * ds)
  return out
}

/**
 * Validate that product quantisation parameters are positive and dimensions divide evenly.
 *
 * @param d - Vector dimensionality $d$.
 * @param M - Number of sub-spaces $M$.
 * @param K - Number of codewords per sub-space $K$.
 * @param op - Operation name for error reporting.
 */
function checkPq(d: number, M: number, K: number, op: string): void {
  if (!(Number.isInteger(M) && M >= 1 && d % M === 0))
    throw new DomainError(op, `${op}: the width ${d} must be a multiple of the number of sub-spaces ${M}`)
  if (!(Number.isInteger(K) && K >= 1)) throw new DomainError(op, `${op}: codewords must be a positive integer`)
}

/**
 * Compute mean squared reconstruction error between original and decoded points.
 *
 * @param a - Original data array.
 * @param b - Decoded reconstruction array.
 * @param n - Number of points $n$.
 * @returns Mean squared error.
 */
const meanSquaredError = (a: Float64Array, b: Float64Array, n: number) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
  return s / n
}

/**
 * Train a product quantiser on the rows of $x$ ($n \times d$): one $k$-means codebook of $K$ codewords per sub-space.
 *
 * @param x - Input data matrix ($n \times d$).
 * @param options - PQ training options.
 * @param options.subspaces - Number of sub-spaces $M$.
 * @param options.codewords - Number of codewords per sub-space $K$.
 * @param options.stream - Random stream.
 * @param options.iterations - Lloyd iterations per sub-space (default 25).
 * @returns Trained product quantiser structure.
 *
 * @example Train a product quantiser
 * const s = stream(42)
 * const data = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
 * const pq = productQuantiser(data, { subspaces: 2, codewords: 2, stream: s })
 * print('Distortion:', pq.distortion)
 */
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

/** An OPQ quantiser and the distortion after each round (round 0 is plain PQ, $R = I$). */
export interface OpqResult extends ProductQuantiser {
  /** Distortion evaluated at each round. */
  readonly distortions: readonly number[]
}

/**
 * Optimised product quantisation, non-parametric (Ge et al. 2014, Algorithm 1): start from $\mathbf{R} = \mathbf{I}$; each round runs
 * Lloyd's iterations on $\mathbf{X}\mathbf{R}$ from the last round's sub-codebooks, decodes $\hat{\mathbf{Y}}$, and sets $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$ from the SVD
 * $\mathbf{X}^\top \hat{\mathbf{Y}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$, the rotation that best maps $\mathbf{X}$ onto $\hat{\mathbf{Y}}$. Every step lowers $\|\mathbf{X}\mathbf{R} - \hat{\mathbf{Y}}\|^2$, so the distortion never rises.
 *
 * @param x - Input data matrix ($n \times d$).
 * @param options - OPQ training options.
 * @param options.subspaces - Number of sub-spaces $M$.
 * @param options.codewords - Number of codewords per sub-space $K$.
 * @param options.stream - Random stream.
 * @param options.iterations - Lloyd iterations per sub-space (default 25).
 * @param options.rounds - Alternation rounds (default 10).
 * @returns OPQ result with optimized rotation and codebooks.
 *
 * @example Train an optimised product quantiser
 * const s = stream(42)
 * const data = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
 * const opq = optimisedProductQuantiser(data, { subspaces: 2, codewords: 2, rounds: 2, stream: s })
 * print('OPQ distortion:', opq.distortion)
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

/**
 * The PQ codes of the rows of $x$: $[n, M]$ int32 sub-codeword indices.
 *
 * @param pq - Product quantiser instance.
 * @param x - Input data matrix ($n \times d$).
 * @returns Tensor of quantized codes of shape $[n, M]$.
 *
 * @example Encode vectors into PQ codes
 * const s = stream(42)
 * const data = [[0, 0, 0, 0], [1, 1, 1, 1]]
 * const pq = productQuantiser(data, { subspaces: 2, codewords: 2, stream: s })
 * const codes = pqEncode(pq, data)
 * print('Codes:\n' + codes)
 */
export function pqEncode(pq: ProductQuantiser, x: MatrixLike): Tensor {
  const X = rowsOf(x, 'pqEncode')
  if (X.d !== pq.d) throw new ShapeError('pqEncode', `pqEncode: the quantiser codes width ${pq.d}, data ${X.d}`)
  const Y = rotate(X.data, X.n, X.d, pq.rotation)
  return fromData(encodeRows(Y, X.n, X.d, pq.subspaces, pq.codewords, dense.data(pq.codebooks)), [X.n, pq.subspaces])
}

/**
 * The reconstructions of codes $[n, M]$, mapped back by $\mathbf{R}^\top$ when the quantiser has a rotation: $[n, d]$.
 *
 * @param pq - Product quantiser instance.
 * @param codes - Quantized codes tensor $[n, M]$.
 * @returns Reconstructed vectors tensor $[n, d]$.
 *
 * @example Decode PQ codes into reconstructed vectors
 * const s = stream(42)
 * const data = [[0, 0, 0, 0], [1, 1, 1, 1]]
 * const pq = productQuantiser(data, { subspaces: 2, codewords: 2, stream: s })
 * const codes = pqEncode(pq, data)
 * const recon = pqDecode(pq, codes)
 * print('Reconstructed shape:', recon.shape)
 */
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

/**
 * The ADC table of a query: its squared distances to every sub-codeword $[M, K]$ (after the rotation, if any).
 *
 * @param pq - Product quantiser instance.
 * @param query - Query vector of length $d$.
 * @returns Distance lookup table of shape $[M, K]$.
 *
 * @example Compute asymmetric distance table
 * const s = stream(42)
 * const data = [[0, 0, 0, 0], [1, 1, 1, 1]]
 * const pq = productQuantiser(data, { subspaces: 2, codewords: 2, stream: s })
 * const table = pqDistanceTable(pq, [0, 0, 0, 0])
 * print('Table shape:', table.shape)
 */
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
 * The $k$ nearest coded points to the query by asymmetric distance: $\hat{d}^2 = \sum_m T[m, \mathrm{code}_m]$ from the query's table, so a
 * point costs $M$ lookups instead of $d$ multiplications. Distances are the estimates $\hat{d}$ (not squared).
 *
 * @param pq - Product quantiser instance.
 * @param codes - Dataset codes tensor of shape $[n, M]$.
 * @param query - Query vector of length $d$.
 * @param k - Number of nearest neighbours $k$ to return.
 * @returns Query result containing $k$ nearest indices and estimated distances.
 *
 * @example Query nearest neighbours using PQ asymmetric distances
 * const s = stream(42)
 * const data = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
 * const pq = productQuantiser(data, { subspaces: 2, codewords: 2, stream: s })
 * const codes = pqEncode(pq, data)
 * const res = pqQuery(pq, codes, [0.1, 0.1, 0.1, 0.1], 2)
 * print('Nearest index:', res.indices[0])
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

/**
 * Search PQ-encoded dataset across multiple queries using asymmetric distance computation.
 *
 * @param pq - Product quantiser instance.
 * @param codes - Encoded dataset tensor $[n, M]$.
 * @param queries - Query points matrix ($m \times d$).
 * @param k - Number of nearest neighbours $k$ to return per query.
 * @returns Stacked `Neighbours` object.
 *
 * @example Search PQ-encoded dataset for multiple queries
 * const s = stream(42)
 * const data = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]
 * const queries = [[0.1, 0.1, 0.1, 0.1]]
 * const pq = productQuantiser(data, { subspaces: 2, codewords: 2, stream: s })
 * const codes = pqEncode(pq, data)
 * const res = pqSearch(pq, codes, queries, 2)
 * print('Nearest indices:\n' + res.indices)
 */
export function pqSearch(pq: ProductQuantiser, codes: Tensor, queries: MatrixLike, k: Size): Neighbours {
  const Q = rowsOf(queries, 'pqSearch')
  return stackResults(
    Array.from({ length: Q.n }, (_, i) => pqQuery(pq, codes, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k)),
    k,
  )
}
