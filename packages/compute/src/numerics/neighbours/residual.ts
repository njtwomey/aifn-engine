/**
 * Residual vector quantisation (RVQ; Juang and Gray, 1982; Chen, Guan and Wang, 2010, "Approximate nearest neighbor
 * search by residual vector quantization", Sensors 10(12)): $D$ codebooks applied in turn, each quantising what the
 * ones before it left over.
 *
 * A vector $\xvec$ is coded level by level: $\rvec_0 = \xvec$, $k_\ell$ is the codeword of codebook $\ell$ nearest to
 * $\rvec_{\ell - 1}$, and $\rvec_\ell = \rvec_{\ell - 1} - \cvec^{(\ell)}_{k_\ell}$. The reconstruction is the sum of
 * the chosen codewords, $\hat\xvec = \sum_{\ell=1}^D \cvec^{(\ell)}_{k_\ell}$, so a code of $D$ indices into codebooks of
 * $K$ codewords costs $D \log_2 K$ bits and every prefix of it is a coarser reconstruction. Unlike product quantisation,
 * each level sees the whole vector. The codebooks are trained greedily, each by $k$-means (`trainCodebook`) on the
 * residuals the previous levels leave, as Chen et al. do; the encoding is greedy too (no beam search).
 *
 * This is the quantiser inside the RQ-VAE (`aifn-methods/generative/autoencoders`, kind `'rqvae'`), and the code of
 * $D$ indices is what generative retrieval calls a semantic ID.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { assignNearest, trainCodebook } from './codebook'
import { rowsOf } from './search'

/** A residual vector quantiser: $D$ codebooks of $K$ codewords in $\reals^d$. */
export interface ResidualQuantiser {
  /** Discriminator kind. */
  readonly kind: 'residual-quantiser'
  /** Vector dimensionality $d$. */
  readonly d: Size
  /** The number of levels $D$. */
  readonly levels: Size
  /** Codewords per level $K$. */
  readonly codewords: Size
  /** The codebooks $[D, K, d]$: codebook $\ell$ is `codebooks[ℓ]`. */
  readonly codebooks: Tensor
  /** The mean squared reconstruction error $\lVert \xvec - \hat\xvec \rVert^2$ over the training rows, after all levels. */
  readonly distortion: number
  /** The mean squared error after each level $1, \dots, D$: it never increases from one level to the next. */
  readonly distortionByLevel: Float64Array
}

/** Options of `residualQuantiser`. */
export interface ResidualQuantiserOptions {
  /** The number of levels $D \ge 1$. */
  levels: Size
  /** Codewords per level $K \ge 1$. */
  codewords: Size
  /** Random stream: level $\ell$ seeds its $k$-means from `child(stream, 'level', ℓ)`. */
  stream: Stream
  /** Lloyd iterations per level (default 25). */
  iterations?: number
}

/**
 * Subtract each row's nearest codeword from it, in place, and report the codes.
 *
 * @param residual The residuals $[n, d]$, row-major; overwritten with what the codebook leaves.
 * @param n The number of rows.
 * @param d The dimensionality.
 * @param codebook The codebook $[K, d]$, row-major.
 * @param K The number of codewords.
 * @returns The index of the codeword chosen for each row.
 */
function quantiseLevel(residual: Float64Array, n: Size, d: Size, codebook: Float64Array, K: Size): Int32Array {
  const labels = dense.data(assignNearest(fromData(residual, [n, d]), fromData(codebook, [K, d])).labels)
  const codes = Int32Array.from(labels)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) residual[i * d + j] -= codebook[codes[i] * d + j]
  return codes
}

/**
 * Train a residual vector quantiser on the rows of $x$: level $\ell$'s codebook is $k$-means of the residuals the
 * levels before it leave (Chen, Guan and Wang, 2010). Rows that a level reconstructs exactly leave zero residuals, so a
 * deeper level may waste codewords on them; with fewer distinct residuals than $K$, `trainCodebook` gives every one its
 * own codeword.
 *
 * @param x The training vectors, $n \times d$.
 * @param options The number of levels and codewords, the stream and the Lloyd iterations.
 * @returns The quantiser, with its distortion after each level.
 *
 * @example Each level shrinks the error
 * const s = stream(1)
 * const data = normal(s, 0, 1, { shape: [400, 4] })
 * const rq = residualQuantiser(data, { levels: 4, codewords: 8, stream: s })
 * print('mean squared error after each level:', rq.distortionByLevel)
 */
export function residualQuantiser(x: MatrixLike, options: ResidualQuantiserOptions): ResidualQuantiser {
  const { levels: D, codewords: K, stream, iterations } = options
  const X = rowsOf(x, 'residualQuantiser')
  if (!(Number.isInteger(D) && D >= 1))
    throw new DomainError('residualQuantiser', `residualQuantiser: levels must be a positive integer, got ${D}`)
  if (!(Number.isInteger(K) && K >= 1))
    throw new DomainError('residualQuantiser', `residualQuantiser: codewords must be a positive integer, got ${K}`)
  const { n, d } = X
  const residual = Float64Array.from(X.data)
  const books = new Float64Array(D * K * d)
  const distortionByLevel = new Float64Array(D)
  for (let l = 0; l < D; l++) {
    const trained = trainCodebook(fromData(Float64Array.from(residual), [n, d]), K, {
      stream: child(stream, 'level', l),
      iterations,
    })
    // With fewer rows than K, trainCodebook returns n codewords; the rest stay at the origin, which no row prefers.
    const centroids = dense.data(trained.centroids)
    const codebook = new Float64Array(K * d)
    codebook.set(centroids.subarray(0, Math.min(centroids.length, K * d)))
    books.set(codebook, l * K * d)
    quantiseLevel(residual, n, d, codebook, K)
    let err = 0
    for (const v of residual) err += v * v
    distortionByLevel[l] = err / n
  }
  return {
    kind: 'residual-quantiser',
    d,
    levels: D,
    codewords: K,
    codebooks: fromData(books, [D, K, d]),
    distortion: distortionByLevel[D - 1],
    distortionByLevel,
  }
}

/**
 * The codes of the rows of $x$: level by level, the index of the codeword nearest to what the levels before left.
 * The coding is greedy, so a row's error can rise from one level to the next when no codeword of a level is nearer to
 * its residual than the origin is; averaged over the training rows it never rises, since each codebook is $k$-means of
 * those residuals.
 *
 * @param rq The quantiser.
 * @param x The vectors, $n \times d$.
 * @returns The codes $[n, D]$ (int32): row $i$ is the index of its codeword at each level.
 *
 * @example A point's code, and its reconstruction level by level
 * // The mean error over the training rows falls at every level; one point's error usually does, but need not.
 * const s = stream(2)
 * const data = normal(s, 0, 1, { shape: [300, 2] })
 * const rq = residualQuantiser(data, { levels: 3, codewords: 4, stream: s })
 * const point = [0.8, -1.3]
 * const codes = rqEncode(rq, [point])
 * print('codes:', codes)
 * for (const l of [1, 2, 3]) {
 *   const r = toArray(rqDecode(rq, codes, l))[0]
 *   print(`first ${l} level(s):`, r, ' squared error', (r[0] - point[0]) ** 2 + (r[1] - point[1]) ** 2)
 * }
 */
export function rqEncode(rq: ResidualQuantiser, x: MatrixLike): Tensor {
  const X = rowsOf(x, 'rqEncode')
  if (X.d !== rq.d) throw new ShapeError('rqEncode', `rqEncode: the quantiser codes width ${rq.d}, data ${X.d}`)
  const { n, d } = X
  const { levels: D, codewords: K } = rq
  const residual = Float64Array.from(X.data)
  const books = dense.data(rq.codebooks)
  const codes = new Int32Array(n * D)
  for (let l = 0; l < D; l++) {
    const level = quantiseLevel(residual, n, d, books.subarray(l * K * d, (l + 1) * K * d), K)
    for (let i = 0; i < n; i++) codes[i * D + l] = level[i]
  }
  return fromData(codes, [n, D])
}

/**
 * The reconstructions of codes: the sum of each row's codewords over the first `levels` levels, a coarser
 * reconstruction for fewer levels.
 *
 * @param rq The quantiser.
 * @param codes The codes $[n, D]$, as `rqEncode` gives them.
 * @param levels How many leading levels to sum, from 0 to $D$ (default $D$; 0 gives zeros).
 * @returns The reconstructions $[n, d]$.
 *
 * @example The reconstruction of a code by hand
 * const s = stream(3)
 * const rq = residualQuantiser(normal(s, 0, 1, { shape: [200, 2] }), { levels: 2, codewords: 4, stream: s })
 * const books = toArray(rq.codebooks)
 * print('codeword 1 of level 1 plus codeword 2 of level 2:', books[0][1].map((v, j) => v + books[1][2][j]))
 * print('rqDecode:', rqDecode(rq, [[1, 2]]))
 */
export function rqDecode(rq: ResidualQuantiser, codes: Tensor | MatrixLike, levels: Size = rq.levels): Tensor {
  const C = rowsOf(codes, 'rqDecode')
  const { levels: D, codewords: K, d } = rq
  if (C.d !== D) throw new ShapeError('rqDecode', `rqDecode: codes have ${C.d} levels, the quantiser ${D}`)
  if (!(Number.isInteger(levels) && levels >= 0 && levels <= D))
    throw new DomainError('rqDecode', `rqDecode: levels must be an integer from 0 to ${D}, got ${levels}`)
  const books = dense.data(rq.codebooks)
  const out = new Float64Array(C.n * d)
  for (let i = 0; i < C.n; i++)
    for (let l = 0; l < levels; l++) {
      const k = C.data[i * D + l]
      if (!(Number.isInteger(k) && k >= 0 && k < K))
        throw new DomainError('rqDecode', `rqDecode: code ${k} at level ${l} is not a codeword 0 … ${K - 1}`)
      for (let j = 0; j < d; j++) out[i * d + j] += books[(l * K + k) * d + j]
    }
  return fromData(out, [C.n, d])
}

/** A node of a code prefix tree: the rows whose codes begin with `prefix`. */
export interface CodeTreeNode {
  /** The code prefix, one index per level; empty at the root. */
  readonly prefix: readonly number[]
  /** The prefix's length: 0 at the root, $D$ at a leaf. */
  readonly depth: Size
  /** How many rows have this prefix. */
  readonly count: Size
  /** How many rows come before this node's rows in the codes' lexicographic order: its rows are `[offset, offset + count)`. */
  readonly offset: Size
  /** The parent's index in `nodes` (−1 at the root). */
  readonly parent: number
  /** The children's indices in `nodes`, in order of their last code. */
  readonly children: readonly number[]
}

/**
 * The tree of code prefixes of residual (or any multi-level) codes: the root holds every row, and a node at depth $d$
 * holds the rows whose first $d$ codes are its prefix, so each level refines the one above it into at most $K$ parts.
 * Only prefixes some row has are nodes. Nodes come breadth first, children in order of their code, and each has the
 * span `[offset, offset + count)` of its rows in lexicographic order, so an icicle or sunburst plot draws it directly.
 *
 * @param codes The codes $[n, D]$, non-negative integers, as `rqEncode` gives them.
 * @returns The nodes, the root first.
 *
 * @example Six codes of two levels as a tree
 * const tree = codePrefixTree([[0, 1], [0, 1], [0, 2], [1, 0], [1, 0], [1, 0]])
 * for (const node of tree) print(`prefix [${node.prefix}] holds ${node.count} rows from offset ${node.offset}`)
 */
export function codePrefixTree(codes: Tensor | MatrixLike): CodeTreeNode[] {
  const C = rowsOf(codes, 'codePrefixTree')
  const { n, d: D } = C
  for (const v of C.data)
    if (!(Number.isInteger(v) && v >= 0))
      throw new DomainError('codePrefixTree', `codePrefixTree: code ${v} is not a non-negative integer`)
  // Sort rows lexicographically by code, then walk the levels: a node is a run of equal prefixes.
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => {
    for (let l = 0; l < D; l++) {
      const diff = C.data[a * D + l] - C.data[b * D + l]
      if (diff) return diff
    }
    return a - b
  })
  type Mutable = { prefix: number[]; depth: number; count: number; offset: number; parent: number; children: number[] }
  const nodes: Mutable[] = [{ prefix: [], depth: 0, count: n, offset: 0, parent: -1, children: [] }]
  let frontier = [0]
  for (let l = 0; l < D; l++) {
    const next: number[] = []
    for (const p of frontier) {
      const parent = nodes[p]
      for (let at = parent.offset; at < parent.offset + parent.count;) {
        const code = C.data[order[at] * D + l]
        let end = at
        while (end < parent.offset + parent.count && C.data[order[end] * D + l] === code) end++
        nodes.push({
          prefix: [...parent.prefix, code],
          depth: l + 1,
          count: end - at,
          offset: at,
          parent: p,
          children: [],
        })
        parent.children.push(nodes.length - 1)
        next.push(nodes.length - 1)
        at = end
      }
    }
    frontier = next
  }
  return nodes
}
