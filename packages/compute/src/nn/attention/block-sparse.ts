/**
 * Query-dependent block selection for trainable sparse attention. The keys are cut into consecutive blocks of $B$
 * tokens. Each block gets a summary key, here the mean of its keys (MoBA's mean pooling; NSA learns the summary with a
 * small network and scores it by attention). Each query scores every visible block by the dot product of the query
 * with the summary, keeps the $k$ best, always keeps its own block (and optionally the first), and then attends with an
 * ordinary softmax over the tokens of the kept blocks. `blockSparseMask` turns the choices into a token-level mask
 * (usable as `scaledDotProductAttention`'s `mask`), and `capturedWeight` measures how much of the full softmax weight
 * of each query the kept blocks hold.
 *
 * Sources: Lu et al. (2025), "MoBA: Mixture of Block Attention for Long-Context LLMs", §2.2 (gating by the inner
 * product with mean-pooled block keys, top-$k$, the current block always selected with a causal mask, future blocks
 * excluded); Yuan et al. (2025), "Native Sparse Attention", §3.3 (block importance from the attention scores of
 * compressed keys, top-$n$ blocks with fixed initial and local blocks). One head; mean pooling for the summary in place
 * of NSA's learned compression. Scores and masks are tensors; the kept blocks are index lists, one per query.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { dense, fromData, toRows, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'

/** Options of `blockScores`. */
export type BlockScoreOptions = {
  /** Exclude blocks that start after the query (default true). */
  causal?: boolean
  /** The scale of the scores (default $1/\sqrt{d}$). */
  scale?: number
  /**
   * Return softmax probabilities over the visible blocks (NSA's importance scores from attention to compressed keys)
   * instead of raw scores (MoBA's gate scores). The ranking is the same (default false).
   */
  normalise?: boolean
}

/**
 * The score of every query against every block of keys, $s_{ib} = \text{scale} \cdot \qvec_i^\top \bar{\kvec}_b$
 * with $\bar{\kvec}_b$ the mean of the keys of block $b$. Blocks that start after query $i$ get $-\infty$ when causal
 * (0 when normalised).
 *
 * @param q The queries, $n \times d$ (a tensor or rows of numbers).
 * @param k The keys, $n \times d$.
 * @param blockSize The block length $B \ge 1$; the last block may be shorter.
 * @param options Causality, scale and normalisation.
 * @returns The scores, $n \times \lceil n/B \rceil$, row $i$ for query $i$.
 *
 * @example Eight keys in blocks of four: query 3 cannot see block 1
 * const k = tensor([[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [-1, 0], [0, -1], [0, -1]])
 * print('scores =', blockScores(k, k, 4, { scale: 1 }))
 */
export function blockScores(q: MatrixLike, k: MatrixLike, blockSize: Size, options: BlockScoreOptions = {}): Tensor {
  const K = dense.toMatrixF64(k, 'blockScores')
  const Q = dense.toMatrixF64(q, 'blockScores', undefined, K.n)
  const n = K.m
  const d = K.n
  const { causal = true, scale = 1 / Math.sqrt(d), normalise = false } = options
  const nb = Math.ceil(n / blockSize)
  const summary = new Float64Array(nb * d)
  for (let b = 0; b < nb; b++) {
    const lo = b * blockSize
    const hi = Math.min(n, lo + blockSize)
    for (let j = lo; j < hi; j++) for (let t = 0; t < d; t++) summary[b * d + t] += K.data[j * d + t] / (hi - lo)
  }
  const out = new Float64Array(Q.m * nb)
  for (let i = 0; i < Q.m; i++) {
    const row = out.subarray(i * nb, (i + 1) * nb)
    for (let b = 0; b < nb; b++) {
      if (causal && b * blockSize > i) {
        row[b] = Number.NEGATIVE_INFINITY
        continue
      }
      let dot = 0
      for (let t = 0; t < d; t++) dot += Q.data[i * d + t] * summary[b * d + t]
      row[b] = scale * dot
    }
    if (!normalise) continue
    const top = Math.max(...row)
    let z = 0
    for (let b = 0; b < nb; b++) z += row[b] = Number.isFinite(row[b]) ? Math.exp(row[b] - top) : 0
    for (let b = 0; b < nb; b++) row[b] /= z
  }
  return fromData(out, [Q.m, nb])
}

/** Options of `selectBlocks`. */
export type BlockSelectOptions = {
  /** How many blocks each query chooses by score, besides the fixed ones (default 2). */
  topK?: Size
  /** Always keep the query's own block (MoBA, NSA's local blocks; default true). */
  current?: boolean
  /** Always keep the first block (NSA's initial block, an attention sink; default false). */
  first?: boolean
  /** Never choose a block that starts after the query (default true). */
  causal?: boolean
}

/**
 * The blocks each query keeps: the fixed ones (its own block, the first) and the `topK` highest-scoring others among
 * the visible blocks (finite scores; ties go to the earlier block).
 *
 * @param scores The $n \times n_b$ block scores from `blockScores` (a tensor or rows of numbers).
 * @param blockSize The block length $B$, to find each query's own block.
 * @param options How many to choose and which to keep always.
 * @returns For each query, the kept block indices in increasing order.
 *
 * @example The best block besides its own, for each of eight queries
 * const k = tensor([[1, 0], [1, 0], [0, 1], [0, 1], [-1, 0], [-1, 0], [0, -1], [0, -1]])
 * print('kept =', selectBlocks(blockScores(k, k, 2, { scale: 1 }), 2, { topK: 1 }))
 */
export function selectBlocks(scores: MatrixLike, blockSize: Size, options: BlockSelectOptions = {}): number[][] {
  const { topK = 2, current = true, first = false, causal = true } = options
  const rows = Array.isArray(scores) ? (scores as readonly (readonly number[])[]) : toRows(scores as Tensor)
  return rows.map((row, i) => {
    const own = Math.floor(i / blockSize)
    const kept = new Set<number>()
    if (current) kept.add(own)
    if (first) kept.add(0)
    const ranked = Array.from(row, (s, b) => ({ s, b }))
      .filter(({ s, b }) => Number.isFinite(s) && !kept.has(b) && (!causal || b * blockSize <= i))
      .sort((x, y) => y.s - x.s || x.b - y.b)
    for (const { b } of ranked.slice(0, topK)) kept.add(b)
    return [...kept].sort((a, b) => a - b)
  })
}

/**
 * The token-level mask of a block selection: query $i$ scores key $j$ when $j$'s block is kept for $i$ (and
 * $j \le i$ when causal).
 *
 * @param n The sequence length $n$.
 * @param blockSize The block length $B$.
 * @param selected The kept blocks of each query, from `selectBlocks`.
 * @param causal Hide keys after the query (default true).
 * @returns The $n \times n$ mask of ones and zeros (one row per entry of `selected`).
 *
 * @example Blocks of two; query 5 keeps blocks 0 and 2
 * print('mask =', blockSparseMask(6, 2, [[0], [0], [1], [1], [2], [0, 2]]))
 */
export function blockSparseMask(
  n: Size,
  blockSize: Size,
  selected: readonly (readonly number[])[],
  causal = true,
): Tensor {
  const out = new Float64Array(selected.length * n)
  selected.forEach((blocks, i) => {
    for (const b of blocks)
      for (let j = b * blockSize; j < Math.min(n, (b + 1) * blockSize); j++) if (!causal || j <= i) out[i * n + j] = 1
  })
  return fromData(out, [selected.length, n])
}

/**
 * The share of each query's attention weight that falls inside a mask, $\sum_j W_{ij} [M_{ij} \ne 0]$. With the full
 * softmax weights it measures how much a sparse pattern keeps.
 *
 * @param weights The $n \times m$ attention weights, rows summing to 1 (a tensor or rows of numbers).
 * @param mask An $n \times m$ mask of ones and zeros.
 * @returns The $n$ shares, in $[0, 1]$.
 *
 * @example Half of a uniform row falls in a mask of two keys out of four
 * print('captured =', capturedWeight([[0.25, 0.25, 0.25, 0.25]], [[1, 1, 0, 0]]))
 */
export function capturedWeight(weights: MatrixLike, mask: MatrixLike): Tensor {
  const W = dense.toMatrixF64(weights, 'capturedWeight')
  const M = dense.toMatrixF64(mask, 'capturedWeight', W.m, W.n)
  const out = new Float64Array(W.m)
  for (let i = 0; i < W.m; i++) {
    let s = 0
    for (let j = 0; j < W.n; j++) s += M.data[i * W.n + j] ? W.data[i * W.n + j] : 0
    out[i] = s
  }
  return fromData(out, [W.m])
}
