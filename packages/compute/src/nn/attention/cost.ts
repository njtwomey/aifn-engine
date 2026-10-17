/**
 * Leading-order cost of one attention head on one sequence of $n$ tokens with head width $d$, forward pass, counted in
 * multiply–adds, and the number of attention scores (or running statistics) held at once. Methods: full softmax
 * attention, FlashAttention (same arithmetic, two running statistics per query instead of the score matrix), a window
 * of $w$ keys per query, block selection (score $\lceil n/B \rceil$ block summaries, then attend to the current block
 * and $k$ chosen blocks of $B$ keys), linear attention with $m$ features, Linformer with keys and values projected to
 * $k$ rows, and Reformer's LSH attention ($b$ buckets, $r$ hashing rounds, each query attending to its own chunk and
 * the previous one, chunks of $n/b$ tokens). `layerLinearCost` gives the multiply–adds of the projections and
 * feed-forward network of a layer, which attention is compared against. Costs ignore softmax exponentials, sorting and
 * causal halving.
 *
 * Sources: Vaswani et al. (2017), §4 and Table 1 (per-layer complexity); Dao et al. (2022), "FlashAttention", §3
 * (memory $O(n)$); Wang et al. (2020), "Linformer", §4 ($O(nk)$); Katharopoulos et al. (2020), §3.2 ($O(nmd)$);
 * Kitaev, Kaiser and Levskaya (2020), "Reformer", §2 (chunked LSH attention); Lu et al. (2025), "MoBA", §2.2.
 * Multiply–adds are counted as `convolutionCost` of `aifn-compute/nn/functional` counts them.
 */

import type { Size } from 'aifn-compute/foundation/contracts'

/** The attention whose cost `attentionCost` counts. */
export type AttentionMethod = 'full' | 'flash' | 'window' | 'block' | 'linear' | 'linformer' | 'lsh'

/** The sizes each method needs; unused ones are ignored. */
export type AttentionCostOptions = {
  /** The head width $d$ (default 64). */
  headWidth?: Size
  /** The keys per query $w$ of a window (default 512). */
  window?: Size
  /** The block length $B$ of block selection (default 64). */
  blockSize?: Size
  /** The blocks $k$ chosen per query besides its own (default 8). */
  topK?: Size
  /** The feature dimension $m$ of linear attention (default $d$). */
  features?: Size
  /** The projected length $k$ of Linformer (default 256). */
  rank?: Size
  /** The buckets $b$ of LSH attention (default 64; at least 2). */
  buckets?: Size
  /** The hashing rounds $r$ of LSH attention (default 4). */
  rounds?: Size
}

/** The multiply–adds of one head, and the scores or running statistics held at once. */
export type AttentionCost = {
  /** The multiply–adds of the forward pass. */
  multiplyAdds: number
  /** The number of scores (or running statistics, or state entries) held at once. */
  scoreMemory: number
}

/**
 * The leading-order cost of one attention head over $n$ tokens: full attention $2n^2d$ multiply–adds and $n^2$
 * scores, FlashAttention the same arithmetic with $2n$ statistics, a window $2nwd$, and so on (see the file comment).
 *
 * @param method Which attention.
 * @param n The sequence length $n \ge 1$.
 * @param options The head width and the method's sizes.
 * @returns The multiply–adds and the score memory (the number of stored scores or statistics).
 *
 * @example Full attention against a window and linear attention at 8192 tokens
 * for (const method of ['full', 'flash', 'window', 'linear']) print(method, attentionCost(method, 8192))
 */
export function attentionCost(method: AttentionMethod, n: Size, options: AttentionCostOptions = {}): AttentionCost {
  const d = options.headWidth ?? 64
  switch (method) {
    case 'full':
      return { multiplyAdds: 2 * n * n * d, scoreMemory: n * n }
    case 'flash':
      return { multiplyAdds: 2 * n * n * d, scoreMemory: 2 * n }
    case 'window': {
      const w = Math.min(n, options.window ?? 512)
      return { multiplyAdds: 2 * n * w * d, scoreMemory: n * w }
    }
    case 'block': {
      const B = options.blockSize ?? 64
      const blocks = Math.ceil(n / B)
      const kept = Math.min(blocks, (options.topK ?? 8) + 1)
      const keys = Math.min(n, kept * B)
      return { multiplyAdds: n * d + n * blocks * d + 2 * n * keys * d, scoreMemory: n * blocks + n * keys }
    }
    case 'linear': {
      const m = options.features ?? d
      return { multiplyAdds: 2 * n * m * d + 2 * n * m, scoreMemory: m * d + m }
    }
    case 'linformer': {
      const k = Math.min(n, options.rank ?? 256)
      return { multiplyAdds: 4 * n * k * d, scoreMemory: n * k }
    }
    case 'lsh': {
      const b = Math.max(2, options.buckets ?? 64)
      const r = options.rounds ?? 4
      const keys = Math.min(n, 2 * Math.ceil(n / b))
      return { multiplyAdds: r * (n * d * (b / 2) + 2 * n * keys * d), scoreMemory: r * n * keys }
    }
  }
}

/**
 * The multiply–adds of the parts of a transformer layer that act on each token separately: the query, key, value and
 * output projections ($4nd^2$) and a feed-forward network of hidden width $\rho d$ ($2\rho nd^2$).
 *
 * @param n The sequence length $n$.
 * @param dModel The layer width $d$.
 * @param ffnRatio The feed-forward expansion $\rho$ (default 4).
 * @returns The multiply–adds, $(4 + 2\rho)nd^2$.
 *
 * @example Attention over all heads overtakes the rest of the layer at n = 6 d
 * const d = 1024
 * print('attention:', (d / 64) * attentionCost('full', 6 * d).multiplyAdds)
 * print('projections and feed-forward:', layerLinearCost(6 * d, d))
 */
export function layerLinearCost(n: number, dModel: number, ffnRatio = 4): number {
  return (4 + 2 * ffnRatio) * n * dModel * dModel
}
