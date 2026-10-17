/**
 * Fixed sparse attention patterns as query-by-key masks, and the reach of a stack of masked layers. A mask is an
 * $n \times n$ tensor of ones (query $i$ scores key $j$) and zeros, usable as `scaledDotProductAttention`'s `mask`.
 * The patterns are a symmetric sliding window of half-width $w$ (Longformer's local attention), a dilated window
 * (every $\delta$-th key out to $w\delta$), the strided pattern of the Sparse Transformer (the previous $l$ keys and
 * every $l$-th key), local attention plus $g$ global tokens, and BigBird (window, global tokens and $r$ random keys per
 * query). Every pattern can be made causal. `attentionReach` composes a mask with itself over $L$ layers, with the
 * residual path (each position keeps its own information), to give which inputs can influence which outputs.
 *
 * Note the window's meaning: here `window` is the half-width $w$, keys with $\lvert i - j \rvert \le w$ (a band of
 * $2w + 1$), whereas `positionMask`'s `window` hides keys $w$ or more positions away (a band of $2w - 1$). The sliding
 * pattern of half-width $w$ is `positionMask` with `window` $w + 1$.
 *
 * Sources: Child et al. (2019), "Generating Long Sequences with Sparse Transformers", §4.2 (the strided pattern);
 * Beltagy, Peters and Cohan (2020), "Longformer", §3.1 (sliding, dilated and global attention); Zaheer et al. (2020),
 * "Big Bird", §2 (random, window and global). Masks are token-level here, where BigBird applies its pattern to blocks
 * of tokens.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, type MatrixLike, type Tensor } from 'aifn-compute/foundation/tensor'

/** A sparse attention pattern: `full`, `sliding`, `dilated`, `strided`, `global` or `bigbird`. */
export type SparsePattern = 'full' | 'sliding' | 'dilated' | 'strided' | 'global' | 'bigbird'

/** Options of `sparseAttentionMask`. Fields a pattern does not use are ignored. */
export type SparseMaskOptions = {
  /**
   * The half-width $w$ of the local window: keys with $\lvert i - j \rvert \le w$ (default 2). Not `positionMask`'s
   * `window`, which counts $w + 1$ for the same band.
   */
  window?: Size
  /**
   * The dilation $\delta$ of the dilated window: keys at $\lvert i - j \rvert \in \{0, \delta, \dots, w\delta\}$
   * (default 2).
   */
  dilation?: Size
  /** The stride $l$ of the strided pattern: keys with $\lvert i - j \rvert < l$ or a multiple of $l$ (default 4). */
  stride?: Size
  /** The number $g$ of global tokens, the first $g$ positions (default 0). */
  globals?: Size
  /**
   * The number $r$ of random keys per query (BigBird; default 0). They are drawn with replacement, so a query may get
   * fewer than $r$ distinct ones.
   */
  random?: Size
  /** Hide keys after the query (default false). */
  causal?: boolean
  /** The stream for BigBird's random keys (required when `random` is positive). */
  stream?: Stream
}

/**
 * The $n \times n$ mask of a sparse attention pattern: entry $(i, j)$ is 1 when query $i$ scores key $j$, else 0. The
 * diagonal is always 1, so every query sees at least itself. BigBird's random keys are drawn query by query, one
 * uniform per key, among the keys the query may see. Throws `DomainError` when BigBird needs random keys and no
 * stream is given.
 *
 * @param n The sequence length $n$.
 * @param pattern Which pattern.
 * @param options The window, dilation, stride, global and random counts, causality and stream.
 * @returns The mask, $n \times n$ (row $i$ for query $i$), of ones and zeros.
 *
 * @example A sliding window of half-width 1, and the causal strided pattern
 * print('sliding:', sparseAttentionMask(6, 'sliding', { window: 1 }))
 * print('strided, causal:', sparseAttentionMask(6, 'strided', { stride: 2, causal: true }))
 *
 * @example BigBird: a window, one global token and a random key per query
 * print('bigbird:', sparseAttentionMask(8, 'bigbird', { window: 1, globals: 1, random: 1, stream: stream(1) }))
 */
export function sparseAttentionMask(n: Size, pattern: SparsePattern, options: SparseMaskOptions = {}): Tensor {
  const { window: w = 2, dilation = 2, stride = 4, globals = 0, random = 0, causal = false } = options
  const m = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const d = Math.abs(i - j)
      const local = d <= w
      const global = i < globals || j < globals
      let on: boolean
      if (pattern === 'full') on = true
      else if (pattern === 'sliding') on = local
      else if (pattern === 'dilated') on = d <= w * dilation && d % dilation === 0
      else if (pattern === 'strided') on = d < stride || d % stride === 0
      else on = local || global
      m[i * n + j] = on ? 1 : 0
    }
  if (pattern === 'bigbird' && random > 0) {
    if (!options.stream)
      throw new DomainError('sparseAttentionMask', 'sparseAttentionMask: BigBird with random keys needs a stream')
    for (let i = 0; i < n; i++) {
      const hi = causal ? i + 1 : n
      for (let k = 0; k < random; k++) m[i * n + Math.min(hi - 1, Math.floor(uniform(options.stream) * hi))] = 1
    }
  }
  if (causal) for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m[i * n + j] = 0
  for (let i = 0; i < n; i++) m[i * n + i] = 1
  return fromData(m, [n, n])
}

/**
 * The mask as rows of 0 or 1 flags.
 *
 * @param mask A square mask, a tensor or rows of numbers (any non-zero entry counts as visible).
 * @param where The caller's name, for error messages.
 * @returns The flags, row-major, with the side $n$.
 */
function flags(mask: MatrixLike, where: string): { on: Uint8Array; n: number } {
  const { data, m, n } = dense.toMatrixF64(mask, where)
  if (m !== n) throw new DomainError(where, `${where}: the mask must be square (got ${m} x ${n})`)
  return { on: Uint8Array.from(data, (v) => (v ? 1 : 0)), n }
}

/**
 * The number of scored pairs of a mask (its non-zero entries), and the mean number of keys per query.
 *
 * @param mask The $n \times n$ mask, a tensor or rows of numbers.
 * @returns `pairs`, the count of visible (query, key) pairs, and `keysPerQuery`, that count over $n$.
 *
 * @example A window of half-width 2 over 16 tokens scores 16 times 5, less 6 at the edges
 * print('density =', maskDensity(sparseAttentionMask(16, 'sliding', { window: 2 })))
 */
export function maskDensity(mask: MatrixLike): { pairs: number; keysPerQuery: number } {
  const { on, n } = flags(mask, 'maskDensity')
  let pairs = 0
  for (const v of on) pairs += v
  return { pairs, keysPerQuery: pairs / Math.max(1, n) }
}

/**
 * Which inputs can reach which outputs through `layers` stacked layers that all use `mask`, with a residual path:
 * $\Rmat_1 = \Mmat \lor \Imat$ and $\Rmat_L = \Rmat_{L-1}\Mmat$ (a boolean product), so entry $(i, j)$ is 1 when input
 * $j$ can influence output $i$. Costs $O(n \cdot \mathrm{nnz})$ per layer.
 *
 * @param mask The $n \times n$ mask of one layer, a tensor or rows of numbers.
 * @param layers The number $L \ge 1$ of layers.
 * @returns The $n \times n$ reach matrix of ones and zeros, row $i$ for output $i$.
 *
 * @example A causal window of the current token and two before: after three layers position 10 reaches 4 to 10
 * const reach = attentionReach(sparseAttentionMask(16, 'sliding', { window: 2, causal: true }), 3)
 * print('row 10 =', slice(reach, 10))
 */
export function attentionReach(mask: MatrixLike, layers: Size): Tensor {
  const { on, n } = flags(mask, 'attentionReach')
  for (let i = 0; i < n; i++) on[i * n + i] = 1
  // Column lists of each row of the one-layer mask, so the product costs O(n · nnz) rather than O(n³).
  const keys = Array.from({ length: n }, (_, i) => {
    const out: number[] = []
    for (let j = 0; j < n; j++) if (on[i * n + j]) out.push(j)
    return out
  })
  let reach = on.slice()
  for (let l = 1; l < layers; l++) {
    const next = new Uint8Array(n * n)
    for (let i = 0; i < n; i++)
      for (const k of keys[i]) for (let j = 0; j < n; j++) if (reach[k * n + j]) next[i * n + j] = 1
    reach = next
  }
  return fromData(Float64Array.from(reach), [n, n])
}
