/**
 * Tiled attention with the online softmax (Milakov and Gimelshein, 2018; Rabe and Staats, 2021; Dao et al., 2022,
 * FlashAttention): the scores are processed one tile of queries by keys at a time and never stored whole. Each query
 * keeps a running maximum $m$, a running normaliser $\ell$ and an unnormalised output $\avec$; a new tile with scores
 * $s_j$ and value rows $\vvec_j$ updates them as
 *
 * $m' = \max(m, \max_j s_j)$, $\ell' = e^{m - m'} \ell + \sum_j e^{s_j - m'}$ and
 * $\avec' = e^{m - m'} \avec + \sum_j e^{s_j - m'} \vvec_j$,
 *
 * and the output is $\avec / \ell$ once every tile is seen: exactly $\mathrm{softmax}(\svec)\Vmat$, in memory
 * linear in the length. Under a causal mask, tiles wholly above the diagonal are skipped. The pass computes on
 * concrete numbers, one head at a time, to show the algorithm: it is not differentiable.
 */

import type { Algorithm, Size, Status } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A tile: `queries`, the query rows from the first index up to but not including the second, against `keys`, the key
 * rows likewise.
 */
export type AttentionTile = { readonly queries: readonly [number, number]; readonly keys: readonly [number, number] }

/** Options of `flashAttentionSteps`. */
export type FlashAttentionOptions = {
  /** Queries per tile (default 4). */
  queryBlock?: Size
  /** Keys per tile (default 4). */
  keyBlock?: Size
  /** Causal masking (query $i$ sees keys $j \le i + T_k - T_q$, as `causalMask`). Default false. */
  causal?: boolean
  /** Score scale (default $1/\sqrt{d}$). */
  scale?: number
}

/** The state of `flashAttentionSteps` after $t$ tiles. */
export interface FlashAttentionState extends Status {
  /** Running maximum score $m$ per query, `[Tq]` ($-\infty$ before any tile). */
  readonly max: Tensor
  /** Running normaliser $\ell$ per query, `[Tq]`. */
  readonly normaliser: Tensor
  /** Running unnormalised output $\avec$ per query, `[Tq, d_v]`. */
  readonly accumulator: Tensor
  /** The tile processed by the last step (null at $t = 0$). */
  readonly tile: AttentionTile | null
  /** The tiles in processing order (query blocks outer, key blocks inner; causal skips removed). */
  readonly tiles: readonly AttentionTile[]
  /** Tiles skipped because the causal mask hides them entirely. */
  readonly skipped: Size
  /**
   * $\avec / \ell$ once every tile is processed, else null. With no queries or no keys there are no tiles, and it is
   * zeros `[Tq, d_v]` from the start, as `scaledDotProductAttention` gives.
   */
  readonly output: Tensor | null
  /** True once every tile is processed (at once when there are none). */
  readonly terminated: boolean
}

/**
 * The concrete values of a `[T, d]` matrix, row-major. Throws `ShapeError` for anything that is not a matrix.
 *
 * @param v The matrix: a tensor, or a traced value whose primal is read.
 * @param what The argument's name (`q`, `k` or `v`), for the error message.
 * @returns The number of rows and columns and the row-major data.
 */
const matrix = (v: Value, what: string) => {
  const t = unwrap(v) as Tensor
  if (typeof t === 'number' || t.shape.length !== 2)
    throw new ShapeError('flashAttentionSteps', `flashAttentionSteps: ${what} must be [T, d]`)
  return { rows: t.shape[0], cols: t.shape[1], data: toFlat(t) }
}

/**
 * FlashAttention's tiled forward pass for one head as a step-through algorithm: each step folds one tile of scores
 * into the running maximum, normaliser and output by the online softmax. The final `output` equals
 * `scaledDotProductAttention(q, k, v)` (with the same `causal` and `scale`) up to rounding. Throws `ShapeError` when
 * an input is not a matrix or the shapes do not agree.
 *
 * @param q The queries `[Tq, d]`.
 * @param k The keys `[Tk, d]`.
 * @param v The values `[Tk, d_v]`.
 * @param options The tile sizes, causal masking and score scale.
 * @returns The algorithm: run it with `run(alg, undefined, steps)`; it terminates after the last tile.
 *
 * @example Three tokens in tiles of two: one tile is skipped by the causal mask, and the end matches attention
 * const q = normals(stream(0), [3, 4])
 * const k = normals(stream(1), [3, 4])
 * const v = normals(stream(2), [3, 4])
 * const alg = flashAttentionSteps(q, k, v, { queryBlock: 2, keyBlock: 2, causal: true })
 * const first = run(alg, undefined, 1)
 * print('tiles (queries, keys):', first.tiles.map((tile) => [tile.queries, tile.keys]), 'skipped:', first.skipped)
 * print('after one tile, max:', first.max, 'normaliser:', first.normaliser)
 * const end = run(alg, undefined, Infinity)
 * print('steps:', end.t)
 * print('output:', end.output)
 * print('attention:', scaledDotProductAttention(q, k, v, { causal: true }).output)
 */
export function flashAttentionSteps(
  q: Value,
  k: Value,
  v: Value,
  options: FlashAttentionOptions = {},
): Algorithm<void, FlashAttentionState> {
  const Q = matrix(q, 'q')
  const K = matrix(k, 'k')
  const V = matrix(v, 'v')
  if (Q.cols !== K.cols || K.rows !== V.rows)
    throw new ShapeError('flashAttentionSteps', 'flashAttentionSteps: q, k and v do not agree')
  const { queryBlock = 4, keyBlock = 4, causal = false } = options
  const scale = options.scale ?? 1 / Math.sqrt(Q.cols)
  const shift = K.rows - Q.rows
  const tiles: AttentionTile[] = []
  let skipped = 0
  for (let q0 = 0; q0 < Q.rows; q0 += queryBlock)
    for (let k0 = 0; k0 < K.rows; k0 += keyBlock) {
      const q1 = Math.min(q0 + queryBlock, Q.rows)
      // The last query of the block sees keys ≤ q1 − 1 + shift: a tile starting after that is hidden.
      if (causal && k0 > q1 - 1 + shift) skipped++
      else tiles.push({ queries: [q0, q1], keys: [k0, Math.min(k0 + keyBlock, K.rows)] })
    }
  const dv = V.cols
  return {
    name: 'flash-attention',
    init: () => ({
      t: 0,
      max: fromData(new Float64Array(Q.rows).fill(-Infinity), [Q.rows]),
      normaliser: fromData(new Float64Array(Q.rows), [Q.rows]),
      accumulator: fromData(new Float64Array(Q.rows * dv), [Q.rows, dv]),
      tile: null,
      tiles,
      skipped,
      // No queries or no keys: there are no tiles, and the output is zeros, as scaledDotProductAttention gives.
      output: tiles.length === 0 ? fromData(new Float64Array(Q.rows * dv), [Q.rows, dv]) : null,
      terminated: tiles.length === 0,
    }),
    step: (state) => {
      const tile = tiles[state.t]
      const m = Float64Array.from(toFlat(state.max))
      const l = Float64Array.from(toFlat(state.normaliser))
      const a = Float64Array.from(toFlat(state.accumulator))
      const [q0, q1] = tile.queries
      const [k0, k1] = tile.keys
      for (let i = q0; i < q1; i++) {
        const scores: number[] = []
        let tileMax = -Infinity
        for (let j = k0; j < k1; j++) {
          let s = -Infinity
          if (!causal || j <= i + shift) {
            s = 0
            for (let c = 0; c < Q.cols; c++) s += Q.data[i * Q.cols + c] * K.data[j * K.cols + c]
            s *= scale
          }
          scores.push(s)
          tileMax = Math.max(tileMax, s)
        }
        const mNew = Math.max(m[i], tileMax)
        if (mNew === -Infinity) continue
        const decay = m[i] === -Infinity ? 0 : Math.exp(m[i] - mNew)
        l[i] *= decay
        for (let c = 0; c < dv; c++) a[i * dv + c] *= decay
        scores.forEach((s, jj) => {
          if (s === -Infinity) return
          const p = Math.exp(s - mNew)
          l[i] += p
          for (let c = 0; c < dv; c++) a[i * dv + c] += p * V.data[(k0 + jj) * dv + c]
        })
        m[i] = mNew
      }
      const t = state.t + 1
      const done = t >= tiles.length
      const output = done
        ? fromData(
            a.map((x, idx) => x / l[Math.floor(idx / dv)]),
            [Q.rows, dv],
          )
        : null
      return {
        ...state,
        t,
        max: fromData(m, [Q.rows]),
        normaliser: fromData(l, [Q.rows]),
        accumulator: fromData(a, [Q.rows, dv]),
        tile,
        output,
        terminated: done,
      }
    },
  }
}

/**
 * FlashAttention's tiled forward pass run to the end: $\mathrm{softmax}(s\Qmat\Kmat^\top)\Vmat$ (`[Tq, d_v]`, $s$
 * the scale) by the online softmax, never forming the whole score matrix. Not differentiable; see
 * `flashAttentionSteps`.
 *
 * @param q The queries `[Tq, d]`.
 * @param k The keys `[Tk, d]`.
 * @param v The values `[Tk, d_v]`.
 * @param options The tile sizes, causal masking and score scale.
 * @returns The attention output `[Tq, d_v]`.
 *
 * @example Tiles of one query and one key give the attention output exactly
 * const q = normals(stream(0), [3, 4])
 * const k = normals(stream(1), [3, 4])
 * const v = normals(stream(2), [3, 4])
 * print('flash:', flashAttention(q, k, v, { queryBlock: 1, keyBlock: 1 }))
 * print('attention:', scaledDotProductAttention(q, k, v).output)
 */
export function flashAttention(q: Value, k: Value, v: Value, options: FlashAttentionOptions = {}): Tensor {
  return run(flashAttentionSteps(q, k, v, options), undefined, Infinity).output!
}
