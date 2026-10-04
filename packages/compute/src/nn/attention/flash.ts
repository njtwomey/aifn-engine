/**
 * Tiled attention with the online softmax (Milakov and Gimelshein, 2018; Rabe and Staats, 2021; Dao et al., 2022,
 * FlashAttention): the scores are processed one tile of queries × keys at a time and never stored whole. Each query
 * keeps a running maximum m, a running normaliser ℓ and an unnormalised output a; a new tile with scores S and values
 * V updates them as
 *
 *   m′ = max(m, max_j S_j),   ℓ′ = e^{m − m′}·ℓ + Σ_j e^{S_j − m′},   a′ = e^{m − m′}·a + Σ_j e^{S_j − m′}·V_j,
 *
 * and the output is a/ℓ once every tile is seen: exactly softmax(S)·V, in memory linear in the length. Under a causal
 * mask, tiles wholly above the diagonal are skipped.
 */

import type { Algorithm, Size, Status } from 'aifn-compute/foundation/contracts'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** A tile: query rows [q0, q1) against key rows [k0, k1). */
export type AttentionTile = { readonly queries: readonly [number, number]; readonly keys: readonly [number, number] }

/** Options of `flashAttentionSteps`. */
export type FlashAttentionOptions = {
  /** Queries per tile (default 4). */
  queryBlock?: Size
  /** Keys per tile (default 4). */
  keyBlock?: Size
  /** Causal masking (query i sees keys ≤ i). Default false. */
  causal?: boolean
  /** Score scale (default 1/√d). */
  scale?: number
}

/** The state of `flashAttentionSteps` after t tiles. */
export interface FlashAttentionState extends Status {
  /** Running maximum score per query [Tq] (−∞ before any tile). */
  readonly max: Tensor
  /** Running normaliser ℓ per query [Tq]. */
  readonly normaliser: Tensor
  /** Running unnormalised output a [Tq, d_v]. */
  readonly accumulator: Tensor
  /** The tile processed by the last step (null at t = 0). */
  readonly tile: AttentionTile | null
  /** The tiles in processing order (query blocks outer, key blocks inner; causal skips removed). */
  readonly tiles: readonly AttentionTile[]
  /** Tiles skipped because the causal mask hides them entirely. */
  readonly skipped: Size
  /** a/ℓ once every tile is processed, else null. */
  readonly output: Tensor | null
  readonly terminated: boolean
}

const matrix = (v: Value, what: string) => {
  const t = unwrap(v) as Tensor
  if (typeof t === 'number' || t.shape.length !== 2)
    throw new ShapeError('flashAttentionSteps', `flashAttentionSteps: ${what} must be [T, d]`)
  return { rows: t.shape[0], cols: t.shape[1], data: toFlat(t) }
}

/**
 * FlashAttention's tiled forward pass for one head as a step-through algorithm: q [Tq, d], k [Tk, d], v [Tk, d_v];
 * each step folds one tile of scores into the running maximum, normaliser and output by the online softmax. The final
 * `output` equals `scaledDotProductAttention(q, k, v)` (with the same `causal` and `scale`).
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
      output: null,
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

/** FlashAttention's tiled forward pass run to the end: softmax(q·kᵀ·scale)·v [Tq, d_v] by the online softmax. */
export function flashAttention(q: Value, k: Value, v: Value, options: FlashAttentionOptions = {}): Tensor {
  return run(flashAttentionSteps(q, k, v, options), undefined, Infinity).output!
}
