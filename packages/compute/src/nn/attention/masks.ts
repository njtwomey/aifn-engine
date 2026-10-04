/**
 * Attention masks: which keys each query may see, as tensors of ones (visible) and zeros (hidden) over [Tq, Tk]. Masks
 * are built from absolute positions, so they stay right when a key–value cache holds earlier tokens or a rolling
 * window has dropped some: query position p sees key position q when q ≤ p (causal) and p − q < w (a sliding window of
 * w tokens, Child et al., 2019; Beltagy et al., 2020).
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Size } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Consecutive positions start, start + 1, …, start + n − 1. */
export function positionRange(n: Size, start = 0): number[] {
  return Array.from({ length: n }, (_, i) => start + i)
}

/**
 * The positions of `n` new tokens: `given` (checked), else consecutive from one past the last of `previous` (a cache's
 * positions), else 0, 1, ….
 */
export function continuePositions(
  n: Size,
  previous: readonly number[] | undefined,
  given?: readonly number[],
): number[] {
  if (given) {
    if (given.length !== n)
      throw new ShapeError('continuePositions', `continuePositions: ${given.length} positions for ${n} tokens`)
    return [...given]
  }
  const start = previous?.length ? previous[previous.length - 1] + 1 : 0
  return positionRange(n, start)
}

/** Options of `positionMask`. */
export type MaskOptions = {
  /** Hide keys after the query (q > p). */
  causal?: boolean
  /**
   * Hide keys w or more positions from the query (|p − q| ≥ w): without `causal`, a symmetric band of 2w − 1 keys
   * (Longformer's local attention); with `causal`, a sliding window of the w tokens ending at the query.
   */
  window?: Size
}

/**
 * The mask [Tq, Tk] of query positions `queries` against key positions `keys`: 1 where the query may attend to the
 * key, 0 where not. With neither option every key is visible.
 */
export function positionMask(queries: readonly number[], keys: readonly number[], options: MaskOptions = {}): Tensor {
  const { causal = false, window } = options
  const out = new Float64Array(queries.length * keys.length)
  queries.forEach((p, i) =>
    keys.forEach((q, j) => {
      const hidden = (causal && q > p) || (window !== undefined && Math.abs(p - q) >= window)
      out[i * keys.length + j] = hidden ? 0 : 1
    }),
  )
  return fromData(out, [queries.length, keys.length])
}

/**
 * The causal mask [Tq, Tk]: lower-triangular ones, aligned at the end when Tq < Tk (the queries are the last Tq
 * positions, as when decoding against a cache), so query i sees keys ≤ i + Tk − Tq.
 */
export function causalMask(tq: Size, tk: Size = tq): Tensor {
  return positionMask(positionRange(tq, tk - tq), positionRange(tk), { causal: true })
}

/** The causal sliding-window mask [Tq, Tk]: query i sees the `window` keys ending at its own position. */
export function slidingWindowMask(tq: Size, window: Size, tk: Size = tq): Tensor {
  return positionMask(positionRange(tq, tk - tq), positionRange(tk), { causal: true, window })
}

/**
 * The padding mask of a batch of sequences with `lengths` valid tokens each, padded to `tk`: shape [B, 1, 1, Tk]
 * (broadcasting over heads and queries), 1 on real tokens and 0 on padding.
 */
export function paddingMask(lengths: readonly number[], tk: Size): Tensor {
  const out = new Float64Array(lengths.length * tk)
  lengths.forEach((n, b) => {
    for (let j = 0; j < Math.min(n, tk); j++) out[b * tk + j] = 1
  })
  return fromData(out, [lengths.length, 1, 1, tk])
}
