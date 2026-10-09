/**
 * Attention masks: which keys each query may see, as tensors of ones (visible) and zeros (hidden) of shape
 * `[Tq, Tk]`, one row per query and one column per key.
 *
 * Masks are built from absolute positions, so they stay right when a key–value cache holds earlier tokens or a rolling
 * window has dropped some: query position $p$ sees key position $q$ when $q \le p$ (causal) and $p - q < w$ (a sliding
 * window of $w$ tokens; Child et al., 2019; Beltagy et al., 2020). `scaledDotProductAttention` sets the scores of the
 * zeros to $-\infty$, so they get no weight.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Size } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * Consecutive positions $s, s + 1, \dots, s + n - 1$, with $s$ = `start`.
 *
 * @param n The number of positions $n$.
 * @param start The first position $s$.
 * @returns The $n$ positions in increasing order.
 *
 * @example The positions of three tokens, from the start and after four cached ones
 * print('from 0:', positionRange(3))
 * print('from 4:', positionRange(3, 4))
 */
export function positionRange(n: Size, start = 0): number[] {
  return Array.from({ length: n }, (_, i) => start + i)
}

/**
 * The positions of `n` new tokens: `given` (checked), else consecutive from one past the last of `previous` (a cache's
 * positions), else $0, 1, \dots, n - 1$. Throws `ShapeError` when `given` does not hold `n` positions.
 *
 * @param n The number of new tokens.
 * @param previous The positions already held (a key–value cache's `positions`); only the last is read. Undefined or
 *   empty for none.
 * @param given Positions the caller chose for the new tokens. When present they are returned as given (copied), and
 *   `previous` is ignored.
 * @returns The $n$ positions of the new tokens.
 *
 * @example New tokens continue where the cache stopped
 * print('fresh:', continuePositions(2, undefined))
 * print('after a cache of 0 to 4:', continuePositions(2, [0, 1, 2, 3, 4]))
 * print('given:', continuePositions(2, [0, 1], [7, 9]))
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

/** Options of `positionMask`: with neither, every key is visible. */
export type MaskOptions = {
  /** Hide keys after the query ($q > p$). */
  causal?: boolean
  /**
   * Hide keys $w$ or more positions from the query ($\lvert p - q \rvert \ge w$): without `causal`, a symmetric band
   * of $2w - 1$ keys (Longformer's local attention); with `causal`, a sliding window of the $w$ tokens ending at the
   * query.
   */
  window?: Size
}

/**
 * The mask `[Tq, Tk]` of query positions `queries` against key positions `keys`: 1 where the query may attend to the
 * key, 0 where not. With neither option every key is visible.
 *
 * @param queries The absolute positions $p$ of the $T_q$ queries, one per row of the mask.
 * @param keys The absolute positions $q$ of the $T_k$ keys, one per column of the mask (with a cache, the cached
 *   positions followed by the new ones).
 * @param options Which keys to hide: `causal` and `window`.
 * @returns A `[Tq, Tk]` tensor of ones and zeros.
 *
 * @example Causal, and a symmetric band
 * print('causal:', positionMask([0, 1, 2], [0, 1, 2], { causal: true }))
 * print('band, window 2:', positionMask([0, 1, 2], [0, 1, 2], { window: 2 }))
 *
 * @example Three new queries against two cached keys and their own
 * print('mask:', positionMask([2, 3, 4], [0, 1, 2, 3, 4], { causal: true }))
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
 * The causal mask `[Tq, Tk]`: lower-triangular ones, aligned at the end when $T_q < T_k$ (the queries are the last
 * $T_q$ positions, as when decoding against a cache), so query $i$ sees keys $j \le i + T_k - T_q$.
 *
 * @param tq The number of queries $T_q$.
 * @param tk The number of keys $T_k$ (default $T_q$), at least $T_q$ for every query to see a key.
 * @returns A `[Tq, Tk]` tensor of ones and zeros.
 *
 * @example Zeros above the diagonal, and one query decoding against three keys
 * print('3 x 3:', causalMask(3))
 * print('1 query, 3 keys:', causalMask(1, 3))
 */
export function causalMask(tq: Size, tk: Size = tq): Tensor {
  return positionMask(positionRange(tq, tk - tq), positionRange(tk), { causal: true })
}

/**
 * The causal sliding-window mask `[Tq, Tk]`: query $i$ sees the `window` keys ending at its own position (aligned at
 * the end when $T_q < T_k$, as in `causalMask`).
 *
 * @param tq The number of queries $T_q$.
 * @param window The number of keys $w$ each query sees, its own included.
 * @param tk The number of keys $T_k$ (default $T_q$).
 * @returns A `[Tq, Tk]` tensor of ones and zeros: a band of width $w$ on and below the diagonal.
 *
 * @example Each of four tokens sees itself and the one before
 * print('window 2:', slidingWindowMask(4, 2))
 */
export function slidingWindowMask(tq: Size, window: Size, tk: Size = tq): Tensor {
  return positionMask(positionRange(tq, tk - tq), positionRange(tk), { causal: true, window })
}

/**
 * The padding mask of a batch of sequences with `lengths` valid tokens each, padded to `tk`: shape `[B, 1, 1, Tk]`
 * (broadcasting over heads and queries), 1 on real tokens and 0 on padding.
 *
 * @param lengths The number of real tokens of each of the $B$ sequences; the first that many keys are visible (all
 *   $T_k$ when a length exceeds it).
 * @param tk The padded length $T_k$.
 * @returns A `[B, 1, 1, Tk]` tensor of ones and zeros.
 *
 * @example Two sequences of lengths 3 and 1, padded to 3
 * const mask = paddingMask([3, 1], 3)
 * print('shape:', shapeOf(mask))
 * print('mask:', mask)
 */
export function paddingMask(lengths: readonly number[], tk: Size): Tensor {
  const out = new Float64Array(lengths.length * tk)
  lengths.forEach((n, b) => {
    for (let j = 0; j < Math.min(n, tk); j++) out[b * tk + j] = 1
  })
  return fromData(out, [lengths.length, 1, 1, tk])
}
