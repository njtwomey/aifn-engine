/**
 * Key–value caches for autoregressive decoding (Pope et al., 2022): a decoder's keys and values for past tokens do not
 * change, so each step computes them only for the new token and appends them, turning the cost of a step from
 * quadratic to linear in the length. Caches are plain values (tensors and positions), extended by returning a new
 * cache, so a decoding state can be stored, replayed and branched (beam search). Tokens are stacked along the
 * second-to-last axis, and each cache keeps the absolute positions of its tokens, which the masks are built from.
 *
 * `kvCacheMemory` gives the accounting: bytes per token and in total for multi-head, grouped-query, multi-query and
 * latent attention, with an optional sliding window that caps the tokens kept.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { concat, shapeOfValue, slice, type Value } from 'aifn-compute/foundation/tensor'

/**
 * One layer's cache: `keys` and `values` `[..., g, T, d_h]` ($g$ key–value heads of width $d_h$, keys after rotary
 * positions) and `positions`, the $T$ tokens' absolute positions in order.
 */
export type KvCache = { readonly keys: Value; readonly values: Value; readonly positions: readonly number[] }

/**
 * One layer's latent cache (multi-head latent attention): `latent` `[..., T, d_c]`, the compressed key–value latents,
 * `ropeKeys` `[..., T, d_R]`, the shared rotary keys (after rotation), and `positions`, the tokens' absolute positions.
 */
export type LatentCache = { readonly latent: Value; readonly ropeKeys: Value; readonly positions: readonly number[] }

/**
 * Concatenation along `axis` that starts from nothing: the returned function joins an existing tensor (or undefined,
 * for an empty cache) and the new one.
 *
 * @param axis The axis the tokens are stacked on.
 * @returns A function of the cached value (or undefined) and the new value, returning their concatenation.
 */
const along = (axis: number) => (a: Value | undefined, b: Value) => (a === undefined ? b : concat([a, b], axis))

/**
 * The cache with keys and values of new tokens at `positions` appended along the token axis. The old cache is not
 * modified, so it can be kept and branched.
 *
 * @param cache The cache so far, or null for an empty one.
 * @param keys The new tokens' keys `[..., g, T, d_h]`, with the same leading axes, heads and width as the cache's.
 * @param values The new tokens' values `[..., g, T, d_h]`.
 * @param positions The $T$ absolute positions of the new tokens.
 * @returns A new cache holding the old tokens followed by the new ones.
 *
 * @example Two tokens, then a third, in one head of width 2
 * const first = appendKvCache(null, tensor([[[1, 0], [0, 1]]]), tensor([[[1, 1], [2, 2]]]), [0, 1])
 * const cache = appendKvCache(first, tensor([[[1, 1]]]), tensor([[[3, 3]]]), [2])
 * print('keys:', shapeOf(cache.keys))
 * print('values:', cache.values)
 * print('positions:', cache.positions)
 */
export function appendKvCache(
  cache: KvCache | null,
  keys: Value,
  values: Value,
  positions: readonly number[],
): KvCache {
  const rank = shapeOfValue(keys).length
  const cat = along(rank - 2)
  return {
    keys: cat(cache?.keys, keys),
    values: cat(cache?.values, values),
    positions: [...(cache?.positions ?? []), ...positions],
  }
}

/**
 * The latent cache with latents and rotary keys of new tokens appended along the token axis. The old cache is not
 * modified.
 *
 * @param cache The cache so far, or null for an empty one.
 * @param latent The new tokens' key–value latents `[..., T, d_c]`.
 * @param ropeKeys The new tokens' rotary keys `[..., T, d_R]`, already rotated.
 * @param positions The $T$ absolute positions of the new tokens.
 * @returns A new cache holding the old tokens followed by the new ones.
 *
 * @example Two tokens, then a third, with a latent of width 3 and a rotary key of width 2
 * const first = appendLatentCache(null, tensor([[1, 2, 3], [4, 5, 6]]), tensor([[0.1, 0.2], [0.3, 0.4]]), [0, 1])
 * const cache = appendLatentCache(first, tensor([[7, 8, 9]]), tensor([[0.5, 0.6]]), [2])
 * print('latent:', shapeOf(cache.latent), 'rotary keys:', shapeOf(cache.ropeKeys))
 * print('positions:', cache.positions)
 */
export function appendLatentCache(
  cache: LatentCache | null,
  latent: Value,
  ropeKeys: Value,
  positions: readonly number[],
): LatentCache {
  const cat = along(shapeOfValue(latent).length - 2)
  return {
    latent: cat(cache?.latent, latent),
    ropeKeys: cat(cache?.ropeKeys, ropeKeys),
    positions: [...(cache?.positions ?? []), ...positions],
  }
}

/**
 * The cache keeping only its last `keep` tokens: a rolling buffer for sliding-window attention of window $w$ = `keep`
 * (Jiang et al., 2023), whose memory stops growing at $w$ tokens. A cache of at most `keep` tokens is returned as it
 * is.
 *
 * @param cache The cache to trim.
 * @param keep The number of most recent tokens to keep.
 * @returns The cache of the last `keep` tokens, with their keys, values and positions.
 *
 * @example Four cached tokens trimmed to the last two
 * const keys = tensor([[[0, 0], [1, 1], [2, 2], [3, 3]]])
 * const kept = trimKvCache({ keys, values: keys, positions: [0, 1, 2, 3] }, 2)
 * print('keys kept:', kept.keys)
 * print('positions kept:', kept.positions)
 */
export function trimKvCache(cache: KvCache, keep: Size): KvCache {
  const n = cache.positions.length
  if (n <= keep) return cache
  const rank = shapeOfValue(cache.keys).length
  const specs = (v: Value) => slice(v, ...Array.from({ length: rank - 2 }, () => null), [n - keep, n])
  return { keys: specs(cache.keys), values: specs(cache.values), positions: cache.positions.slice(n - keep) }
}

/**
 * The number of tokens a cache holds.
 *
 * @param cache A key–value or latent cache, or null for none.
 * @returns The number of cached tokens (0 for null).
 *
 * @example An empty cache and one of three tokens
 * print('empty:', cacheLength(null))
 * print('three tokens:', cacheLength(appendKvCache(null, zeros([1, 3, 2]), zeros([1, 3, 2]), [0, 1, 2])))
 */
export function cacheLength(cache: KvCache | LatentCache | null): Size {
  return cache?.positions.length ?? 0
}

/** A decoder's attention layout, for `kvCacheMemory`. */
export type KvCacheLayout = {
  /** Decoder layers. */
  layers: Size
  /** Query heads $h$. */
  heads: Size
  /** Key–value heads $g$ (default $h$; 1 for multi-query). Ignored with `latentDim`. */
  kvHeads?: Size
  /** Width of a head $d_h$. */
  headDim: Size
  /** Multi-head latent attention: the cached latent width $d_c$ (and `ropeDim`, the shared rotary key $d_R$). */
  latentDim?: Size
  /** With `latentDim`, the width $d_R$ of the shared rotary key (default 0). */
  ropeDim?: Size
  /** Bytes per stored number (2 for 16-bit floats, the default; 1 for 8-bit). */
  bytesPerValue?: number
  /** A sliding window: at most this many tokens are kept. */
  window?: Size
}

/** The memory of a key–value cache. */
export type KvCacheMemory = {
  /**
   * Numbers stored per token across all $\ell$ layers: $2 \ell g d_h$, or $\ell (d_c + d_R)$ for latent attention.
   */
  valuesPerToken: number
  /** Bytes per token. */
  bytesPerToken: number
  /** Tokens actually held (the sequence length, capped by the window), times the batch. */
  tokens: number
  /** Bytes in total. */
  bytes: number
  /** The ratio to multi-head attention with the same $h$ and $d_h$ (1 for MHA, $g/h$ for GQA). */
  relativeToMultiHead: number
}

/**
 * The memory of a decoder's key–value cache for `batch` sequences of `length` tokens (Pope et al., 2022, §3.1): every
 * one of the $\ell$ layers stores a key and a value of width $d_h$ for each of its $g$ key–value heads,
 * $2 \ell g d_h$ numbers per token, which grouped-query ($g < h$) and multi-query ($g = 1$) attention shrink by $h/g$;
 * latent attention stores $\ell (d_c + d_R)$ (DeepSeek-AI, 2024). A sliding window caps the tokens at $w$.
 *
 * @param layout The decoder's layers, heads, widths, bytes per number and window.
 * @param length The number of tokens in each sequence.
 * @param batch The number of sequences cached together.
 * @returns The numbers and bytes per token, the tokens held, the total bytes and the ratio to multi-head attention.
 *
 * @example A 32-layer, 32-head decoder at 4096 tokens: 2 GiB in 16-bit floats, and what each variant saves
 * const layout = { layers: 32, heads: 32, headDim: 128 }
 * print('multi-head:', kvCacheMemory(layout, 4096))
 * print('8 kv heads, ratio:', kvCacheMemory({ ...layout, kvHeads: 8 }, 4096).relativeToMultiHead)
 * print('latent 512 + 64, ratio:', kvCacheMemory({ ...layout, latentDim: 512, ropeDim: 64 }, 4096).relativeToMultiHead)
 * print('window 1024, tokens held:', kvCacheMemory({ ...layout, window: 1024 }, 4096).tokens)
 */
export function kvCacheMemory(layout: KvCacheLayout, length: Size, batch: Size = 1): KvCacheMemory {
  const { layers, heads, headDim, bytesPerValue = 2 } = layout
  const full = 2 * layers * heads * headDim
  const valuesPerToken =
    layout.latentDim !== undefined
      ? layers * (layout.latentDim + (layout.ropeDim ?? 0))
      : 2 * layers * (layout.kvHeads ?? heads) * headDim
  const tokens = Math.min(length, layout.window ?? Infinity) * batch
  return {
    valuesPerToken,
    bytesPerToken: valuesPerToken * bytesPerValue,
    tokens,
    bytes: valuesPerToken * bytesPerValue * tokens,
    relativeToMultiHead: valuesPerToken / full,
  }
}
