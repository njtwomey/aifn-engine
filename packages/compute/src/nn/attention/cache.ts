/**
 * Key–value caches for autoregressive decoding (Pope et al., 2022): a decoder's keys and values for past tokens do not
 * change, so each step computes them only for the new token and appends them, turning the cost of a step from
 * quadratic to linear in the length. Caches are plain values (tensors and positions), extended by returning a new
 * cache, so a decoding state can be stored, replayed and branched (beam search).
 *
 * `kvCacheMemory` gives the accounting: bytes per token and in total for multi-head, grouped-query, multi-query and
 * latent attention, with an optional sliding window that caps the tokens kept.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { concat, shapeOfValue, slice, type Value } from 'aifn-compute/foundation/tensor'

/** One layer's cache: keys and values [..., g, T, d_h] (after rotary positions) and the tokens' absolute positions. */
export type KvCache = { readonly keys: Value; readonly values: Value; readonly positions: readonly number[] }

/** One layer's latent cache (multi-head latent attention): latents [..., T, d_c] and rotary keys [..., T, d_R]. */
export type LatentCache = { readonly latent: Value; readonly ropeKeys: Value; readonly positions: readonly number[] }

const along = (axis: number) => (a: Value | undefined, b: Value) => (a === undefined ? b : concat([a, b], axis))

/** The cache with keys and values [..., g, T, d_h] of new tokens at `positions` appended. */
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

/** The cache with latents [..., T, d_c] and rotary keys [..., T, d_R] of new tokens appended. */
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
 * The cache keeping only its last `keep` tokens: a rolling buffer for sliding-window attention of window w = keep
 * (Jiang et al., 2023), whose memory stops growing at w tokens.
 */
export function trimKvCache(cache: KvCache, keep: Size): KvCache {
  const n = cache.positions.length
  if (n <= keep) return cache
  const rank = shapeOfValue(cache.keys).length
  const specs = (v: Value) => slice(v, ...Array.from({ length: rank - 2 }, () => null), [n - keep, n])
  return { keys: specs(cache.keys), values: specs(cache.values), positions: cache.positions.slice(n - keep) }
}

/** The tokens a cache holds. */
export function cacheLength(cache: KvCache | LatentCache | null): Size {
  return cache?.positions.length ?? 0
}

/** A decoder's attention layout, for `kvCacheMemory`. */
export type KvCacheLayout = {
  /** Decoder layers. */
  layers: Size
  /** Query heads h. */
  heads: Size
  /** Key–value heads g (default h; 1 for multi-query). Ignored with `latentDim`. */
  kvHeads?: Size
  /** Width of a head d_h. */
  headDim: Size
  /** Multi-head latent attention: the cached latent width d_c (and `ropeDim`, the shared rotary key d_R). */
  latentDim?: Size
  ropeDim?: Size
  /** Bytes per stored number (2 for 16-bit floats, the default; 1 for 8-bit). */
  bytesPerValue?: number
  /** A sliding window: at most this many tokens are kept. */
  window?: Size
}

/** The memory of a key–value cache. */
export type KvCacheMemory = {
  /** Numbers stored per token across all layers: 2·layers·g·d_h, or layers·(d_c + d_R) for latent attention. */
  valuesPerToken: number
  /** Bytes per token. */
  bytesPerToken: number
  /** Tokens actually held (the sequence length, capped by the window), times the batch. */
  tokens: number
  /** Bytes in total. */
  bytes: number
  /** The ratio to multi-head attention with the same h and d_h (1 for MHA, g/h for GQA). */
  relativeToMultiHead: number
}

/**
 * The memory of a decoder's key–value cache for `batch` sequences of `length` tokens (Pope et al., 2022, §3.1): every
 * layer stores a key and a value of width d_h for each of its g key–value heads, 2·layers·g·d_h numbers per token,
 * which grouped-query (g < h) and multi-query (g = 1) attention shrink by h/g; latent attention stores layers·(d_c + d_R)
 * (DeepSeek-AI, 2024). A sliding window caps the tokens at w.
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
