/**
 * Attention: scaled dot-product attention (Vaswani et al., 2017, §3.2.1) with masks, additive position biases, logit
 * soft-capping (Gemma Team, 2024) and a key–value cache; multi-head attention (§3.2.2) with shared key–value heads
 * (multi-query attention, Shazeer, 2019; grouped-query attention, Ainslie et al., 2023), query–key normalisation
 * (Henry et al., 2020) and rotary positions; and multi-head latent attention, which caches one low-rank latent per
 * token instead of every head's keys and values (DeepSeek-AI, 2024, "DeepSeek-V2"). Attention weights are returned and
 * tapped, so figures can draw them.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import { softmax } from 'aifn-compute/numerics/special'
import {
  add,
  broadcastTo,
  concat,
  expandDims,
  matmul,
  mul,
  ones,
  permute,
  reshape,
  shapeOfValue,
  tanh,
  transpose,
  where,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { xavierUniform, zerosInit } from 'aifn-compute/nn/init'
import { Linear, linear, rmsNorm, tap, type Layer, type LinearParams, type NormParams } from 'aifn-compute/nn/layers'
import { appendKvCache, appendLatentCache, trimKvCache, type KvCache, type LatentCache } from './cache'
import { continuePositions, positionMask, positionRange } from './masks'
import { applyRope, type RopeOptions } from './positions'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Logit soft-capping c·tanh(x/c) (Gemma Team, 2024): smooth, odd, ≈ x for |x| ≪ c, and bounded by ±c, so no score can
 * grow without limit and saturate the softmax.
 */
export function softCap(x: Value, cap: number): Value {
  if (!(cap > 0)) throw new DomainError('softCap', `softCap: the cap ${cap} must be positive`)
  return mul(cap, tanh(mul(1 / cap, x)))
}

/** Options of `scaledDotProductAttention`. */
export type AttentionOptions = {
  /** 1 where a query may attend to a key, 0 where not; broadcast to [..., Tq, Tk]. */
  mask?: Value
  /** Apply the causal mask (query i sees keys ≤ i, aligned at the end when Tq < Tk). */
  causal?: boolean
  /** With `causal`, a sliding window: query i sees only the `window` keys ending at its position. */
  window?: Size
  /** Added to the scores before the softmax, broadcast to [..., Tq, Tk] (ALiBi, T5 biases). */
  bias?: Value
  /** The scale of the scores (default 1/√d_k). */
  scale?: number
  /** Soft-cap the scores at ±cap (after the scale and bias, before masking). */
  softCap?: number
  /** Called with the weights before they multiply V; returns the weights to use (for recording and probing). */
  tapWeights?: (weights: Value) => Value
}

/** The output of attention and the weights that produced it. */
export type AttentionResult = {
  /** softmax(QKᵀ·scale + bias)·V, shape [..., Tq, d_v]. */
  output: Value
  /** The attention weights, shape [..., Tq, Tk], each row summing to 1 over the keys it may see. */
  weights: Value
}

/**
 * Scaled dot-product attention softmax(QKᵀ/√d_k + B)·V for queries [..., Tq, d_k], keys [..., Tk, d_k] and values
 * [..., Tk, d_v] (leading axes broadcast). Masked scores are set to −∞ before the softmax, so they get weight 0 and no
 * gradient. A query that may see no key gets NaN weights.
 */
export function scaledDotProductAttention(
  q: Value,
  k: Value,
  v: Value,
  options: AttentionOptions = {},
): AttentionResult {
  const qs = shapeOfValue(q)
  const ks = shapeOfValue(k)
  const dk = qs[qs.length - 1]
  const scale = options.scale ?? 1 / Math.sqrt(dk)
  const rank = ks.length
  const axes = ks.map((_, i) => i)
  ;[axes[rank - 2], axes[rank - 1]] = [axes[rank - 1], axes[rank - 2]]
  let scores = mul(matmul(q, transpose(k, axes)), scale)
  if (options.bias !== undefined) scores = add(scores, options.bias)
  if (options.softCap !== undefined) scores = softCap(scores, options.softCap)
  const tq = qs[qs.length - 2]
  const tk = ks[ks.length - 2]
  if (options.causal || options.window !== undefined)
    scores = where(
      positionMask(positionRange(tq, tk - tq), positionRange(tk), { causal: options.causal, window: options.window }),
      scores,
      -Infinity,
    )
  if (options.mask !== undefined) scores = where(options.mask, scores, -Infinity)
  const computed = softmax(scores, { axis: -1 })
  const weights = options.tapWeights ? options.tapWeights(computed) : computed
  return { output: matmul(weights, v), weights }
}

// ── Heads ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** [..., T, h·d] → [..., h, T, d]. */
export function splitHeads(x: Value, heads: Size): Value {
  const s = shapeOfValue(x)
  const lead = s.slice(0, -2)
  const [t, d] = s.slice(-2)
  const r = lead.length
  const split = reshape(x, [...lead, t, heads, d / heads])
  return permute(split, [...lead.map((_, i) => i), r + 1, r, r + 2])
}

/** [..., h, T, d] → [..., T, h·d]. */
export function mergeHeads(x: Value): Value {
  const s = shapeOfValue(x)
  const lead = s.slice(0, -3)
  const [h, t, d] = s.slice(-3)
  const r = lead.length
  return reshape(permute(x, [...lead.map((_, i) => i), r + 1, r, r + 2]), [...lead, t, h * d])
}

/**
 * Each of g key–value heads [..., g, T, d] repeated `groups` times in place, [..., g·groups, T, d], so query head j
 * reads key–value head ⌊j/groups⌋ (grouped-query attention). The gradient sums over the copies.
 */
export function repeatKvHeads(x: Value, groups: Size): Value {
  if (groups === 1) return x
  const s = shapeOfValue(x)
  const lead = s.slice(0, -3)
  const [g, t, d] = s.slice(-3)
  const wide = broadcastTo(expandDims(x, lead.length + 1), [...lead, g, groups, t, d])
  return reshape(wide, [...lead, g * groups, t, d])
}

/**
 * Parameters of `MultiHeadAttention`: the query [d, h·d_h], key and value [d, g·d_h] and output [h·d_h, d]
 * projections, and with `qkNorm` the RMS-norm scales of queries and keys ([d_h], shared by the heads).
 */
export type MultiHeadAttentionParams = {
  query: LinearParams
  key: LinearParams
  value: LinearParams
  output: LinearParams
  queryNorm?: NormParams
  keyNorm?: NormParams
}

/** Options of `multiHeadAttention` and `MultiHeadAttention`. */
export type MultiHeadOptions = {
  /** Query heads h. */
  heads: Size
  /**
   * Key–value heads g, dividing h (default h). g = 1 is multi-query attention (Shazeer, 2019); 1 < g < h is
   * grouped-query attention (Ainslie et al., 2023). The cache shrinks by h/g.
   */
  kvHeads?: Size
  causal?: boolean
  /** A sliding window of this many keys per query (with `causal`; Child et al., 2019; Jiang et al., 2023). */
  window?: Size
  mask?: Value
  /** An additive bias [h, Tq, Tk] or broadcastable (ALiBi, T5). */
  bias?: Value
  /** Rotate queries and keys by position (RoPE); `true` for the defaults. */
  rope?: RopeOptions | boolean
  /** Soft-cap the scores at ±cap. */
  softCap?: number
  /** The score scale (default 1/√d_h). */
  scale?: number
  /** See `AttentionOptions.tapWeights`. */
  tapWeights?: (weights: Value) => Value
}

/** Where a call sits in a sequence: the positions of its tokens and the cache of earlier ones. */
export type AttentionState = {
  /** Absolute positions of the tokens of this call (default: after the cache's, or 0, 1, …). */
  positions?: readonly number[]
  /** Keys and values of earlier tokens (decoding); null or absent for none. */
  cache?: KvCache | null
}

/**
 * `AttentionResult` with the cache extended by this call's keys and values (trimmed to the window, if any) and the
 * position mask used ([Tq, Tk], null without `causal` or `window`).
 */
export type CachedAttentionResult = AttentionResult & { cache: KvCache; mask: Tensor | null }

const ropeOf = (r: RopeOptions | boolean | undefined): RopeOptions | null =>
  r === true ? {} : r === false || r === undefined ? null : r

/**
 * Multi-head attention (Vaswani et al., 2017, §3.2.2): queries from `xq` [..., Tq, d_model] and keys and values from
 * `xkv` [..., Tk, d_model] (the same tensor for self-attention), projected into h query heads and g key–value heads of
 * width d_h, normalised (QK-norm, when the parameters carry its scales), rotated (RoPE), attended separately with the
 * key–value heads shared by groups of h/g query heads, concatenated and projected back.
 *
 * `state` places the call in a sequence: the new keys and values are appended to `state.cache` (a key–value cache),
 * and positions continue from it, so decoding one token at a time gives the same outputs as one causal pass. Returns
 * the output [..., Tq, d_model], the weights [..., h, Tq, Tk], the extended cache and the mask used.
 */
export function multiHeadAttention(
  params: MultiHeadAttentionParams,
  xq: Value,
  xkv: Value,
  options: MultiHeadOptions,
  state: AttentionState = {},
): CachedAttentionResult {
  const { heads } = options
  const kvHeads = options.kvHeads ?? heads
  if (heads % kvHeads !== 0)
    throw new DomainError('multiHeadAttention', `multiHeadAttention: ${kvHeads} key–value heads do not divide ${heads}`)
  let q = splitHeads(linear(xq, params.query.weight, params.query.bias), heads)
  let k = splitHeads(linear(xkv, params.key.weight, params.key.bias), kvHeads)
  const v = splitHeads(linear(xkv, params.value.weight, params.value.bias), kvHeads)
  if (params.queryNorm) q = rmsNorm(q, params.queryNorm.gamma)
  if (params.keyNorm) k = rmsNorm(k, params.keyNorm.gamma)
  const qs = shapeOfValue(q)
  const tNew = shapeOfValue(k).at(-2)!
  const tq = qs[qs.length - 2]
  const cache = state.cache ?? null
  const positions = continuePositions(tNew, cache?.positions, state.positions)
  const queryPositions = tq === tNew ? positions : positionRange(tq)
  const rope = ropeOf(options.rope)
  if (rope) {
    q = applyRope(q, queryPositions, rope)
    k = applyRope(k, positions, rope)
  }
  const next = appendKvCache(cache, k, v, positions)
  const masked = options.causal || options.window !== undefined
  const mask = masked
    ? positionMask(queryPositions, next.positions, { causal: options.causal, window: options.window })
    : null
  const combined = mask && options.mask !== undefined ? mul(mask, options.mask) : (mask ?? options.mask)
  const { output, weights } = scaledDotProductAttention(
    q,
    repeatKvHeads(next.keys, heads / kvHeads),
    repeatKvHeads(next.values, heads / kvHeads),
    {
      mask: combined,
      bias: options.bias,
      scale: options.scale,
      softCap: options.softCap,
      tapWeights: options.tapWeights,
    },
  )
  return {
    output: linear(mergeHeads(output), params.output.weight, params.output.bias),
    weights,
    // A sliding window needs only the last w tokens from here on: a rolling buffer.
    cache: options.window !== undefined ? trimKvCache(next, options.window) : next,
    mask,
  }
}

/** Options of the `MultiHeadAttention` layer. */
export type MultiHeadLayerOptions = MultiHeadOptions & {
  /** Width of each head (default d_model/h). */
  headDim?: Size
  /** Normalise queries and keys by RMS norm per head before the dot product (Henry et al., 2020). Default false. */
  qkNorm?: boolean
  /** Biases on the projections (default true). */
  projectionBias?: boolean
}

/**
 * Multi-head self-attention as a layer over [..., T, d_model], with `kvHeads`, `qkNorm`, `rope`, masks and
 * soft-capping as in `multiHeadAttention`. The weights are tapped at `<path>.weights` and the output at `<path>`.
 */
export function MultiHeadAttention(dModel: Size, options: MultiHeadLayerOptions): Layer<MultiHeadAttentionParams> {
  const { heads, kvHeads = heads, qkNorm = false, projectionBias = true } = options
  const headDim = options.headDim ?? dModel / heads
  if (!Number.isInteger(headDim))
    throw new DomainError('MultiHeadAttention', `MultiHeadAttention: ${heads} heads do not divide ${dModel}`)
  if (heads % kvHeads !== 0)
    throw new DomainError('MultiHeadAttention', `MultiHeadAttention: ${kvHeads} key–value heads do not divide ${heads}`)
  const proj = (i: Size, o: Size) =>
    Linear(i, o, { init: xavierUniform(), biasInit: zerosInit(), bias: projectionBias })
  const query = proj(dModel, heads * headDim)
  const kv = proj(dModel, kvHeads * headDim)
  const out = proj(heads * headDim, dModel)
  const kind = kvHeads === heads ? '' : kvHeads === 1 ? ', multi-query' : `, ${kvHeads} kv heads`
  return {
    kind: 'MultiHeadAttention',
    label: `MultiHeadAttention(${dModel}, ${heads} heads${kind}${options.causal ? ', causal' : ''})`,
    init: (s: Stream) => ({
      query: query.init(child(s, 'query')),
      key: kv.init(child(s, 'key')),
      value: kv.init(child(s, 'value')),
      output: out.init(child(s, 'output')),
      ...(qkNorm ? { queryNorm: { gamma: ones([headDim]) }, keyNorm: { gamma: ones([headDim]) } } : {}),
    }),
    apply: (p, x, ctx) => {
      const { output } = multiHeadAttention(p, x, x, { ...options, tapWeights: (w) => tap(ctx, w, 'weights') })
      return tap(ctx, output)
    },
  }
}

// ── Multi-head latent attention ──────────────────────────────────────────────────────────────────────────────────────

/**
 * Parameters of multi-head latent attention (DeepSeek-AI, 2024, §2.1): keys and values are decompressed from a
 * per-token latent c = x·W_DKV [d_c] (`kvDown`, then `keyUp` and `valueUp` to h·d_h); a shared rotary key
 * k_R = RoPE(x·W_KR) [d_R] (`keyRope`) carries position; queries come from x directly (`query`, h·d_h) or through
 * their own latent (`queryDown`), with rotary parts (`queryRope`, h·d_R).
 */
export type LatentAttentionParams = {
  queryDown?: LinearParams
  query: LinearParams
  queryRope: LinearParams
  kvDown: LinearParams
  kvNorm?: NormParams
  keyUp: LinearParams
  valueUp: LinearParams
  keyRope: LinearParams
  output: LinearParams
}

/** Options of `multiHeadLatentAttention`. */
export type LatentAttentionOptions = {
  heads: Size
  causal?: boolean
  window?: Size
  /** The rotary embedding of the decoupled rotary parts (default RoPE with base 10000). */
  rope?: RopeOptions
  softCap?: number
  tapWeights?: (weights: Value) => Value
}

/** `AttentionResult` with the latent cache extended by this call. */
export type LatentAttentionResult = AttentionResult & { cache: LatentCache; mask: Tensor | null }

/**
 * Multi-head latent attention (DeepSeek-AI, 2024, "DeepSeek-V2", §2.1) over x [..., T, d_model]. Each head's key is
 * [k_C; k_R] with k_C decompressed from the latent and k_R the shared rotary key; its query [q_C; q_R] likewise; the
 * score scale is 1/√(d_h + d_R). Only the latent c [d_c] and k_R [d_R] are cached per token, against 2·h·d_h for
 * multi-head attention, and the result is the same as decompressing every past token's keys and values.
 */
export function multiHeadLatentAttention(
  params: LatentAttentionParams,
  x: Value,
  options: LatentAttentionOptions,
  state: { positions?: readonly number[]; cache?: LatentCache | null } = {},
): LatentAttentionResult {
  const { heads } = options
  const rope = options.rope ?? {}
  const cache = state.cache ?? null
  const t = shapeOfValue(x).at(-2)!
  const positions = continuePositions(t, cache?.positions, state.positions)
  const qIn = params.queryDown ? linear(x, params.queryDown.weight, params.queryDown.bias) : x
  const qC = splitHeads(linear(qIn, params.query.weight, params.query.bias), heads)
  const qR = applyRope(splitHeads(linear(qIn, params.queryRope.weight, params.queryRope.bias), heads), positions, rope)
  let latent = linear(x, params.kvDown.weight, params.kvDown.bias)
  if (params.kvNorm) latent = rmsNorm(latent, params.kvNorm.gamma)
  const kR = applyRope(linear(x, params.keyRope.weight, params.keyRope.bias), positions, rope)
  const next = appendLatentCache(cache, latent, kR, positions)
  const kC = splitHeads(linear(next.latent, params.keyUp.weight, params.keyUp.bias), heads)
  const v = splitHeads(linear(next.latent, params.valueUp.weight, params.valueUp.bias), heads)
  const ks = shapeOfValue(kC)
  const ropeShared = broadcastTo(expandDims(next.ropeKeys, -3), [
    ...ks.slice(0, -1),
    shapeOfValue(next.ropeKeys).at(-1)!,
  ])
  const q = concat([qC, qR], -1)
  const k = concat([kC, ropeShared], -1)
  const masked = options.causal || options.window !== undefined
  const mask = masked
    ? positionMask(positions, next.positions, { causal: options.causal, window: options.window })
    : null
  const { output, weights } = scaledDotProductAttention(q, k, v, {
    mask: mask ?? undefined,
    softCap: options.softCap,
    tapWeights: options.tapWeights,
  })
  return { output: linear(mergeHeads(output), params.output.weight, params.output.bias), weights, cache: next, mask }
}

/** Options of the `MultiHeadLatentAttention` layer. */
export type LatentAttentionLayerOptions = LatentAttentionOptions & {
  /** Width of each head's content part d_h (default d_model/h). */
  headDim?: Size
  /** The key–value latent width d_c. */
  latentDim: Size
  /** The rotary width d_R (even). */
  ropeDim: Size
  /** A query latent width d_c′ (default none: queries from x directly). */
  queryLatentDim?: Size
}

/** Multi-head latent attention as a layer over [..., T, d_model]; weights tapped at `<path>.weights`. */
export function MultiHeadLatentAttention(
  dModel: Size,
  options: LatentAttentionLayerOptions,
): Layer<LatentAttentionParams> {
  const { heads, latentDim, ropeDim, queryLatentDim } = options
  const headDim = options.headDim ?? dModel / heads
  const proj = (i: Size, o: Size) => Linear(i, o, { init: xavierUniform(), bias: false })
  const qIn = queryLatentDim ?? dModel
  return {
    kind: 'MultiHeadLatentAttention',
    label: `MultiHeadLatentAttention(${dModel}, ${heads} heads, latent ${latentDim})`,
    init: (s: Stream) => ({
      ...(queryLatentDim ? { queryDown: proj(dModel, queryLatentDim).init(child(s, 'queryDown')) } : {}),
      query: proj(qIn, heads * headDim).init(child(s, 'query')),
      queryRope: proj(qIn, heads * ropeDim).init(child(s, 'queryRope')),
      kvDown: proj(dModel, latentDim).init(child(s, 'kvDown')),
      kvNorm: { gamma: ones([latentDim]) },
      keyUp: proj(latentDim, heads * headDim).init(child(s, 'keyUp')),
      valueUp: proj(latentDim, heads * headDim).init(child(s, 'valueUp')),
      keyRope: proj(dModel, ropeDim).init(child(s, 'keyRope')),
      output: proj(heads * headDim, dModel).init(child(s, 'output')),
    }),
    apply: (p, x, ctx) => {
      const { output } = multiHeadLatentAttention(p, x, { ...options, tapWeights: (w) => tap(ctx, w, 'weights') })
      return tap(ctx, output)
    },
  }
}
