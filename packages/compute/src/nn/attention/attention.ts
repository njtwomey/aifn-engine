/**
 * Attention: scaled dot-product attention (Vaswani et al., 2017, §3.2.1) with masks, additive position biases, logit
 * soft-capping (Gemma Team, 2024) and a key–value cache; multi-head attention (§3.2.2) with shared key–value heads
 * (multi-query attention, Shazeer, 2019; grouped-query attention, Ainslie et al., 2023), query–key normalisation
 * (Henry et al., 2020) and rotary positions; and multi-head latent attention, which caches one low-rank latent per
 * token instead of every head's keys and values (DeepSeek-AI, 2024, "DeepSeek-V2"). Attention weights are returned and
 * tapped, so figures can draw them.
 *
 * Tensors are row-major with tokens on the second-to-last axis: a sequence is `[..., T, d]`, and heads are split out
 * in front of it, `[..., h, T, d_h]`. Every leading axis broadcasts, so a batch is a leading axis. Everything is
 * differentiable in the parameters and inputs; masked scores get weight 0 and no gradient.
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
 * Logit soft-capping $c \tanh(x / c)$ (Gemma Team, 2024): smooth, odd, $\approx x$ for $\lvert x \rvert \ll c$, and
 * bounded by $\pm c$, so no score can grow without limit and saturate the softmax. Throws `DomainError` unless $c > 0$.
 *
 * @param x The scores to cap, elementwise.
 * @param cap The cap $c$, positive.
 * @returns The capped scores, the shape of `x`.
 *
 * @example Small scores pass almost unchanged; large ones stop at the cap
 * print('x:', [-100, -1, 0, 1, 100])
 * print('capped at 5:', softCap(tensor([-100, -1, 0, 1, 100]), 5))
 */
export function softCap(x: Value, cap: number): Value {
  if (!(cap > 0)) throw new DomainError('softCap', `softCap: the cap ${cap} must be positive`)
  return mul(cap, tanh(mul(1 / cap, x)))
}

/** Options of `scaledDotProductAttention`. */
export type AttentionOptions = {
  /** 1 where a query may attend to a key, 0 where not; broadcast to `[..., Tq, Tk]`. */
  mask?: Value
  /** Apply the causal mask (query $i$ sees keys $j \le i$, aligned at the end when $T_q < T_k$). */
  causal?: boolean
  /**
   * A window of keys: with `causal`, query $i$ sees only the `window` keys ending at its position; without, a symmetric
   * band (see `MaskOptions.window`).
   */
  window?: Size
  /** Added to the scores before the softmax, broadcast to `[..., Tq, Tk]` (ALiBi, T5 biases). */
  bias?: Value
  /** The scale of the scores (default $1/\sqrt{d_k}$). */
  scale?: number
  /** Soft-cap the scores at $\pm$`softCap` (after the scale and bias, before masking). */
  softCap?: number
  /** Called with the weights before they multiply V; returns the weights to use (for recording and probing). */
  tapWeights?: (weights: Value) => Value
}

/** The output of attention and the weights that produced it. */
export type AttentionResult = {
  /** $\mathrm{softmax}(s\Qmat\Kmat^\top + \Bmat)\Vmat$, shape `[..., Tq, d_v]`, with $s$ the scale. */
  output: Value
  /** The attention weights, shape `[..., Tq, Tk]`, each row summing to 1 over the keys it may see. */
  weights: Value
}

/**
 * Scaled dot-product attention $\mathrm{softmax}(\Qmat\Kmat^\top / \sqrt{d_k} + \Bmat)\Vmat$ for queries
 * `[..., Tq, d_k]`, keys `[..., Tk, d_k]` and values `[..., Tk, d_v]` (leading axes broadcast), the softmax taken over
 * the keys (Vaswani et al., 2017, §3.2.1). The scores are scaled, biased, soft-capped and then masked: masked scores
 * are set to $-\infty$ before the softmax, so they get weight 0 and no gradient. A query that may see no key gets NaN
 * weights.
 *
 * @param q The queries $\Qmat$ `[..., Tq, d_k]`.
 * @param k The keys $\Kmat$ `[..., Tk, d_k]`.
 * @param v The values $\Vmat$ `[..., Tk, d_v]`, one row per key.
 * @param options Masks (`mask`, `causal`, `window`), the additive bias $\Bmat$, the scale, soft-capping and a tap on
 *   the weights.
 * @returns The output `[..., Tq, d_v]` and the weights `[..., Tq, Tk]`.
 *
 * @example Three tokens of width 4, causal: zeros above the diagonal and rows summing to one
 * const q = normals(stream(0), [3, 4])
 * const k = normals(stream(1), [3, 4])
 * const v = normals(stream(2), [3, 4])
 * const { output, weights } = scaledDotProductAttention(q, k, v, { causal: true })
 * print('output:', shapeOf(output), 'weights:', shapeOf(weights))
 * print('weights:', weights)
 * print('row sums:', sum(weights, 1))
 *
 * @example One query, three keys: equal keys get equal weight, a mask removes one, a larger scale sharpens
 * const q = tensor([[1, 0]])
 * const k = tensor([[1, 0], [0, 1], [1, 0]])
 * const v = tensor([[1, 0], [0, 1], [0, 0]])
 * print('weights:', scaledDotProductAttention(q, k, v).weights)
 * print('third key masked:', scaledDotProductAttention(q, k, v, { mask: tensor([[1, 1, 0]]) }).weights)
 * print('scale 10:', scaledDotProductAttention(q, k, v, { scale: 10 }).weights)
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

/**
 * Split the last axis into heads and move them in front of the tokens: `[..., T, h * d]` to `[..., h, T, d]`. Head
 * $j$ takes the $j$-th block of $d$ consecutive columns.
 *
 * @param x The projected tokens `[..., T, h * d]`.
 * @param heads The number of heads $h$, which must divide the last axis.
 * @returns The heads `[..., h, T, d]`.
 *
 * @example Three tokens of width 4 as two heads of width 2, and back
 * const x = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
 * const heads = splitHeads(x, 2)
 * print('shape:', shapeOf(heads))
 * print('heads:', heads)
 * print('merged back:', mergeHeads(heads))
 */
export function splitHeads(x: Value, heads: Size): Value {
  const s = shapeOfValue(x)
  const lead = s.slice(0, -2)
  const [t, d] = s.slice(-2)
  const r = lead.length
  const split = reshape(x, [...lead, t, heads, d / heads])
  return permute(split, [...lead.map((_, i) => i), r + 1, r, r + 2])
}

/**
 * Concatenate heads back along the last axis, the inverse of `splitHeads`: `[..., h, T, d]` to `[..., T, h * d]`.
 *
 * @param x The heads `[..., h, T, d]`.
 * @returns The tokens `[..., T, h * d]`, head $j$ in columns $jd$ to $jd + d - 1$.
 *
 * @example Two heads of width 2 over two tokens, side by side
 * const heads = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
 * print('merged:', mergeHeads(heads))
 */
export function mergeHeads(x: Value): Value {
  const s = shapeOfValue(x)
  const lead = s.slice(0, -3)
  const [h, t, d] = s.slice(-3)
  const r = lead.length
  return reshape(permute(x, [...lead.map((_, i) => i), r + 1, r, r + 2]), [...lead, t, h * d])
}

/**
 * Each of $g$ key–value heads `[..., g, T, d]` repeated `groups` times in place, giving `[..., g * groups, T, d]`, so
 * query head $j$ reads key–value head $\lfloor j / \mathrm{groups} \rfloor$ (grouped-query attention). The gradient
 * sums over the copies.
 *
 * @param x The key or value heads `[..., g, T, d]`.
 * @param groups How many query heads share each key–value head ($h/g$); 1 returns `x` itself.
 * @returns The repeated heads `[..., g * groups, T, d]`.
 *
 * @example Two key–value heads, each read by two query heads
 * const kv = tensor([[[1, 1]], [[2, 2]]])
 * const repeated = repeatKvHeads(kv, 2)
 * print('shape:', shapeOf(repeated))
 * print('heads:', repeated)
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
 * Parameters of `MultiHeadAttention`: the query `[d, h * d_h]`, key and value `[d, g * d_h]` and output
 * `[h * d_h, d]` projections, and with `qkNorm` the RMS-norm scales of queries and keys (`[d_h]`, shared by the heads).
 */
export type MultiHeadAttentionParams = {
  /** The query projection `[d, h * d_h]`. */
  query: LinearParams
  /** The key projection `[d, g * d_h]`. */
  key: LinearParams
  /** The value projection `[d, g * d_h]`. */
  value: LinearParams
  /** The output projection `[h * d_h, d]`, applied to the concatenated heads. */
  output: LinearParams
  /** The RMS-norm scale `[d_h]` of every query head (QK-norm); absent for none. */
  queryNorm?: NormParams
  /** The RMS-norm scale `[d_h]` of every key head (QK-norm); absent for none. */
  keyNorm?: NormParams
}

/** Options of `multiHeadAttention` and `MultiHeadAttention`. */
export type MultiHeadOptions = {
  /** Query heads $h$. */
  heads: Size
  /**
   * Key–value heads $g$, dividing $h$ (default $h$). $g = 1$ is multi-query attention (Shazeer, 2019); $1 < g < h$ is
   * grouped-query attention (Ainslie et al., 2023). The cache shrinks by $h/g$.
   */
  kvHeads?: Size
  /** Hide keys at later positions than the query, by absolute position (so it stays right with a cache). */
  causal?: boolean
  /**
   * A sliding window of this many keys per query (with `causal`; Child et al., 2019; Jiang et al., 2023). The returned
   * cache is trimmed to it.
   */
  window?: Size
  /** A mask of ones and zeros broadcast to `[..., h, Tq, Tk]` (`Tk` counting cached keys), combined with `causal`. */
  mask?: Value
  /** An additive bias `[h, Tq, Tk]` or broadcastable (ALiBi, T5). */
  bias?: Value
  /** Rotate queries and keys by position (RoPE); `true` for the defaults. */
  rope?: RopeOptions | boolean
  /** Soft-cap the scores at $\pm$`softCap`. */
  softCap?: number
  /** The score scale (default $1/\sqrt{d_h}$). */
  scale?: number
  /** See `AttentionOptions.tapWeights`. */
  tapWeights?: (weights: Value) => Value
}

/** Where a call sits in a sequence: the positions of its tokens and the cache of earlier ones. */
export type AttentionState = {
  /** Absolute positions of the tokens of this call (default: after the cache's, or $0, 1, \dots$). */
  positions?: readonly number[]
  /** Keys and values of earlier tokens (decoding); null or absent for none. */
  cache?: KvCache | null
}

/**
 * `AttentionResult` with `cache`, the cache extended by this call's keys and values (trimmed to the window, if any),
 * and `mask`, the position mask used (`[Tq, Tk]`, null without `causal` or `window`).
 */
export type CachedAttentionResult = AttentionResult & { cache: KvCache; mask: Tensor | null }

/**
 * The rotary options a `rope` option stands for.
 *
 * @param r The option as given: `true` for the defaults, false or undefined for no rotation, or the options.
 * @returns The options, or null when queries and keys are not rotated.
 */
const ropeOf = (r: RopeOptions | boolean | undefined): RopeOptions | null =>
  r === true ? {} : r === false || r === undefined ? null : r

/**
 * Multi-head attention (Vaswani et al., 2017, §3.2.2): queries from `xq` `[..., Tq, d_model]` and keys and values from
 * `xkv` `[..., Tk, d_model]` (the same tensor for self-attention), projected into $h$ query heads and $g$ key–value
 * heads of width $d_h$, normalised (QK-norm, when the parameters carry its scales), rotated (RoPE), attended separately
 * with the key–value heads shared by groups of $h/g$ query heads, concatenated and projected back. Throws
 * `DomainError` when $g$ does not divide $h$.
 *
 * `state` places the call in a sequence: the new keys and values are appended to `state.cache` (a key–value cache),
 * and positions continue from it, so decoding one token at a time gives the same outputs as one causal pass. When
 * `xq` and `xkv` have different lengths (cross-attention), the queries take positions $0, \dots, T_q - 1$.
 *
 * @param params The query, key, value and output projections, and the QK-norm scales if any.
 * @param xq The tokens the queries come from, `[..., Tq, d_model]`.
 * @param xkv The tokens the keys and values come from, `[..., Tk, d_model]`: `xq` itself for self-attention.
 * @param options The heads, masks, bias, rotary positions, soft-cap, scale and weight tap.
 * @param state The positions of this call's tokens and the cache of earlier ones (default: a fresh sequence from 0).
 * @returns The output `[..., Tq, d_model]`, the weights `[..., h, Tq, Tk]` (`Tk` counting cached keys), the extended
 *   cache (trimmed to `window`, if any) and the position mask used.
 *
 * @example Self-attention over three tokens of width 4 with two heads
 * const params = MultiHeadAttention(4, { heads: 2 }).init(stream(0))
 * const x = normals(stream(1), [3, 4])
 * const { output, weights, mask } = multiHeadAttention(params, x, x, { heads: 2, causal: true })
 * print('output:', shapeOf(output), 'weights:', shapeOf(weights))
 * print('mask:', mask)
 * print('weights per head:', weights)
 *
 * @example Decoding one token at a time against the cache gives the outputs of one causal pass
 * const options = { heads: 2, causal: true, rope: true }
 * const params = MultiHeadAttention(4, options).init(stream(0))
 * const x = normals(stream(1), [3, 4])
 * const first = multiHeadAttention(params, slice(x, [0, 1]), slice(x, [0, 1]), options)
 * const second = multiHeadAttention(params, slice(x, [1, 2]), slice(x, [1, 2]), options, { cache: first.cache })
 * const third = multiHeadAttention(params, slice(x, [2, 3]), slice(x, [2, 3]), options, { cache: second.cache })
 * print('one pass:', multiHeadAttention(params, x, x, options).output)
 * print('one at a time:', concat([first.output, second.output, third.output], 0))
 * print('cached positions:', third.cache.positions)
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
  /** Width of each head $d_h$ (default $d_{\mathrm{model}}/h$). */
  headDim?: Size
  /** Normalise queries and keys by RMS norm per head before the dot product (Henry et al., 2020). Default false. */
  qkNorm?: boolean
  /** Biases on the projections (default true). */
  projectionBias?: boolean
}

/**
 * Multi-head self-attention as a layer over `[..., T, d_model]`, with `kvHeads`, `qkNorm`, `rope`, masks and
 * soft-capping as in `multiHeadAttention`; it keeps no cache. Projections are Xavier-uniform with zero biases, and the
 * QK-norm scales start at one. The weights are tapped at `<path>.weights` and the output at `<path>`. Throws
 * `DomainError` when $h$ does not divide $d_{\mathrm{model}}$ (and no `headDim` is given) or $g$ does not divide $h$.
 *
 * @param dModel The model width $d_{\mathrm{model}}$, of input and output.
 * @param options The heads and attention options, the head width, QK-norm and whether the projections have biases.
 * @returns The layer: `init` draws `MultiHeadAttentionParams`; `apply` attends the input to itself.
 *
 * @example Multi-query attention (one key–value head) over three tokens, with the weights read from the tap
 * const layer = MultiHeadAttention(4, { heads: 2, kvHeads: 1, causal: true })
 * const params = layer.init(stream(0))
 * print(layer.label)
 * print('query, key projections:', shapeOf(params.query.weight), shapeOf(params.key.weight))
 * const tapped = {}
 * const y = layer.apply(params, normals(stream(1), [3, 4]), { tap: (path, v) => ((tapped[path] = v), v) })
 * print('output:', shapeOf(y))
 * print('weights:', tapped.weights)
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
 * per-token latent $\cvec = \xvec\Wmat_{DKV}$ of width $d_c$ (`kvDown`, then `keyUp` and `valueUp` to $h d_h$); a
 * shared rotary key $\kvec_R = \mathrm{RoPE}(\xvec\Wmat_{KR})$ of width $d_R$ (`keyRope`) carries position; queries
 * come from $\xvec$ directly (`query`, $h d_h$) or through their own latent (`queryDown`), with rotary parts
 * (`queryRope`, $h d_R$).
 */
export type LatentAttentionParams = {
  /** The query down-projection `[d, d_c']` to a query latent; absent when queries come from the input directly. */
  queryDown?: LinearParams
  /** The content query projection, from the input or the query latent to `h * d_h`. */
  query: LinearParams
  /** The rotary query projection, from the input or the query latent to `h * d_R`. */
  queryRope: LinearParams
  /** The key–value down-projection `[d, d_c]`, whose output is the cached latent. */
  kvDown: LinearParams
  /** The RMS-norm scale `[d_c]` of the latent; absent for none. */
  kvNorm?: NormParams
  /** The key up-projection `[d_c, h * d_h]`, from the latent to the content keys. */
  keyUp: LinearParams
  /** The value up-projection `[d_c, h * d_h]`, from the latent to the values. */
  valueUp: LinearParams
  /** The shared rotary key projection `[d, d_R]`. */
  keyRope: LinearParams
  /** The output projection `[h * d_h, d]`, applied to the concatenated heads. */
  output: LinearParams
}

/** Options of `multiHeadLatentAttention`. */
export type LatentAttentionOptions = {
  /** Heads $h$. */
  heads: Size
  /** Hide keys at later positions than the query, by absolute position. */
  causal?: boolean
  /** A sliding window of this many keys per query (with `causal`). The latent cache is not trimmed to it. */
  window?: Size
  /** The rotary embedding of the decoupled rotary parts (default RoPE with base 10000). */
  rope?: RopeOptions
  /** Soft-cap the scores at $\pm$`softCap`. */
  softCap?: number
  /** See `AttentionOptions.tapWeights`. */
  tapWeights?: (weights: Value) => Value
}

/**
 * `AttentionResult` with `cache`, the latent cache extended by this call, and `mask`, the position mask used
 * (`[T, Tk]`, null without `causal` or `window`).
 */
export type LatentAttentionResult = AttentionResult & { cache: LatentCache; mask: Tensor | null }

/**
 * Multi-head latent attention (DeepSeek-AI, 2024, "DeepSeek-V2", §2.1), self-attention over `x` `[..., T, d_model]`.
 * Each head's key is $[\kvec_C; \kvec_R]$ with $\kvec_C$ decompressed from the latent and $\kvec_R$ the shared rotary
 * key; its query $[\qvec_C; \qvec_R]$ likewise; the score scale is $1/\sqrt{d_h + d_R}$. Only the latent $\cvec$
 * ($d_c$ numbers) and $\kvec_R$ ($d_R$) are cached per token, against $2 h d_h$ for multi-head attention, and every
 * past token's keys and values are decompressed from the cache at each call.
 *
 * @param params The down, up, rotary and output projections.
 * @param x The tokens `[..., T, d_model]`.
 * @param options The heads, masks, rotary options, soft-cap and weight tap.
 * @param state The positions of this call's tokens and the latent cache of earlier ones (default: a fresh sequence
 *   from 0).
 * @returns The output `[..., T, d_model]`, the weights `[..., h, T, Tk]`, the extended latent cache and the mask used.
 *
 * @example Three tokens in one pass, then the last one again after a cache of the first two
 * const options = { heads: 2, latentDim: 2, ropeDim: 2, causal: true }
 * const params = MultiHeadLatentAttention(4, options).init(stream(0))
 * const x = normals(stream(1), [3, 4])
 * const full = multiHeadLatentAttention(params, x, options)
 * print('output:', shapeOf(full.output), 'weights:', shapeOf(full.weights))
 * print('cached latent, rotary keys:', shapeOf(full.cache.latent), shapeOf(full.cache.ropeKeys))
 * const first = multiHeadLatentAttention(params, slice(x, [0, 2]), options)
 * const last = multiHeadLatentAttention(params, slice(x, [2, 3]), options, { cache: first.cache })
 * print('last token, one pass:', slice(full.output, [2, 3]))
 * print('last token, cached:', last.output)
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
  /** Width of each head's content part $d_h$ (default $d_{\mathrm{model}}/h$). */
  headDim?: Size
  /** The key–value latent width $d_c$. */
  latentDim: Size
  /** The rotary width $d_R$ (even). */
  ropeDim: Size
  /** A query latent width $d_c'$ (default none: queries from the input directly). */
  queryLatentDim?: Size
}

/**
 * Multi-head latent attention as a layer over `[..., T, d_model]`: `multiHeadLatentAttention` with Xavier-uniform
 * projections without biases and an RMS norm on the latent (scale one). It keeps no cache. The weights are tapped at
 * `<path>.weights` and the output at `<path>`.
 *
 * @param dModel The model width $d_{\mathrm{model}}$, of input and output.
 * @param options The heads, the latent, rotary and head widths, the query latent if any, and the attention options.
 * @returns The layer: `init` draws `LatentAttentionParams`; `apply` attends the input to itself.
 *
 * @example Width 4, two heads, a latent of width 2
 * const layer = MultiHeadLatentAttention(4, { heads: 2, latentDim: 2, ropeDim: 2, causal: true })
 * const params = layer.init(stream(0))
 * print(layer.label)
 * print('parameters:', Object.keys(params))
 * print('kvDown, keyUp:', shapeOf(params.kvDown.weight), shapeOf(params.keyUp.weight))
 * print('output:', shapeOf(layer.apply(params, normals(stream(1), [3, 4]))))
 */
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
