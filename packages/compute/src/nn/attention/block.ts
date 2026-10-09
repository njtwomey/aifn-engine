/**
 * The transformer block: self-attention and a feed-forward network, each in a residual branch with normalisation.
 * Post-norm, $\mathrm{norm}(\xvec + f(\xvec))$, is the original placement (Vaswani et al., 2017); pre-norm,
 * $\xvec + f(\mathrm{norm}(\xvec))$, keeps an identity path from input to output and trains deep stacks without
 * warm-up (Xiong et al., 2020); the parallel form $\xvec + \mathrm{attn}(\mathrm{norm}(\xvec)) +
 * \mathrm{ffn}(\mathrm{norm}(\xvec))$ runs both branches from one normalisation (Wang and Komatsuzaki, 2021, GPT-J;
 * Chowdhery et al., 2022). Relative positions (RoPE, ALiBi, T5 biases) enter inside attention; absolute ones belong to
 * the embedding, before the first block.
 */

import { child, type Stream } from 'aifn-compute/foundation/random'
import type { Size } from 'aifn-compute/foundation/contracts'
import { add, ones, shapeOfValue, zeros, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { normalInit } from 'aifn-compute/nn/init'
import {
  childContext,
  Dropout,
  layerNorm,
  rmsNorm,
  tap,
  type Context,
  type Layer,
  type NormParams,
} from 'aifn-compute/nn/layers'
import { multiHeadAttention, MultiHeadAttention, type MultiHeadAttentionParams } from './attention'
import type { KvCache } from './cache'
import { feedForward, FeedForward, type FeedForwardKind, type FeedForwardParams } from './feedforward'
import { continuePositions } from './masks'
import { alibiBias, t5RelativeBias, type RopeOptions, type T5BucketOptions } from './positions'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Relative position schemes applied inside a block: `none` (positions come from the embedding, or nowhere), `rope`
 * (rotary queries and keys), `alibi` (linear score biases) or `t5` (a learned bias per distance bucket).
 */
export type RelativePosition = 'none' | 'rope' | 'alibi' | 't5'

/** Options of `TransformerBlock` and `transformerBlock`. */
export type TransformerBlockOptions = {
  /** Query heads $h$. */
  heads: Size
  /** Key–value heads (grouped-query or multi-query attention); default `heads`. */
  kvHeads?: Size
  /** Width of each head (default $d_{\mathrm{model}}/h$). Read by `TransformerBlock` only. */
  headDim?: Size
  /** Causal self-attention (a decoder). Default false. */
  causal?: boolean
  /** A sliding attention window of this many keys per query (with `causal`). */
  window?: Size
  /** `pre` (default): $\xvec + f(\mathrm{norm}(\xvec))$; `post`: $\mathrm{norm}(\xvec + f(\xvec))$. */
  placement?: 'pre' | 'post'
  /** Both branches from one normalisation, added together (pre-norm only; `post` with it throws). Default false. */
  parallel?: boolean
  /** `layer` (default) or `rms` normalisation. */
  norm?: 'layer' | 'rms'
  /** The feed-forward kind (default `mlp`). */
  feedForward?: FeedForwardKind
  /** Feed-forward hidden width (default `feedForwardWidth`). Read by `TransformerBlock` only. */
  hidden?: Size
  /** Activation of a plain feed-forward network (default GELU). */
  activation?: 'gelu' | 'relu' | 'silu' | 'tanh'
  /** The relative position scheme (default `none`: positions come from the embedding, or nowhere, NoPE). */
  position?: RelativePosition
  /** The rotary options, with `position: 'rope'` (default RoPE with base 10000). */
  rope?: RopeOptions
  /**
   * The T5 bucketing, with `position: 't5'` (default 32 buckets up to distance 128, bidirectional unless `causal`).
   */
  t5?: T5BucketOptions
  /**
   * Query–key RMS normalisation. Default false. Read by `TransformerBlock`, which adds the scales to the parameters;
   * `transformerBlock` normalises whenever the parameters hold them.
   */
  qkNorm?: boolean
  /** Soft-cap attention scores at $\pm$`softCap`. */
  softCap?: number
  /** Dropout on each residual branch while training (default 0). */
  dropout?: number
  /**
   * Biases on the projections, read by `TransformerBlock`. Default: on attention's projections, and on the
   * feed-forward's for `mlp` only (gated kinds have none, as `FeedForward`).
   */
  bias?: boolean
}

/** Parameters of a transformer block. */
export type TransformerBlockParams = {
  /** The normalisation before attention (pre-norm, parallel) or after its residual sum (post-norm). */
  attentionNorm: NormParams
  /** The multi-head attention projections. */
  attention: MultiHeadAttentionParams
  /** The normalisation of the feed-forward branch; absent in the parallel form, which shares `attentionNorm`. */
  feedForwardNorm?: NormParams
  /** The feed-forward projections. */
  feedForward: FeedForwardParams
  /** T5's bias table `[buckets, heads]`, with `position: 't5'`. */
  positionBias?: Tensor
}

/**
 * A block's `output` `[..., T, d_model]`, its attention `weights` `[..., h, T, Tk]` and its extended key–value `cache`.
 */
export type BlockResult = { output: Value; weights: Value; cache: KvCache }

/**
 * One transformer block applied to `x` `[..., T, d_model]`, continuing from `state` (positions and the key–value cache
 * of earlier tokens), as `multiHeadAttention` does. With a tapping context it records, below its path:
 * `attentionNorm` and `feedForwardNorm` (the normalisations; in post-norm these are the residual sums normalised),
 * `attention.weights` `[..., h, T, Tk]`, `attention` (the branch output), `residual` (the stream between the
 * branches, not in the parallel form), `feedForward.hidden`, `feedForward`, the two dropouts (`attentionDropout`,
 * `feedForwardDropout`), and the block's output at the path itself. Throws `DomainError` for the parallel form with
 * post-norm, and for T5 positions without a `positionBias` table.
 *
 * @param params The normalisations, attention, feed-forward and (T5) position-bias parameters.
 * @param x The input `[..., T, d_model]`.
 * @param options The attention, normalisation, feed-forward and position options.
 * @param state The positions of this call's tokens and the cache of earlier ones (default: a fresh sequence from 0).
 * @param ctx The forward-pass context: its tap records the activations above, and `train` with `stream` turns dropout
 *   on.
 * @returns The output, the attention weights and the extended cache.
 *
 * @example A pre-norm causal block with ALiBi over three tokens of width 4, and what it taps
 * const options = { heads: 2, causal: true, position: 'alibi' }
 * const params = TransformerBlock(4, options).init(stream(0))
 * const x = normals(stream(1), [3, 4])
 * const tapped = []
 * const ctx = { tap: (path, v) => (tapped.push(path), v) }
 * const { output, weights, cache } = transformerBlock(params, x, options, {}, ctx)
 * print('output:', shapeOf(output), 'weights:', shapeOf(weights), 'cached keys:', shapeOf(cache.keys))
 * print('tapped:', tapped)
 *
 * @example The last token after a cache of the first two matches one pass over all three
 * const options = { heads: 2, causal: true, position: 'rope' }
 * const params = TransformerBlock(4, options).init(stream(0))
 * const x = normals(stream(1), [3, 4])
 * const first = transformerBlock(params, slice(x, [0, 2]), options)
 * const last = transformerBlock(params, slice(x, [2, 3]), options, { cache: first.cache })
 * print('last token, one pass:', slice(transformerBlock(params, x, options).output, [2, 3]))
 * print('last token, cached:', last.output)
 */
export function transformerBlock(
  params: TransformerBlockParams,
  x: Value,
  options: TransformerBlockOptions,
  state: { positions?: readonly number[]; cache?: KvCache | null } = {},
  ctx?: Context,
): BlockResult {
  const { placement = 'pre', parallel = false, position = 'none', heads, causal = false } = options
  if (parallel && placement !== 'pre')
    throw new DomainError('transformerBlock', 'transformerBlock: the parallel form is pre-norm')
  const normalise = (np: NormParams, h: Value) =>
    options.norm === 'rms' ? rmsNorm(h, np.gamma) : layerNorm(h, np.gamma, np.beta)
  const drop = Dropout(options.dropout ?? 0)
  const t = shapeOfValue(x).at(-2)!
  const cache = state.cache ?? null
  const positions = continuePositions(t, cache?.positions, state.positions)
  const keys = [...(cache?.positions ?? []), ...positions]
  let bias: Value | undefined
  if (position === 'alibi') bias = alibiBias(heads, positions, keys)
  if (position === 't5') {
    if (!params.positionBias)
      throw new DomainError('transformerBlock', 'transformerBlock: T5 positions need a positionBias table')
    bias = t5RelativeBias(params.positionBias, positions, keys, { bidirectional: !causal, ...options.t5 })
  }
  let weights: Value = 0
  let nextCache: KvCache | null = null
  const attend = (h: Value) => {
    const r = multiHeadAttention(
      params.attention,
      h,
      h,
      {
        heads,
        kvHeads: options.kvHeads,
        causal,
        window: options.window,
        bias,
        rope: position === 'rope' ? (options.rope ?? true) : undefined,
        softCap: options.softCap,
        tapWeights: (w) => tap(childContext(ctx, 'attention'), w, 'weights'),
      },
      { positions, cache },
    )
    weights = r.weights
    nextCache = r.cache
    return tap(childContext(ctx, 'attention'), r.output)
  }
  const ffnCtx = childContext(ctx, 'feedForward')
  const ffn = (h: Value) =>
    tap(
      ffnCtx,
      feedForward(params.feedForward, h, { kind: options.feedForward, activation: options.activation }, ffnCtx),
    )
  const dropped = (name: string, v: Value) => drop.apply({}, v, childContext(ctx, `${name}Dropout`))
  // Normalisations are tapped at `attentionNorm` and `feedForwardNorm`, the stream between the branches at `residual`.
  const norm1 = (h: Value) => tap(ctx, normalise(params.attentionNorm, h), 'attentionNorm')
  const norm2 = (h: Value) => tap(ctx, normalise(params.feedForwardNorm!, h), 'feedForwardNorm')
  const mid = (h: Value) => tap(ctx, h, 'residual')
  let out: Value
  if (parallel) {
    const h = norm1(x)
    out = add(add(x, dropped('attention', attend(h))), dropped('feedForward', ffn(h)))
  } else if (placement === 'pre') {
    const h = mid(add(x, dropped('attention', attend(norm1(x)))))
    out = add(h, dropped('feedForward', ffn(norm2(h))))
  } else {
    const h = mid(norm1(add(x, dropped('attention', attend(x)))))
    out = norm2(add(h, dropped('feedForward', ffn(h))))
  }
  return { output: tap(ctx, out), weights, cache: nextCache! }
}

/**
 * A transformer block over `[..., T, d_model]` as a layer (see `transformerBlock`); it keeps no cache. Norms start at
 * scale one and shift zero, and a T5 table is drawn from $\Gauss(0, 0.02^2)$. Its attention weights are tapped at
 * `<path>.attention.weights`.
 *
 * @param dModel The model width $d_{\mathrm{model}}$, of input and output.
 * @param options The attention, normalisation, feed-forward and position options.
 * @returns The layer: `init` draws `TransformerBlockParams`; `apply` runs `transformerBlock` from position 0.
 *
 * @example An RMS-norm SwiGLU block with T5 biases on a batch of two sequences of three tokens
 * const layer = TransformerBlock(4, { heads: 2, causal: true, norm: 'rms', feedForward: 'swiglu', position: 't5' })
 * const params = layer.init(stream(0))
 * print(layer.label)
 * print('parameters:', Object.keys(params))
 * print('T5 table:', shapeOf(params.positionBias))
 * print('output:', shapeOf(layer.apply(params, normals(stream(1), [2, 3, 4]))))
 */
export function TransformerBlock(dModel: Size, options: TransformerBlockOptions): Layer<TransformerBlockParams> {
  const { placement = 'pre', parallel = false, position = 'none', heads, norm = 'layer' } = options
  const attention = MultiHeadAttention(dModel, {
    heads,
    kvHeads: options.kvHeads,
    headDim: options.headDim,
    qkNorm: options.qkNorm,
    projectionBias: options.bias ?? true,
  })
  const ff = FeedForward(dModel, { kind: options.feedForward, hidden: options.hidden, bias: options.bias })
  const normInit = (): NormParams =>
    norm === 'rms' ? { gamma: ones([dModel]) } : { gamma: ones([dModel]), beta: zeros([dModel]) }
  const buckets = options.t5?.buckets ?? 32
  return {
    kind: 'TransformerBlock',
    label: `TransformerBlock(${dModel}, ${heads} heads, ${parallel ? 'parallel' : `${placement}-norm`}${
      position === 'none' ? '' : `, ${position}`
    })`,
    init: (s: Stream) => ({
      attentionNorm: normInit(),
      attention: attention.init(child(s, 'attention')),
      ...(parallel ? {} : { feedForwardNorm: normInit() }),
      feedForward: ff.init(child(s, 'feedForward')),
      ...(position === 't5'
        ? {
            positionBias: normalInit(0.02)(child(s, 'positionBias'), [buckets, heads], {
              fanIn: buckets,
              fanOut: heads,
            }),
          }
        : {}),
    }),
    apply: (p, x, ctx) => transformerBlock(p, x, options, {}, ctx).output,
  }
}
