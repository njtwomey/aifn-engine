/**
 * The transformer block: self-attention and a feed-forward network, each in a residual branch with normalisation.
 * Post-norm, norm(x + f(x)), is the original placement (Vaswani et al., 2017); pre-norm, x + f(norm(x)), keeps an
 * identity path from input to output and trains deep stacks without warm-up (Xiong et al., 2020); the parallel form
 * x + attention(norm(x)) + ffn(norm(x)) runs both branches from one normalisation (Wang and Komatsuzaki, 2021, GPT-J;
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

/** Relative position schemes applied inside a block. */
export type RelativePosition = 'none' | 'rope' | 'alibi' | 't5'

/** Options of `TransformerBlock` and `transformerBlock`. */
export type TransformerBlockOptions = {
  heads: Size
  /** Key–value heads (grouped-query or multi-query attention); default `heads`. */
  kvHeads?: Size
  /** Width of each head (default d_model/heads). */
  headDim?: Size
  /** Causal self-attention (a decoder). Default false. */
  causal?: boolean
  /** A sliding attention window. */
  window?: Size
  /** `pre` (default): x + f(norm(x)); `post`: norm(x + f(x)). */
  placement?: 'pre' | 'post'
  /** Both branches from one normalisation, added together (pre-norm only). Default false. */
  parallel?: boolean
  /** `layer` (default) or `rms` normalisation. */
  norm?: 'layer' | 'rms'
  /** The feed-forward kind (default `mlp`). */
  feedForward?: FeedForwardKind
  /** Feed-forward hidden width. */
  hidden?: Size
  /** Activation of a plain feed-forward network (default GELU). */
  activation?: 'gelu' | 'relu' | 'silu' | 'tanh'
  /** The relative position scheme (default `none`: positions come from the embedding, or nowhere, NoPE). */
  position?: RelativePosition
  rope?: RopeOptions
  t5?: T5BucketOptions
  /** Query–key RMS normalisation. Default false. */
  qkNorm?: boolean
  /** Soft-cap attention scores at ±cap. */
  softCap?: number
  /** Dropout on each residual branch while training (default 0). */
  dropout?: number
  /** Biases on projections (default true). */
  bias?: boolean
}

/** Parameters of a transformer block. */
export type TransformerBlockParams = {
  attentionNorm: NormParams
  attention: MultiHeadAttentionParams
  feedForwardNorm?: NormParams
  feedForward: FeedForwardParams
  /** T5's bias table [buckets, heads], with `position: 't5'`. */
  positionBias?: Tensor
}

/** A block's output, its attention weights [..., h, T, Tk] and its extended cache. */
export type BlockResult = { output: Value; weights: Value; cache: KvCache }

/**
 * One transformer block applied to x [..., T, d_model], continuing from `state` (positions and the key–value cache of
 * earlier tokens), as `multiHeadAttention` does. With a tapping context it records, below its path: `attentionNorm`
 * and `feedForwardNorm` (the normalised inputs of the branches), `attention.weights` [..., h, T, Tk], `attention` (the
 * branch output), `residual` (the stream between the branches), `feedForward.hidden`, `feedForward`, and the block's
 * output at the path itself.
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
 * A transformer block over [..., T, d_model] as a layer (see `transformerBlock`). Its attention weights are tapped at
 * `<path>.attention.weights`.
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
