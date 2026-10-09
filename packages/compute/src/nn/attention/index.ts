/**
 * `aifn-compute/nn/attention`: attention and the transformer block, with masks, positional encodings, key–value caches
 * and the tiled FlashAttention forward pass.
 *
 * - Attention: `scaledDotProductAttention` (masks, additive biases, `softCap`); `multiHeadAttention` and the
 *   `MultiHeadAttention` layer, with multi-query and grouped-query heads (`kvHeads`), QK-norm, rotary positions and a
 *   key–value cache; `multiHeadLatentAttention` and `MultiHeadLatentAttention`, which cache one low-rank latent per
 *   token; the head plumbing `splitHeads`, `mergeHeads` and `repeatKvHeads`.
 * - Masks of ones and zeros over `[Tq, Tk]`, from absolute positions: `causalMask`, `slidingWindowMask`,
 *   `paddingMask`, and the general `positionMask`; `positionRange` and `continuePositions` give the positions.
 * - Positions: absolute ones added to the embeddings (`sinusoidalPositions`, `LearnedPositions` and
 *   `learnedPositions`), and relative ones inside attention: RoPE (`applyRope`, `ropeFrequencies` with position
 *   interpolation, NTK-aware scaling and YaRN, `ropeTables`), ALiBi (`alibiBias`, `alibiSlopes`) and T5 buckets
 *   (`t5RelativeBias`, `t5RelativeBucket`). NoPE is none of them.
 * - Blocks: `feedForward` and `FeedForward`, plain or gated (SwiGLU, GeGLU, ReGLU), with `feedForwardWidth`;
 *   `transformerBlock` and `TransformerBlock`, pre-norm, post-norm or parallel.
 * - Caches: `appendKvCache`, `appendLatentCache`, `trimKvCache` (a rolling window) and `cacheLength`; `kvCacheMemory`
 *   counts the bytes of a decoder's cache under each layout.
 * - FlashAttention: `flashAttentionSteps`, the tiled online-softmax forward pass of one head as a step-through
 *   algorithm, and `flashAttention`, run to the end.
 *
 * Sequences are `[..., T, d]` with tokens on the second-to-last axis and heads split out in front, `[..., h, T, d_h]`;
 * leading axes broadcast. Positions are absolute token indices, so masks and biases stay right when a call continues
 * from a cache, and decoding token by token gives the outputs of one causal pass. Everything is differentiable except
 * the FlashAttention pass, which computes on concrete numbers; attention weights are returned and tapped for figures.
 * `attentionFunctions` and `attentionAlgorithms` are the registry entries.
 */

export {
  mergeHeads,
  multiHeadAttention,
  MultiHeadAttention,
  multiHeadLatentAttention,
  MultiHeadLatentAttention,
  repeatKvHeads,
  scaledDotProductAttention,
  softCap,
  splitHeads,
  type AttentionOptions,
  type AttentionResult,
  type AttentionState,
  type CachedAttentionResult,
  type LatentAttentionLayerOptions,
  type LatentAttentionOptions,
  type LatentAttentionParams,
  type LatentAttentionResult,
  type MultiHeadAttentionParams,
  type MultiHeadLayerOptions,
  type MultiHeadOptions,
} from './attention'
export {
  appendKvCache,
  appendLatentCache,
  cacheLength,
  kvCacheMemory,
  trimKvCache,
  type KvCache,
  type KvCacheLayout,
  type KvCacheMemory,
  type LatentCache,
} from './cache'
export {
  causalMask,
  continuePositions,
  paddingMask,
  positionMask,
  positionRange,
  slidingWindowMask,
  type MaskOptions,
} from './masks'
export {
  alibiBias,
  alibiSlopes,
  applyRope,
  LearnedPositions,
  learnedPositions,
  ropeFrequencies,
  ropeTables,
  sinusoidalPositions,
  t5RelativeBias,
  t5RelativeBucket,
  type LearnedPositionParams,
  type PositionScheme,
  type RopeFrequencies,
  type RopeOptions,
  type RopeScaling,
  type T5BucketOptions,
} from './positions'
export {
  FeedForward,
  feedForward,
  feedForwardWidth,
  type FeedForwardKind,
  type FeedForwardLayerOptions,
  type FeedForwardOptions,
  type FeedForwardParams,
} from './feedforward'
export {
  TransformerBlock,
  transformerBlock,
  type BlockResult,
  type RelativePosition,
  type TransformerBlockOptions,
  type TransformerBlockParams,
} from './block'
export {
  flashAttention,
  flashAttentionSteps,
  type AttentionTile,
  type FlashAttentionOptions,
  type FlashAttentionState,
} from './flash'
export { attentionAlgorithms, attentionFunctions } from './registry'
