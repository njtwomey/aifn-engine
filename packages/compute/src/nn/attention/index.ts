/**
 * `aifn-compute/nn/attention`: attention and the transformer block. Scaled dot-product attention with causal, sliding-window
 * and padding masks, additive biases and soft-capping; multi-head attention with multi-query and grouped-query heads,
 * QK-norm, rotary positions and a key–value cache; multi-head latent attention; positional encodings (sinusoidal,
 * learned, RoPE with its rescalings, ALiBi, T5 buckets; NoPE is none of them); plain and gated feed-forward layers
 * (SwiGLU, GeGLU, ReGLU); the pre-norm, post-norm and parallel transformer block; key–value cache accounting; and the
 * tiled online-softmax forward pass of FlashAttention as a step-through algorithm.
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
