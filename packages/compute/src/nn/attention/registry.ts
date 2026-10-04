/**
 * The functions, layers and algorithms of `aifn-compute/nn/attention`, registered with the notes that define them. Layers are
 * registered as constructions (they build a layer), functional forms as transforms.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as attention from './attention'
import * as block from './block'
import * as cache from './cache'
import * as feedforward from './feedforward'
import * as flash from './flash'
import * as masks from './masks'
import * as positions from './positions'

const fn = definer<FunctionInfo>('function', 'nn/attention')
const algorithm = definer<AlgorithmInfo>('algorithm', 'nn/attention')

fn(
  {
    key: 'scaledDotProductAttention',
    name: 'Scaled dot-product attention',
    tex: '\\operatorname{softmax}(QK^\\top/\\sqrt{d_k} + B)V',
    role: 'transform',
    summary: 'softmax(QKᵀ/√d_k + B)·V with masks, additive position biases and logit soft-capping.',
    notes: ['scaled-dot-product-attention', 'causal-masking', 'query-key-normalisation-and-logit-soft-capping'],
    cite: ['vaswani2017', 'gemmateam2024'],
  },
  attention.scaledDotProductAttention,
)
fn(
  {
    key: 'multiHeadAttention',
    name: 'Multi-head attention',
    role: 'transform',
    summary:
      'h attention heads over projected queries, keys and values, with shared key–value heads, QK-norm, RoPE and a cache.',
    notes: ['multi-head-attention', 'multi-query-and-grouped-query-attention', 'key-value-cache'],
    cite: ['vaswani2017', 'shazeer2019', 'ainslie2023', 'henry2020'],
  },
  attention.multiHeadAttention,
)
fn(
  {
    key: 'MultiHeadAttention',
    name: 'Multi-head attention layer',
    role: 'construction',
    summary: 'A multi-head self-attention layer with optional grouped-query heads, QK-norm and rotary positions.',
    notes: ['multi-head-attention', 'multi-query-and-grouped-query-attention'],
    cite: ['vaswani2017', 'ainslie2023'],
  },
  attention.MultiHeadAttention,
)
fn(
  {
    key: 'multiHeadLatentAttention',
    name: 'Multi-head latent attention',
    role: 'transform',
    summary: 'Keys and values decompressed from one cached low-rank latent per token, with a decoupled rotary key.',
    notes: ['multi-head-latent-attention'],
    cite: ['deepseek2024v2'],
  },
  attention.multiHeadLatentAttention,
)
fn(
  {
    key: 'MultiHeadLatentAttention',
    name: 'Multi-head latent attention layer',
    role: 'construction',
    notes: ['multi-head-latent-attention'],
    cite: ['deepseek2024v2'],
  },
  attention.MultiHeadLatentAttention,
)
fn(
  {
    key: 'softCap',
    name: 'Logit soft-capping',
    tex: 'c\\tanh(x/c)',
    role: 'transform',
    summary: 'c·tanh(x/c): scores pass through near zero and are bounded by ±c.',
    notes: ['query-key-normalisation-and-logit-soft-capping'],
    cite: ['gemmateam2024'],
  },
  attention.softCap,
)
fn(
  {
    key: 'positionMask',
    name: 'Attention mask from positions',
    role: 'construction',
    summary: 'Which keys each query sees: causal, sliding-window, or both, from absolute positions.',
    notes: ['causal-masking', 'sliding-window-and-sparse-attention'],
    cite: ['child2019', 'beltagy2020'],
  },
  masks.positionMask,
)
fn({ key: 'causalMask', name: 'Causal mask', role: 'construction', notes: ['causal-masking'] }, masks.causalMask)
fn(
  {
    key: 'slidingWindowMask',
    name: 'Sliding-window mask',
    role: 'construction',
    notes: ['sliding-window-and-sparse-attention'],
    cite: ['beltagy2020', 'jiang2023mistral'],
  },
  masks.slidingWindowMask,
)
fn(
  { key: 'paddingMask', name: 'Padding mask', role: 'construction', notes: ['scaled-dot-product-attention'] },
  masks.paddingMask,
)
fn(
  {
    key: 'sinusoidalPositions',
    name: 'Sinusoidal positional encoding',
    role: 'construction',
    summary: 'PE[p, 2i] = sin(p·base^(−2i/d)), PE[p, 2i + 1] = cos(·), added to the token embeddings.',
    notes: ['sinusoidal-positional-encoding', 'positional-encoding'],
    cite: ['vaswani2017'],
  },
  positions.sinusoidalPositions,
)
fn(
  {
    key: 'LearnedPositions',
    name: 'Learned position embeddings',
    role: 'construction',
    notes: ['learned-absolute-position-embeddings', 'positional-encoding'],
    cite: ['radford2019'],
  },
  positions.LearnedPositions,
)
fn(
  {
    key: 'applyRope',
    name: 'Rotary position embedding',
    tex: 'R_{p\\theta}\\,x',
    role: 'transform',
    summary:
      'Rotate each coordinate pair of queries and keys by p·θ_i, so their dot product depends on the offset only.',
    notes: ['rotary-position-embedding', 'positional-encoding'],
    cite: ['su2021'],
  },
  positions.applyRope,
)
fn(
  {
    key: 'ropeFrequencies',
    name: 'Rotary frequencies and their rescalings',
    role: 'construction',
    summary: 'θ_i = base^(−2i/d), rescaled by position interpolation, NTK-aware base scaling or YaRN.',
    notes: ['rotary-position-embedding-variants', 'long-context-methods'],
    cite: ['su2021', 'chen2023', 'peng2024yarn'],
  },
  positions.ropeFrequencies,
)
fn(
  {
    key: 'alibiBias',
    name: 'ALiBi',
    role: 'construction',
    summary: 'A fixed bias −m_h·|p − q| per head added to attention scores, with geometric slopes m_h.',
    notes: ['alibi', 'positional-encoding'],
    cite: ['press2022'],
  },
  positions.alibiBias,
)
fn(
  {
    key: 't5RelativeBias',
    name: 'T5 relative position bias',
    role: 'transform',
    summary:
      'A learned scalar per head and bucket of relative distance, exact near the query and logarithmic far away.',
    notes: ['t5-relative-position-bias', 'relative-position-representations'],
    cite: ['raffel2020'],
  },
  positions.t5RelativeBias,
)
fn(
  {
    key: 'feedForward',
    name: 'Position-wise feed-forward network',
    role: 'transform',
    summary: 'down(act(up(x))), or gated: down(act(gate(x)) ⊙ up(x)) for SwiGLU, GeGLU and ReGLU.',
    notes: ['gated-feed-forward-layers'],
    cite: ['vaswani2017', 'shazeer2020'],
  },
  feedforward.feedForward,
)
fn(
  {
    key: 'FeedForward',
    name: 'Feed-forward layer',
    role: 'construction',
    notes: ['gated-feed-forward-layers'],
    cite: ['shazeer2020'],
  },
  feedforward.FeedForward,
)
fn(
  {
    key: 'transformerBlock',
    name: 'Transformer block',
    role: 'transform',
    summary: 'Attention and a feed-forward network in residual branches, pre-norm, post-norm or parallel.',
    notes: ['pre-norm-and-post-norm', 'modern-decoder-block', 'parallel-attention-and-feed-forward', 'transformer'],
    cite: ['vaswani2017', 'xiong2020', 'wang2021gptj'],
  },
  block.transformerBlock,
)
fn(
  {
    key: 'TransformerBlock',
    name: 'Transformer block layer',
    role: 'construction',
    notes: ['pre-norm-and-post-norm', 'modern-decoder-block', 'rmsnorm-in-transformers'],
    cite: ['vaswani2017', 'xiong2020', 'zhang2019'],
  },
  block.TransformerBlock,
)
fn(
  {
    key: 'kvCacheMemory',
    name: 'Key–value cache memory',
    role: 'property',
    summary:
      'Bytes per token and in total of a decoder’s key–value cache under MHA, GQA, MQA, latent attention and windows.',
    notes: ['key-value-cache', 'multi-query-and-grouped-query-attention', 'multi-head-latent-attention'],
    cite: ['pope2022', 'ainslie2023', 'deepseek2024v2'],
  },
  cache.kvCacheMemory,
)
fn(
  {
    key: 'flashAttention',
    name: 'Tiled attention with the online softmax',
    role: 'transform',
    notes: ['flash-attention'],
    cite: ['dao2022', 'milakov2018', 'rabe2021'],
  },
  flash.flashAttention,
)
algorithm(
  {
    key: 'flashAttentionSteps',
    name: 'FlashAttention forward pass',
    summary:
      'Attention one tile of queries × keys at a time, folding each into a running maximum, normaliser and output.',
    problem: 'sequence',
    state: { iterate: 'accumulator', flags: ['terminated'] },
    notes: ['flash-attention'],
    cite: ['dao2022', 'milakov2018'],
  },
  flash.flashAttentionSteps,
)

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>

/** Every function and layer constructor of the module, keyed by name. */
export const attentionFunctions = entries<FunctionInfo>(
  'function',
  attention,
  block,
  cache,
  feedforward,
  flash,
  masks,
  positions,
) as Table<FunctionInfo>

/** The step-through algorithms of the module. */
export const attentionAlgorithms = entries<AlgorithmInfo>('algorithm', flash) as Table<AlgorithmInfo>
