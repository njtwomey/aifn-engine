/**
 * `aifn-compute/nn`: neural networks, after torch.nn, as parameter trees with pure forward functions.
 *
 * - `aifn-compute/nn/functional`: stateless operations: activations, 1-D and 2-D convolution and pooling.
 * - `aifn-compute/nn/init`: weight initialisers (Xavier, He, LeCun, normal, zeros) that draw from a stream.
 * - `aifn-compute/nn/layers`: layers (`Linear`, convolutions, normalisation, dropout, `Sequential`, `Mlp`), recurrent
 *   cells, multiple-instance pooling and the neural ODE block.
 * - `aifn-compute/nn/attention`: attention and the transformer block, with masks, positional encodings, key–value
 *   caches and FlashAttention.
 * - `aifn-compute/nn/sequence`: sequence layers beyond the recurrent cells: state-space models (S4, Mamba), parallel
 *   scans, linear attention and encoder–decoder attention.
 * - `aifn-compute/nn/decoding`: decoding a language model token by token: greedy, sampling, beam search and
 *   speculative decoding, with logit processors.
 * - `aifn-compute/nn/training`: training loops as traceable algorithms: minibatch first-order, full-batch L-BFGS,
 *   adversarial, contrastive divergence and DP-SGD, and activation and gradient inspection.
 * - `aifn-compute/nn/experts`: mixtures of experts, with routing, capacity and the auxiliary balancing losses.
 * - `aifn-compute/nn/graph`: graph neural network layers (graph convolution, graph attention, GraphSAGE, message
 *   passing).
 * - `aifn-compute/nn/quantise`: $b$-bit quantisation of tensors and weights, after torch.ao.quantization.
 *
 * Losses are `aifn-compute/learning/losses` (decision D6) and optimisers the pytree update rules of
 * `aifn-compute/optim/first-order`. This index re-exports the most used names: `relu`, `gelu`, `conv2d`,
 * `xavierUniform`, `heNormal`, `Linear`, `Mlp`, `Sequential`, `MultiHeadAttention`, `TransformerBlock` and
 * `trainingLoop`.
 */

export { relu, gelu, conv2d } from './functional'
export { xavierUniform, heNormal } from './init'
export { Linear, Mlp, Sequential } from './layers'
export { MultiHeadAttention, TransformerBlock } from './attention'
export { trainingLoop } from './training'
