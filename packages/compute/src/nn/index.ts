/**
 * `aifn-compute/nn`: neural networks, after torch.nn: functional operations, initialisers, layers and the training loop. Losses
 * are `aifn-compute/learning/losses` (decision D6) and optimisers the pytree update rules of `aifn-compute/optim/first-order`.
 * Children: functional, init, layers, attention, sequence, decoding, training, experts, graph, quantise.
 */

export { relu, gelu, conv2d } from './functional'
export { xavierUniform, heNormal } from './init'
export { Linear, Mlp, Sequential } from './layers'
export { MultiHeadAttention, TransformerBlock } from './attention'
export { trainingLoop } from './training'
