/**
 * `aifn-compute/nn/functional`: the functional layer of neural networks (after torch.nn.functional), stateless
 * operations on tensors.
 *
 * - Activations, elementwise: `relu` (a primitive), `leakyRelu`, `elu`, `gelu` (exact or tanh form), `silu` and
 *   `identity`. `activationFn` turns an `Activation` (a name in `activationFunctions`, or a function) into a function,
 *   for layers configured by name.
 * - Convolution: `conv1d` and `conv2d`, cross-correlations with stride, padding, dilation and channel groups;
 *   `convOutputSize` gives the output length.
 * - Pooling: `avgPool1d`, `avgPool2d` (padded zeros count in the mean) and `maxPool1d`, `maxPool2d` (padding reads
 *   $-\infty$), non-overlapping by default.
 *
 * Inputs are channels-first, `[N, C, H, W]` or `[N, C, L]`, and the batch axis may be left out. Every operation is a
 * primitive or a composition of primitives, so it differentiates to any order and batches under `vmap`.
 */

export {
  avgPool1d,
  avgPool2d,
  conv1d,
  conv2d,
  convOutputSize,
  maxPool1d,
  maxPool2d,
  type ConvOptions,
  type Pair,
  type PoolOptions,
} from './ops'
export {
  activationFn,
  activations as activationFunctions,
  elu,
  gelu,
  identity,
  leakyRelu,
  relu,
  silu,
  type Activation,
  type ActivationName,
} from './activations'
