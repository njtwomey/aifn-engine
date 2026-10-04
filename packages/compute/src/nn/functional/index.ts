/**
 * `aifn-compute/nn/functional`: the functional layer of neural networks (after torch.nn.functional): activations (`relu`,
 * `gelu`, `silu`, `elu`, `leakyRelu`, by name through `activationFn`), 1-D and 2-D convolution and pooling. Every
 * operation is a primitive or a composition of primitives, so it differentiates.
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
