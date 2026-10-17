/**
 * `aifn-compute/nn/functional`: the functional layer of neural networks (after torch.nn.functional), stateless
 * operations on tensors.
 *
 * - Activations, elementwise: `relu` (a primitive), `leakyRelu`, `elu`, `gelu` (exact or tanh form), `silu` and
 *   `identity`. `activationFn` turns an `Activation` (a name in `activationFunctions`, or a function) into a function,
 *   for layers configured by name.
 * - Convolution: `conv1d` and `conv2d`, cross-correlations with stride, padding, dilation and channel groups;
 *   `convOutputSize` gives the output length. `im2col` lays every window out as a column, so a layer is one matrix
 *   product; `deformableConv2d` reads its taps at learned offsets by bilinear interpolation (DCNv1 and, with a mask,
 *   DCNv2).
 * - Pooling: `avgPool1d`, `avgPool2d` (padded zeros count in the mean) and `maxPool1d`, `maxPool2d` (padding reads
 *   $-\infty$), non-overlapping by default.
 *
 * - Convolution arithmetic, on geometry alone: `convolutionCost` (parameters, multiply-accumulates and output shape
 *   of an ordinary, grouped, depthwise or transposed layer), `convolutionStackCost` for a chain, and `networkCost` and
 *   `layerCost` for a whole network given as a tree of `LayerSpec`s (sums, concatenations, gates), with its depth and
 *   receptive field.
 * - Receptive fields: `receptiveField` (size, jump and start after each layer, along one axis),
 *   `effectiveReceptiveField` (the gradient of one output unit), and `dilatedTapCoverage` and `meanValidTaps` (how
 *   many of a dilated kernel's taps land inside a padded map).
 *
 * Inputs are channels-first, `[N, C, H, W]` or `[N, C, L]`, and the batch axis may be left out. Every tensor operation
 * is a primitive or a composition of primitives, so it differentiates to any order and batches under `vmap`
 * (`deformableConv2d` is not batched over its offsets). `functionalFunctions` is the module's registry.
 */

export {
  avgPool1d,
  avgPool2d,
  conv1d,
  conv2d,
  convOutputSize,
  im2col,
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
export {
  convolutionCost,
  convolutionStackCost,
  layerCost,
  networkCost,
  type AddLayer,
  type ChainedLayerCost,
  type ChainedLayerSpec,
  type ChannelScaleLayer,
  type ConcatLayer,
  type ConvLayer,
  type ConvolutionCost,
  type ConvolutionKind,
  type ConvolutionLayerSpec,
  type DenseLayer,
  type DepthwiseLayer,
  type ElementwiseLayer,
  type FieldSpan,
  type GateLayer,
  type GlobalPoolLayer,
  type LayerCost,
  type LayerSpec,
  type NetworkCost,
  type NormLayer,
  type PerAxis,
  type PoolLayer,
  type SequenceLayer,
  type TransposedLayer,
} from './cost'
export {
  dilatedTapCoverage,
  effectiveReceptiveField,
  meanValidTaps,
  receptiveField,
  type ReceptiveFieldLayer,
  type ReceptiveFieldStep,
} from './receptive'
export { deformableConv2d, type DeformableConvOptions } from './deformable'
export { functionalFunctions } from './registry'
