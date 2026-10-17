/**
 * `aifn-compute/nn/layers`: layers as parameter trees with a pure forward function, after torch.nn.modules.
 *
 * - Dense and convolutional: `Linear`, `Embedding`, `Conv1d`, `Conv2d`, the pools `MaxPool1d`, `MaxPool2d`,
 *   `AvgPool1d`, `AvgPool2d`, `Flatten` and `ActivationLayer`.
 * - Normalisation and regularisation: `LayerNorm`, `RmsNorm`, `BatchNorm` (running statistics in `ctx.buffers`),
 *   `GroupNorm` and `InstanceNorm` (per example, over groups of channels and their positions), and `Dropout` (active in
 *   training only).
 * - Containers: `Sequential`, `Mlp` (Linear layers with an activation between them) and `Residual` ($x + f(x)$).
 * - Recurrent: the cells `RnnCell`, `GruCell` and `LstmCell`, in PyTorch's gate order, and `unrollRecurrent` to run
 *   one over a sequence.
 * - Multiple-instance pooling: `MilPooling` as a layer and `milPool` with its instance-level interpretation, in the
 *   five kinds of `MIL_POOLING_KINDS` (embedding, attention, instance, additive and conjunctive, after MILLET).
 * - The ODE block `OdeBlock`, a neural ODE layer whose output is the solution of its field at the end of an interval.
 * - The spectral convolution `SpectralConv1d` of a Fourier layer: the lowest Fourier modes of each channel mixed by
 *   learned complex matrices, resolution-independent (the Fourier neural operator's global convolution).
 * - Functional forms, without parameters of their own: `linear`, `layerNorm`, `rmsNorm`, `batchNorm`, `groupNorm`,
 *   `instanceNorm`, `dropout`. `layerFunctions` registers the group and instance normalisations and `SpectralConv1d`.
 * - For writing layers: `tap` records an activation, `childContext` names a sub-layer's context.
 *
 * A layer is a plain object: `init(stream)` draws its parameters and `apply(params, x, ctx)` gives its output, with
 * no hidden state, so `grad` of any function of the parameters differentiates through it. The `Context` carries the
 * training flag, the dropout stream, the tap and the non-trainable buffers. Attention is `aifn-compute/nn/attention`.
 */

export {
  ActivationLayer,
  AvgPool1d,
  AvgPool2d,
  BatchNorm,
  batchNorm,
  Conv1d,
  Conv2d,
  Dropout,
  dropout,
  Embedding,
  Flatten,
  GroupNorm,
  groupNorm,
  InstanceNorm,
  instanceNorm,
  LayerNorm,
  layerNorm,
  Linear,
  linear,
  MaxPool1d,
  MaxPool2d,
  Mlp,
  Residual,
  RmsNorm,
  rmsNorm,
  Sequential,
  type BatchNormBuffers,
  type BatchNormLayerOptions,
  type BatchNormOptions,
  type Buffers,
  type Context,
  type ConvLayerOptions,
  type ConvParams,
  type EmbeddingParams,
  type Layer,
  type LinearOptions,
  type LinearParams,
  type MlpOptions,
  type NormParams,
} from './layers'
export {
  GruCell,
  LstmCell,
  RnnCell,
  unrollRecurrent,
  type Cell,
  type CellOptions,
  type CellParams,
  type RecurrentState,
  type Unrolled,
} from './recurrent'
export {
  MIL_POOLING_KINDS,
  milPool,
  MilPooling,
  type MilPooled,
  type MilPoolingKind,
  type MilPoolingParams,
} from './mil'
export { OdeBlock, type OdeBlockLayer, type OdeBlockOptions } from './ode'
export { SpectralConv1d, type SpectralConvParams } from './spectral'
export { layerFunctions } from './registry'

// For writing layers: record an activation (`tap`) and name a sub-layer's context (`childContext`).
export { childContext, tap } from './layers'
