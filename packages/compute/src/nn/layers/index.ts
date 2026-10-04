/**
 * `aifn-compute/nn/layers`: layers as parameter trees with a forward function (after torch.nn.modules): `Linear`, `Embedding`,
 * convolutions and pools, `LayerNorm`, `RmsNorm`, `BatchNorm`, `Dropout`, `Sequential`, `Mlp`, `Residual`, and RNN, GRU
 * and LSTM cells, the ODE block (`OdeBlock`, a neural ODE layer), and multiple-instance pooling (`MilPooling`: embedding,
 * attention, instance, additive and conjunctive, after MILLET). Attention is `aifn-compute/nn/attention`.
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

// For writing layers: record an activation (`tap`) and name a sub-layer's context (`childContext`).
export { childContext, tap } from './layers'
