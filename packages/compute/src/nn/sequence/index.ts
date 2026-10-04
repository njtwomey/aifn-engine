/**
 * `aifn-compute/nn/sequence`: sequence layers beyond the recurrent cells of `aifn-compute/nn/layers`. Backpropagation through time with
 * the gradient norm at every step (and truncation); linear state-space layers: diagonal discretisation (the dense one
 * is `aifn-compute/systems` `discretiseSsm`), HiPPO-LegS, the S4 convolution kernel, the recurrent mode by a parallel scan,
 * and the causal convolution; Mamba's selective scan and layer; linear attention in its parallel and recurrent forms (with retention's
 * decay); and Bahdanau and Luong attention for recurrent encoder–decoders.
 */

export { gradientsThroughTime, type ThroughTime, type ThroughTimeOptions } from './gradients'
export {
  causalConvolution,
  discretiseDiagonal,
  hippoLegS,
  linearRecurrence,
  matrixRecurrence,
  ssmKernel,
  ssmRecurrent,
  type RecurrenceOptions,
} from './ssm'
export {
  SelectiveSsm,
  selectiveScan,
  type SelectiveScanOptions,
  type SelectiveScanResult,
  type SelectiveSsmOptions,
  type SelectiveSsmParams,
} from './selective'
export {
  eluFeatureMap,
  linearAttention,
  linearAttentionRecurrent,
  type LinearAttentionOptions,
  type LinearAttentionStates,
} from './linear-attention'
export {
  BahdanauAttention,
  LuongAttention,
  type AlignmentResult,
  type BahdanauParams,
  type LuongParams,
  type LuongScore,
  type SequenceAttention,
} from './seq2seq'
export { sequenceFunctions } from './registry'
