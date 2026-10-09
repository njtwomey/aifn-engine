/**
 * `aifn-compute/nn/sequence`: sequence layers beyond the recurrent cells of `aifn-compute/nn/layers`.
 *
 * - Training recurrent networks: `gradientsThroughTime`, backpropagation through time with
 *   $\lVert \partial L / \partial \hvec_t \rVert$ at every step (to see gradients vanish or explode), and truncation.
 * - Linear state-space layers (S4): `discretiseDiagonal` for a diagonal system (the dense one is `discretiseSsm` of
 *   `aifn-compute/systems`), `hippoLegS` for the HiPPO matrices, and the two equivalent modes: `ssmKernel` with
 *   `causalConvolution`, or `ssmRecurrent` by a parallel scan.
 * - Parallel scans of linear recurrences, every step at once: `linearRecurrence` (elementwise,
 *   $\hvec_t = \avec_t \odot \hvec_{t-1} + \bvec_t$) and `matrixRecurrence`
 *   ($\hvec_t = \Amat_t \hvec_{t-1} + \bvec_t$).
 * - Mamba: `selectiveScan`, the input-dependent diagonal recurrence, and `SelectiveSsm`, the layer around it.
 * - Linear attention: `linearAttention` (the parallel form, optionally non-causal) and `linearAttentionRecurrent` (the
 *   recurrent form with every state $\Smat_t$), with retention's decay $\gamma$ and the feature map `eluFeatureMap`.
 * - Attention for recurrent encoder–decoders: `BahdanauAttention` (additive) and `LuongAttention` (`dot`, `general` or
 *   `concat` scores), each with `init` and `attend`.
 *
 * The recurrences, the state-space functions and `gradientsThroughTime` take sequences time first, $[T, \dots]$;
 * `selectiveScan` and `linearAttention` take $[\dots, T, d]$, batch axes in front. Everything is built from
 * differentiable primitives, so every layer trains by `grad`.
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
