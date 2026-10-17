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
 *   recurrent form with every state $\Smat_t$), with retention's decay $\gamma$ and the feature maps `eluFeatureMap`
 *   and `positiveRandomFeatures` (the Performer's, unbiased for the softmax kernel); `linearAttentionWeights` gives
 *   the $T \times T$ weights themselves.
 * - Fast-weight and gated recurrences: `deltaRule` (DeltaNet and Gated DeltaNet, one correction step per token, state
 *   $d_v \times d_k$), the xLSTM matrix memory `mlstmRecurrent` and `mlstmParallel` (equal forms, stabilised), and
 *   Griffin's `rgLru` with `rgLruLambda`.
 * - The linear recurrent unit: `lruRingInit`, `lruEigenvalues` and `lruLambda` for the eigenvalues on a ring, run by
 *   `lruScan` (a complex scan) or `lruConvolve` (with `lruKernel`), which agree.
 * - HiPPO memories: `hippoLegT` beside `hippoLegS`, `hippoOnline` (either measure over a sampled input),
 *   `hippoProject` (the projection it approximates), `hippoBasis`, `hippoWindow` and `hippoReconstruct`.
 * - Long convolutions (Hyena): `implicitFilter` from `hyenaPositionalEncoding`, and `hyenaOperator`, gated long
 *   convolutions by FFT (`causalConvolution` with `method: 'fft'`).
 * - State space duality (Mamba-2): `semiseparableMask`, and the output by `ssdQuadratic`, `ssdRecurrent` or the chunked
 *   `ssdChunked`, which agree.
 * - Scans over grids: `scanOrder` (raster, column, snake, Hilbert), `routePositions`, `neighbourPairs`,
 *   `sequenceDistance` and `visibleTo`, for state-space models that read images.
 * - Attention for recurrent encoder–decoders: `BahdanauAttention` (additive) and `LuongAttention` (`dot`, `general` or
 *   `concat` scores), each with `init` and `attend`.
 *
 * The recurrences, the state-space functions and `gradientsThroughTime` take sequences time first, $[T, \dots]$;
 * `selectiveScan` and `linearAttention` take $[\dots, T, d]$, batch axes in front. Everything is built from
 * differentiable primitives, so every layer trains by `grad`, except the HiPPO runs and projections, which loop over
 * plain arrays for speed, and the scan orders, which are index lists.
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
  linearAttentionWeights,
  positiveRandomFeatures,
  type LinearAttentionOptions,
  type LinearAttentionStates,
} from './linear-attention'
export { deltaRule, type DeltaRuleOptions, type DeltaRuleResult } from './delta-rule'
export { mlstmParallel, mlstmRecurrent, type MlstmForget, type MlstmOptions, type MlstmResult } from './mlstm'
export { rgLru, rgLruLambda, type RgLruOptions, type RgLruParams, type RgLruResult } from './rg-lru'
export {
  lruConvolve,
  lruEigenvalues,
  lruKernel,
  lruLambda,
  lruRingInit,
  lruScan,
  type LruParams,
  type LruReadout,
  type LruRing,
} from './lru'
export {
  hippoBasis,
  hippoLegT,
  hippoOnline,
  hippoProject,
  hippoReconstruct,
  hippoWindow,
  type HippoMeasure,
  type HippoOptions,
} from './hippo'
export { hyenaOperator, hyenaPositionalEncoding, implicitFilter, type ImplicitFilterOptions } from './hyena'
export { semiseparableMask, ssdChunked, ssdQuadratic, ssdRecurrent, type SsdCost } from './ssd'
export {
  neighbourPairs,
  routePositions,
  scanOrder,
  sequenceDistance,
  visibleTo,
  type ScanOrderKind,
  type ScanRoute,
} from './scan-orders'
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
