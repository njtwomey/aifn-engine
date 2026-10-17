/**
 * `aifn-methods/neural/sequence`: worked studies of efficient sequence models, built on the layers of
 * `aifn-compute/nn/sequence` and `aifn-compute/nn/attention`.
 *
 * - Recall: `multiQueryAssociativeRecall` (MQAR, also re-exported by `aifn-methods/data/synthetic`) and
 *   `associativeRecallByDimension`, the accuracy of softmax attention, linear attention and the delta rule with ideal
 *   projections as the key width grows.
 * - State tracking at finite precision: `parityAccuracyByLength`, the best parity accuracy of a one-state linear
 *   recurrence stored in fixed point, with `roundToGrid`, `parityStates`, `stateValues` and `onesUntilZero`.
 * - Low-rank and hashing attention as explicit matrices: `lowRankSeries` and `lowRankAt` (truncated SVD),
 *   `linformerMatrix` with `segmentMeans`, `nystromMatrix`, and `lshAttentionMatrix` with `angularBuckets` and
 *   `angularRotations`, compared by `rowTotalVariation` on `topicTokens`.
 * - Long convolutions: `convolutionMultiplies`, the multiply counts of a causal convolution directly and by FFT.
 *
 * Every function is a plain function of serialisable inputs (streams, numbers, arrays) returning plain data, so a
 * figure can run the slower ones (the recall probe, the low-rank curves) in the compute worker.
 */

export {
  associativeRecallByDimension,
  multiQueryAssociativeRecall,
  type MqarExample,
  type MqarOptions,
  type RecallProbeOptions,
  type RecallProbeResult,
} from './recall'
export {
  onesUntilZero,
  parityAccuracyByLength,
  parityStates,
  roundToGrid,
  stateValues,
  type ParityAccuracyOptions,
  type ParityReadout,
} from './parity'
export {
  angularBuckets,
  angularRotations,
  linformerMatrix,
  lowRankAt,
  lowRankSeries,
  lshAttentionMatrix,
  nystromMatrix,
  rowTotalVariation,
  segmentMeans,
  topicTokens,
  type LowRankSeries,
} from './approximate'
export { convolutionMultiplies } from './cost'
export { efficientSequenceFunctions } from './registry'
