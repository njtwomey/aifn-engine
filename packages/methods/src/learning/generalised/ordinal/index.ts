/**
 * `aifn-methods/learning/generalised/ordinal`: ordinal regression. Latent-variable models (cumulative-link,
 * continuation-ratio and adjacent-category, on `aifn-compute/probability/likelihoods`' ordinal likelihoods), threshold losses
 * and their linear model (all-threshold and immediate-threshold), Frank and Hall's binary decomposition, and deep
 * ordinal heads (CORAL and cumulative link) on an MLP. GP ordinal regression is `aifn-methods/learning/gaussian-processes`'
 * `gpOrdinalRegression`.
 */

export { ordinalRegression, type OrdinalRegressionModel, type OrdinalRegressionParams } from './ordinal'
export {
  allThresholdLoss,
  immediateThresholdLoss,
  thresholdClasses,
  thresholdOrdinalRegression,
  thresholdPenalty,
  type ThresholdConstruction,
  type ThresholdLossOptions,
  type ThresholdOrdinalRegressionModel,
  type ThresholdOrdinalRegressionParams,
  type ThresholdPenalty,
} from './thresholds'
export {
  binaryDecomposition,
  differenceExceedance,
  type BinaryBase,
  type BinaryDecompositionModel,
  type BinaryDecompositionParams,
} from './decomposition'
export {
  deepOrdinalRegression,
  type DeepOrdinalHead,
  type DeepOrdinalRegressionModel,
  type DeepOrdinalRegressionParams,
} from './deep'
export { ordinalFunctions } from './registry'
