/**
 * `aifn-methods/learning/generalised/ordinal`: regression on ordered classes $0, \dots, K - 1$.
 *
 * - Latent-variable models: `ordinalRegression`, the cumulative-link (proportional odds with the logit link, as R's
 *   `MASS::polr` and statsmodels' `OrderedModel`), continuation-ratio and adjacent-category models on
 *   `aifn-compute/probability/likelihoods`' ordinal likelihoods, fitted by L-BFGS: a slope vector $\betavec$ and
 *   $K - 1$ thresholds $\thetavec$.
 * - Threshold losses: `allThresholdLoss` (bounds the absolute error) and `immediateThresholdLoss` (bounds the zero-one
 *   error) with the margin penalties of `thresholdPenalty`, their linear model `thresholdOrdinalRegression` (not
 *   probabilistic), and `thresholdClasses`, the class $\#\{k : \theta_k < s\}$ of a score.
 * - Frank and Hall's binary decomposition: `binaryDecomposition`, $K - 1$ classifiers of $\pr(y > k)$ differenced
 *   into class probabilities by `differenceExceedance`.
 * - Deep ordinal heads: `deepOrdinalRegression`, CORAL or a cumulative link on an MLP score, trained by Adam.
 * - The registry: `ordinalFunctions`.
 *
 * Labels are class indices; the number of classes defaults to the largest label plus one. A higher score or linear
 * predictor means a higher class in every model. Labels that are not class indices throw `DomainError`; inputs and
 * labels that differ in number throw `ShapeError`. GP ordinal regression is `aifn-methods/learning/gaussian-processes`'
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
