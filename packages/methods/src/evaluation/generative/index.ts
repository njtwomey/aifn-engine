/**
 * `aifn-methods/evaluation/generative`: generative-model metrics (FID, KID, inception score, precision and recall),
 * computed from feature vectors the caller supplies.
 *
 * - Distribution distances: `fid` (the `frechetDistance` of the sets' Gaussian moments, biased with finite samples)
 *   and `kid` (an unbiased squared MMD with a `PolynomialKernel`), with `kidSubsets` averaging KID over random
 *   subsets as practice does for large sets.
 * - Class probabilities: `inceptionScore`, and `inceptionScoreSplits` for the mean and spread over splits.
 * - Fidelity and diversity, from $k$-nearest-neighbour balls: `generativePrecisionRecall`, and `densityCoverage`,
 *   which is more robust to outliers.
 * - Text and image agreement: `clipScore` of matched image and caption embeddings.
 *
 * Features are matrices with one sample per row; the scores depend on the feature extractor, so compare only scores
 * from the same one. The metrics are collected in `evaluationMetricRegistry` of `aifn-methods/evaluation`, and
 * `generativeEvaluationFunctions` registers the other functions.
 */

export {
  clipScore,
  densityCoverage,
  fid,
  frechetDistance,
  generativePrecisionRecall,
  inceptionScore,
  inceptionScoreSplits,
  kid,
  kidSubsets,
  type FrechetDistance,
  type PolynomialKernel,
} from './generative'
export { generativeEvaluationFunctions } from './registry'
