/**
 * `aifn-methods/evaluation/generative`: generative-model metrics (FID, KID, inception score, precision and recall).
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
