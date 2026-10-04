/**
 * `aifn-compute/inference/variational`: variational inference: the ELBO and its estimators (`elbo`, `elboGradient`,
 * `gradientVariance`), black-box VI (`bbvi`), and the mean-field and full-rank Gaussian families.
 */

export { fullRankGaussian, meanFieldGaussian, type FamilyKernels, type GaussianFamily, type VectorLike } from './family'
export {
  elbo,
  elboGradient,
  gradientVariance,
  type Baseline,
  type ElboEstimate,
  type ElboGradient,
  type ElboGradientOptions,
  type GradientEstimator,
  type GradientVariance,
} from './elbo'
export { bbvi, type BbviOptions, type BbviStart, type BbviState } from './bbvi'
export { variationalAlgorithms, variationalFunctions } from './registry'
