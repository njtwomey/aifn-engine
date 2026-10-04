/**
 * `aifn-compute/probability/likelihoods`: likelihoods of a response given a linear predictor η (module tree §2.14).
 *
 * - Links (`link` by name: identity, log, logit, probit, cloglog, inverse, inverse-squared, sqrt), each with the link,
 *   its inverse (the mean function) and dμ/dη.
 * - Exponential-dispersion families `gaussianFamily`, `binomialFamily`, `poissonFamily`, `gammaFamily`,
 *   `inverseGaussianFamily`, `negativeBinomialFamily` (`family` by name), each with its variance function, unit
 *   deviance, canonical and default links, dispersion and pointwise log-likelihood.
 * - `likelihood(family, link?)`: the family through a link, with `mean(η)`, `logLik(y, η)`, the score ∂ℓ/∂η and the
 *   unit deviance.
 * - Distributional-regression (GAMLSS) families (`distributionalFamily`: normal, Student t, Box–Cox Cole–Green,
 *   gamma, Poisson), each parameter with its links, score and expected second derivative, cdf and quantile function;
 *   `quantileResidual` and `wormPlot`.
 * - Ordinal likelihoods (`ordinalLikelihood`): cumulative, continuation-ratio and adjacent-category models over a
 *   logit, probit or cloglog latent cdf, with class probabilities and log-likelihoods in η and the thresholds.
 *
 * - `linkRegistry` and `likelihoodRegistry`: the links and families with their metadata.
 *
 * All are compositions of primitives, so they are differentiable. The fitting machinery (IRLS, smoothing selection)
 * stays in `aifn-methods/learning/generalised`.
 */

export {
  binomialFamily,
  checkLink,
  family,
  gammaFamily,
  gaussianFamily,
  inverseGaussianFamily,
  likelihood,
  link,
  negativeBinomialFamily,
  poissonFamily,
  type Family,
  type FamilyName,
  type Likelihood,
  type LikelihoodOptions,
  type Link,
  type LinkName,
} from './families'
export {
  boxCoxColeGreenDistributional,
  distributionalFamily,
  distributionalLinks,
  gammaDistributional,
  normalDistributional,
  poissonDistributional,
  quantileResidual,
  studentTDistributional,
  type DistributionalFamily,
  type DistributionalFamilyName,
  type DistributionalParameter,
  type DistributionalParameterInfo,
  type WormPlot,
  wormPlot,
} from './distributional'
export { ordinalLikelihood, type OrdinalLikelihood, type OrdinalLinkName, type OrdinalModel } from './ordinal'
export { likelihoodRegistry, linkRegistry } from './registry'
