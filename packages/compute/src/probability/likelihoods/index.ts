/**
 * `aifn-compute/probability/likelihoods`: likelihoods of a response given a linear predictor $\eta$ (module tree
 * §2.14).
 *
 * - Links (`link` by name: identity, log, logit, probit, cloglog, inverse, inverse-squared, sqrt), each with the link,
 *   its inverse (the mean function) and $d\mu/d\eta$.
 * - Exponential-dispersion families `gaussianFamily`, `binomialFamily`, `poissonFamily`, `gammaFamily`,
 *   `inverseGaussianFamily`, `negativeBinomialFamily` (`family` by name), each with its variance function, unit
 *   deviance, canonical and default links, dispersion, pointwise log-likelihood and predictive distribution;
 *   `checkLink` rejects a link the family is not used with.
 * - `likelihood(family, link?)`: the family through a link, with the mean $\mu = g^{-1}(\eta)$, the log-likelihood,
 *   the score $\partial\ell/\partial\eta$ and the unit deviance (from $\eta$ directly where $\mu$ would round).
 * - Distributional-regression (GAMLSS) families (`distributionalFamily` by name, or `normalDistributional`,
 *   `studentTDistributional`, `boxCoxColeGreenDistributional`, `gammaDistributional`, `poissonDistributional`), each
 *   parameter with its links (`distributionalLinks`), score and expected second derivative, with the cdf and quantile
 *   function; `quantileResidual` and `wormPlot` to check a fit.
 * - Ordinal likelihoods (`ordinalLikelihood`): cumulative, continuation-ratio and adjacent-category models over a
 *   logit, probit or cloglog latent cdf, with class probabilities and log-likelihoods in $\eta$ and the thresholds.
 * - `linkRegistry` and `likelihoodRegistry`: the links and families with their metadata.
 *
 * Links, the exponential-dispersion families, `likelihood` and the ordinal likelihoods are compositions of
 * primitives, so they take tensors and traced values and are differentiable. The distributional families work on
 * plain numbers, one observation at a time, and give their derivatives in closed form. Unknown names and links a
 * family does not take throw `DomainError`. The fitting machinery (IRLS, smoothing selection) stays in
 * `aifn-methods/learning/generalised`.
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
