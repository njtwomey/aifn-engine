/**
 * `aifn-methods/inference/mixture-models`: Bayesian inference in mixture models, by variational coordinate ascent and
 * by expectation propagation.
 *
 * - The variational Bayesian Gaussian mixture: `caviGaussianMixture` runs CAVI on
 *   $q(\Zmat)\,q(\pivec)\prod_k q(\muvec_k, \Lambdamat_k)$ as a step-through algorithm whose ELBO never decreases
 *   (state `MixtureState`, prior `MixturePrior`), and `mixturePredictiveDensity` evaluates its Student t mixture
 *   predictive density.
 * - Minka's clutter problem, a scalar $\theta$ seen through points that are signal or clutter (`ClutterProblem`):
 *   `clutterModel` gives its structure in the model language, `sampleClutter` draws data, `clutterLogLikelihood` is
 *   its likelihood, `clutterTilted` its tilted moments in closed form, `clutterEp` the options for
 *   `expectationPropagation`, and `clutterPosterior` the exact posterior on a grid to compare EP with.
 * - `mixtureModelAlgorithms` and `mixtureModelFunctions` are the module's registry entries.
 *
 * Data and results are float64. The mixture is randomised only in its start (the centres are drawn from the run's
 * stream); none of the module is differentiable.
 */

export { caviGaussianMixture, mixturePredictiveDensity, type MixturePrior, type MixtureState } from './mixture'
export { clutterTilted, type ClutterFactorOptions } from './clutter'
export {
  type ClutterProblem,
  clutterLogLikelihood,
  clutterModel,
  sampleClutter,
  clutterEp,
  clutterPosterior,
} from './clutter'
export { mixtureModelAlgorithms, mixtureModelFunctions } from './registry'
