/**
 * `aifn-methods/inference/mixture-models`: mixture models: the clutter problem (model, tilted moments, EP, exact posterior)
 * and CAVI for Gaussian mixtures.
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
