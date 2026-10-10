/**
 * `aifn-methods/learning/mixture-density`: the mixture density network (Bishop, 1994), an MLP whose outputs are the
 * weights, means and scales of a Gaussian mixture over the target, for regression where $\yvec$ given $\xvec$ has
 * several branches.
 *
 * - The estimator: `mixtureDensityNetwork`, whose `predictive` is the mixture and whose `decide` is its most probable
 *   mode.
 * - The network: `mdnSpec` (the defaults), `mdnModel` (the MLP of a config, with the mixture head of width
 *   $K(1 + 2D)$, or $D$ outputs for the squared-error network of the same body), `mdnLoss` (the mixture negative
 *   log-likelihood of `aifn-compute/learning/losses`, or the squared error; differentiable) and `mdnPredict` (the
 *   conditional mean and the mixture, as numbers).
 * - Training: `mdnTraining` (Adam as a step-through algorithm) and `mixtureDensityRun` (a generator that trains an MDN
 *   and the squared-error network side by side and yields snapshots for a page).
 * - Evaluation: `mdnLogLikelihood` (the squared-error network read as a Gaussian with its residual variance, so the two
 *   compare) and `mdnMeanSquaredError`.
 * - Inputs: `inputMatrix` (a vector becomes a column), `inputStandardisation` (column means and standard deviations)
 *   and `standardisedInputs` (what the network sees).
 *
 * `mixtureDensityAlgorithms` lists the training algorithm by key.
 */

export {
  inputMatrix,
  inputStandardisation,
  mdnLogLikelihood,
  mdnLoss,
  mdnMeanSquaredError,
  mdnModel,
  mdnPredict,
  mdnSpec,
  standardisedInputs,
  type MdnConfig,
  type MdnModel,
  type MdnObjective,
  type MdnPrediction,
  type MdnSpec,
} from './model'
export {
  mdnTraining,
  mixtureDensityNetwork,
  mixtureDensityRun,
  type MdnCheckpoint,
  type MdnData,
  type MdnHistory,
  type MdnRunOptions,
  type MdnSnapshot,
  type MdnTrainingOptions,
  type MixtureDensityNetworkParams,
} from './training'
export { mixtureDensityAlgorithms } from './registry'
