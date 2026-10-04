/**
 * `aifn-methods/learning/mixture-density`: the mixture density network (Bishop, 1994): an MLP whose outputs are the
 * weights, means and scales of a Gaussian mixture over the target, trained on the mixture negative log-likelihood of
 * `aifn-compute/learning/losses` (`mdnModel`, `mdnLoss`, `mdnPredict`); the squared-error network of the same body for
 * comparison; `mdnTraining` (Adam, a step-through algorithm); `mixtureDensityRun`, a streaming run that trains both side
 * by side; and `mixtureDensityNetwork`, the registered estimator whose predictive is the mixture.
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
