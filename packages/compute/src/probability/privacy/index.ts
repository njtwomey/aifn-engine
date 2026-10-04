/**
 * `aifn-compute/probability/privacy`: differential privacy. Mechanisms (Laplace, Gaussian with classic and analytic
 * calibration and its exact privacy profile, exponential, randomised response), accounting (sequential and advanced
 * composition, Rényi DP of the subsampled Gaussian with conversion to (ε, δ), zCDP), and DP-SGD's clipped, noised
 * gradient aggregation (the training wrapper is `aifn-compute/nn/training`'s `privateTraining`).
 */

export {
  analyticGaussianSigma,
  classicGaussianSigma,
  exponentialMechanism,
  exponentialMechanismProbabilities,
  gaussianDelta,
  gaussianEpsilon,
  gaussianMechanism,
  laplaceMechanism,
  randomisedResponse,
  randomisedResponseEstimate,
  randomisedResponseKeep,
} from './mechanisms'
export {
  advancedComposition,
  DEFAULT_ORDERS,
  dpSgdEpsilon,
  gaussianZcdp,
  rdpSubsampledGaussian,
  rdpToEpsilon,
  sequentialComposition,
  zcdpToEpsilon,
} from './accounting'
export { clipAndNoise, type PrivateGradient } from './clipping'
export { privacyFunctions } from './registry'
