/**
 * `aifn-compute/probability/privacy`: differential privacy: noise mechanisms, their calibration, privacy accounting
 * and DP-SGD's private gradient aggregation, as Opacus and TensorFlow Privacy.
 *
 * - Mechanisms: `laplaceMechanism` ($\varepsilon$-DP), `gaussianMechanism`, `exponentialMechanism` (a private choice,
 *   with `exponentialMechanismProbabilities`) and `randomisedResponse` (with `randomisedResponseKeep` and the
 *   unbiased `randomisedResponseEstimate`).
 * - Calibrating the Gaussian mechanism: `classicGaussianSigma` (for $\varepsilon < 1$) and `analyticGaussianSigma`
 *   (the smallest $\sigma$, for any $\varepsilon > 0$); its exact privacy profile `gaussianDelta` and the inverse
 *   `gaussianEpsilon`.
 * - Accounting: `sequentialComposition` and `advancedComposition`; Rényi DP of the subsampled Gaussian
 *   (`rdpSubsampledGaussian` at the `DEFAULT_ORDERS`) with conversion to $(\varepsilon, \delta)$ by `rdpToEpsilon`,
 *   and both at once for DP-SGD in `dpSgdEpsilon`; zCDP by `gaussianZcdp` and `zcdpToEpsilon`.
 * - DP-SGD's clipped, noised gradient aggregation, `clipAndNoise` (the training wrapper is
 *   `aifn-compute/nn/training`'s `privateTraining`).
 * - `privacyFunctions` lists the functions with their metadata.
 *
 * Mechanisms take their randomness from an `aifn-compute/foundation/random` stream, the last argument, and work on
 * plain numbers (they are not differentiable). Most functions check their parameters and throw a `DomainError` for one
 * out of range.
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
