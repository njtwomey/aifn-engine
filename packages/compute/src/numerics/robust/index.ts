/**
 * `aifn-compute/numerics/robust`: robust model fitting. `ransac` (random sample consensus, a step algorithm generic over the
 * model through `fit` and `residuals` callbacks, with the adaptive sample count `ransacTrials`) and `ransacFit` (run,
 * then refit on the consensus set).
 */

export {
  ransac,
  ransacFit,
  ransacTrials,
  type RansacFit,
  type RansacOptions,
  type RansacProblem,
  type RansacState,
} from './ransac'
export { robustAlgorithms, robustFunctions } from './registry'
