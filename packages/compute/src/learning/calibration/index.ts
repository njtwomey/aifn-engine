/**
 * `aifn-compute/learning/calibration`: maps from scores to calibrated probabilities, fitted on held-out predictions: isotonic
 * regression by pool adjacent violators (`poolAdjacentViolatorsSteps`, the step-through form; `isotonicRegression` and
 * `isotonicCalibration`), Platt scaling, temperature scaling, beta and Dirichlet calibration, and histogram binning;
 * `topLabelConfidence` for multiclass reliability diagrams. Calibration error metrics and reliability-diagram data are
 * in `aifn-compute/learning/metrics`.
 */

export {
  isotonicRegression,
  poolAdjacentViolatorsSteps,
  type IsotonicFit,
  type IsotonicOptions,
  type PavEvent,
  type PavState,
} from './isotonic'
export { plattScaling, type PlattOptions, type PlattScaling } from './platt'
export {
  betaCalibration,
  dirichletCalibration,
  histogramBinning,
  isotonicCalibration,
  temperatureScaling,
  topLabelConfidence,
  type BetaCalibration,
  type DirichletCalibration,
  type HistogramBinning,
  type IsotonicCalibration,
  type TemperatureScaling,
} from './maps'
export { calibrationAlgorithms, calibrationFunctions } from './registry'
