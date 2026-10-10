/**
 * `aifn-compute/learning/calibration`: maps from a classifier's scores to calibrated probabilities, fitted on held-out
 * predictions.
 *
 * - Isotonic regression by pool adjacent violators: `poolAdjacentViolatorsSteps` (the step-through algorithm) and
 *   `isotonicRegression` (on a covariate, ties pooled), with `isotonicCalibration` as a step map of new scores.
 * - Binary scores: `plattScaling` (a sigmoid of real-valued scores), `betaCalibration` (a sigmoid of $\ln s$ and
 *   $\ln(1 - s)$ for scores in $[0, 1]$) and `histogramBinning` (each bin's fraction of positives).
 * - Multiclass outputs: `temperatureScaling` (one $T$ dividing the logits; the predicted class never changes) and
 *   `dirichletCalibration` (a penalised multinomial logistic regression on the log-probabilities).
 * - `topLabelConfidence`: each case's largest probability and whether it is right, for multiclass reliability
 *   diagrams.
 * - The registry entries of the module: `calibrationAlgorithms` and `calibrationFunctions`.
 *
 * Each fit returns its parameters and an `apply` (`probability` for Platt) that maps new predictions; the parametric
 * maps are fitted by Newton's method (Platt) or L-BFGS. Invalid labels, scores and lengths throw `DomainError` or
 * `ShapeError`. Calibration error metrics and reliability-diagram data are in `aifn-compute/learning/metrics`.
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
