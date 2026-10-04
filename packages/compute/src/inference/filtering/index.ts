/**
 * `aifn-compute/inference/filtering`: state-space filtering: linear-Gaussian models, simulation, the Kalman filter, the
 * Rauch–Tung–Striebel smoother (both also as step-through algorithms), the steady-state filter, the extended and unscented Kalman filters, and Bayesian online
 * changepoint detection (the run-length filter over conjugate segment models).
 */

export {
  kalmanFilter,
  kalmanFilterSteps,
  normalisedEstimationErrorSquared,
  normalisedInnovationSquared,
  rtsSmoother,
  rtsSmootherSteps,
  simulateStateSpace,
  steadyStateKalman,
  type KalmanFilterResult,
  type KalmanFilterState,
  type RtsSmootherState,
  type SmootherResult,
  type StateSpaceModel,
} from './kalman'
// The steps the batch functions, the algorithms and EM (`aifn-methods/timeseries`) share, on tensors.
export {
  filterAll,
  kalmanStep,
  packFilter,
  parseModel,
  rtsStep,
  smoothAll,
  type FilterRun,
  type KalmanStep,
  type Model,
  type SmootherStep,
  type SmootherRun,
} from './kalman'
export {
  extendedKalmanFilter,
  unscentedKalmanFilter,
  type NonlinearStateSpaceModel,
  type UnscentedOptions,
} from './nonlinear'
export {
  betaBernoulli,
  bocpd,
  bocpdForecast,
  bocpdInit,
  bocpdPredictiveDensity,
  bocpdUpdate,
  constantHazard,
  detectChangepoints,
  laggedObservations,
  mapChangepoints,
  normalGamma,
  normalKnownVariance,
  poissonGamma,
  regressionNormalGamma,
  runLengthMass,
  runLengthRow,
  type BocpdOptions,
  type BocpdState,
  type ChangepointDetection,
  type ConjugatePredictive,
  type Hazard,
  type Regressed,
  type RunStats,
} from './changepoint'
export { trackingMetrics, type TrackingMetrics, type TrackingOptions } from './tracking'
export { filteringAlgorithms, filteringFunctions } from './registry'
