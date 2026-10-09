/**
 * `aifn-compute/inference/filtering`: sequential Bayesian inference over time series, from the Kalman filter to online
 * changepoint detection.
 *
 * - Linear-Gaussian state-space models ($\zvec_t = \Amat\zvec_{t-1} + \wvec_t$, $\yvec_t = \Cmat\zvec_t + \vvec_t$):
 *   `simulateStateSpace`, `kalmanFilter` and `rtsSmoother` over a whole series, `kalmanFilterSteps` and
 *   `rtsSmootherSteps` to step through them, and `steadyStateKalman` for the limiting gain.
 * - The single steps they share with EM (`aifn-methods/timeseries`): `parseModel`, `kalmanStep`, `filterAll`,
 *   `packFilter`, `rtsStep` and `smoothAll`.
 * - Nonlinear models: `extendedKalmanFilter` (Jacobians by autodiff, so $f$ and $h$ must be differentiable) and
 *   `unscentedKalmanFilter` (sigma points; any $f$ and $h$).
 * - Checking a filter: `normalisedInnovationSquared` (from the data alone) and `normalisedEstimationErrorSquared`
 *   (against a simulated truth) for consistency, and `trackingMetrics` for lag, overshoot, error and band coverage.
 * - Bayesian online changepoint detection: `bocpdInit` and `bocpdUpdate` to feed values one at a time, `bocpd` as an
 *   algorithm, `detectChangepoints` over a whole series with `mapChangepoints`; the posterior read by `runLengthRow`
 *   and `runLengthMass`, forecasts by `bocpdForecast` and `bocpdPredictiveDensity`; `constantHazard`; and the
 *   conjugate segment models `normalKnownVariance`, `normalGamma`, `poissonGamma`, `betaBernoulli` and
 *   `regressionNormalGamma` (with `laggedObservations` for autoregressions).
 *
 * Missing observations are NaN. Nothing throws for a singular matrix: the step is reported (`singular`,
 * `singularSteps`) and filtering goes on. Shapes that disagree throw `ShapeError`, invalid parameters `DomainError`.
 * Everything is deterministic except `simulateStateSpace`, which draws from the stream it is given.
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
