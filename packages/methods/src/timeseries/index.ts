/**
 * `aifn-methods/timeseries`: models of one series observed in time: ARMA and seasonal ARIMA fitting and forecasting,
 * GARCH, exponential smoothing, classical and STL decomposition, and EM for linear-Gaussian state-space models.
 *
 * - ARMA($p$, $q$) properties: `armaRoots`, `isStationary` and `isInvertible` from the roots of the lag polynomials;
 *   `psiWeights`, `armaAutocovariance` and `armaAutocorrelation` for the theoretical second-order structure.
 * - ARMA data: `simulateArma`, `armaResiduals` (conditional), `armaLogLikelihood` (exact, by the Kalman filter),
 *   `fitArma` and its traceable `armaFitSteps` (CSS or exact maximum likelihood), and `forecastArma` with intervals.
 * - Seasonal ARIMA($p$, $d$, $q$)($P$, $D$, $Q$)$_s$: the same set on the differenced series, `simulateSarima`,
 *   `sarimaResiduals`, `sarimaLogLikelihood`, `fitSarima` and `sarimaFitSteps`, `forecastSarima`, with
 *   `expandSarima` to multiply out the seasonal polynomials and `sarimaSpec` to turn a fit into a model.
 * - Differencing and decomposition: `difference` and `undifference`; `classicalDecomposition` (moving averages,
 *   additive or multiplicative) and `stl` (loess, additive, optionally robust), both returning a
 *   `SeasonalDecomposition`.
 * - Exponential smoothing: `exponentialSmoothing` runs simple, Holt's (damped) and Holt–Winters recursions with
 *   forecasts and intervals; `exponentialSmoothingFitSteps` fits $\alpha$, $\beta^*$, $\gamma$, $\phi$ by least
 *   squares.
 * - Volatility: GARCH(1,1) through `garchProperties`, `simulateGarch`, `garchLogLikelihood`, `fitGarch` and
 *   `garchFitSteps`, and `garchForecast` for the variance path.
 * - State-space models: `stateSpaceEm` estimates $\Amat$, $\Cmat$, $\Qmat$, $\Rmat$ and the prior by EM;
 *   `constantVelocityModel` builds the standard tracking model for `aifn-compute/inference/filtering`.
 * - The registry: `timeseriesAlgorithms` and `timeseriesFunctions`.
 *
 * The ARMA, seasonal ARIMA, smoothing and GARCH fits are step-through `Algorithm`s whose state is a `FitState`:
 * Nelder–Mead in coordinates that keep every iterate valid (stationary and invertible polynomials, smoothing
 * parameters in $(0, 1)$, GARCH persistence below 1). `fitArma`, `fitSarima` and `fitGarch` run them in one call and
 * report `converged`; `stateSpaceEm` is a step-through `Algorithm` too. Signs follow statsmodels: the AR polynomial
 * is $1 - \sum_i \phi_i z^i$ and the MA polynomial $1 + \sum_j \theta_j z^j$. Series are `VectorLike`; returned
 * series and coefficients are tensors. Nothing is clipped: an explosive simulation is reported (`stationary`,
 * `diverged`) and a non-stationary likelihood is $-\infty$.
 */

export {
  armaAutocorrelation,
  armaAutocovariance,
  armaFitSteps,
  armaLogLikelihood,
  armaResiduals,
  armaRoots,
  fitArma,
  forecastArma,
  isInvertible,
  isStationary,
  psiWeights,
  simulateArma,
  type ArmaFit,
  type ArmaFitOptions,
  type ArmaLikelihood,
  type ArmaSimulation,
  type ArmaSpec,
  type Forecast,
  type LagRoots,
} from './arma'
export {
  expandSarima,
  fitSarima,
  forecastSarima,
  sarimaFitSteps,
  sarimaLogLikelihood,
  sarimaResiduals,
  sarimaSpec,
  simulateSarima,
  type SarimaFit,
  type SarimaFitOptions,
  type SarimaSimulation,
  type SarimaSpec,
} from './sarima'
export {
  classicalDecomposition,
  difference,
  stl,
  undifference,
  type SeasonalDecomposition,
  type StlOptions,
} from './decompose'
export {
  exponentialSmoothing,
  exponentialSmoothingFitSteps,
  type SmoothingResult,
  type SmoothingSpec,
  type SmoothingStructure,
} from './smoothing'
export {
  fitGarch,
  garchFitSteps,
  garchForecast,
  garchLogLikelihood,
  garchProperties,
  simulateGarch,
  type GarchSpec,
} from './garch'
export type { FitState } from './fit'
export { stateSpaceEm, type EmEstimate, type StateSpaceEmState } from './state-space'
export { constantVelocityModel, type ConstantVelocityOptions } from './tracking'
export { timeseriesAlgorithms, timeseriesFunctions } from './registry'
