/**
 * The registry of `aifn-methods/timeseries`: the fits as traceable algorithms (each with the one-call function that
 * runs it), and the model properties, simulators, forecasts and decompositions as functions, each linked to the notes
 * it serves.
 */

import {
  definer,
  entries,
  type AlgorithmInfo,
  type Entry,
  type FunctionInfo,
  type StateRoles,
} from 'aifn-compute/foundation/registry'
import * as arma from './arma'
import * as decompose from './decompose'
import * as garch from './garch'
import * as sarima from './sarima'
import * as smoothing from './smoothing'
import * as stateSpace from './state-space'
import * as tracking from './tracking'

const algorithm = definer<AlgorithmInfo>('algorithm', 'timeseries')
const fn = definer<FunctionInfo>('function', 'timeseries')
const fit: StateRoles = { iterate: 'params', objective: 'objective', flags: ['converged', 'diverged'] }

const ARMA = ['autoregressive-moving-average-model', 'autoregressive-model', 'moving-average-model']
const SARIMA = ['seasonal-autoregressive-integrated-moving-average', 'autoregressive-integrated-moving-average-model']
const GARCH = ['garch-model', 'arch-model', 'volatility-clustering']
const SMOOTHING = ['error-trend-seasonal-models', 'simple-exponential-smoothing', 'holt-linear-trend', 'holt-winters']

// ── ARMA ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'armaFitSteps',
    name: 'ARMA fit by maximum likelihood',
    summary: 'Nelder–Mead on the exact Gaussian ARMA likelihood over partial-autocorrelation coordinates.',
    problem: 'sequence',
    state: fit,
    notes: [...ARMA, 'box-jenkins-method'],
    cite: ['box2015', 'brockwell1991'],
  },
  arma.armaFitSteps,
)
fn({ key: 'fitArma', name: 'Fit an ARMA model', role: 'fit', notes: [...ARMA, 'box-jenkins-method'] }, arma.fitArma)
fn(
  {
    key: 'armaLogLikelihood',
    name: 'ARMA log-likelihood',
    summary: 'The exact Gaussian log-likelihood of an ARMA model by the innovations algorithm.',
    role: 'estimator',
    notes: ARMA,
    cite: ['brockwell1991'],
  },
  arma.armaLogLikelihood,
)
fn(
  { key: 'armaResiduals', name: 'ARMA residuals', role: 'transform', notes: [...ARMA, 'ljung-box-test'] },
  arma.armaResiduals,
)
fn(
  {
    key: 'armaRoots',
    name: 'ARMA lag-polynomial roots',
    summary:
      'The roots of the AR and MA lag polynomials; stationary and invertible when all lie outside the unit circle.',
    role: 'property',
    notes: [...ARMA, 'stationarity'],
  },
  arma.armaRoots,
)
fn(
  { key: 'isStationary', name: 'AR stationarity', role: 'property', notes: ['stationarity', 'autoregressive-model'] },
  arma.isStationary,
)
fn(
  { key: 'isInvertible', name: 'MA invertibility', role: 'property', notes: ['moving-average-model'] },
  arma.isInvertible,
)
fn(
  {
    key: 'psiWeights',
    name: 'ψ weights (MA(∞) form)',
    role: 'property',
    notes: [...ARMA, 'prediction-intervals'],
    cite: ['brockwell1991'],
  },
  arma.psiWeights,
)
fn(
  {
    key: 'armaAutocovariance',
    name: 'ARMA autocovariance',
    role: 'property',
    notes: [...ARMA, 'autocorrelation-and-partial-autocorrelation'],
  },
  arma.armaAutocovariance,
)
fn(
  {
    key: 'armaAutocorrelation',
    name: 'ARMA autocorrelation',
    role: 'property',
    notes: ['autocorrelation-and-partial-autocorrelation', ...ARMA],
  },
  arma.armaAutocorrelation,
)
fn(
  {
    key: 'simulateArma',
    name: 'Simulate an ARMA process',
    role: 'simulation',
    random: true,
    notes: [...ARMA, 'white-noise-and-random-walk'],
  },
  arma.simulateArma,
)
fn(
  {
    key: 'forecastArma',
    name: 'ARMA forecast',
    role: 'inference',
    notes: [...ARMA, 'prediction-intervals', 'probabilistic-forecasting'],
  },
  arma.forecastArma,
)

// ── Seasonal ARIMA ───────────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'sarimaFitSteps',
    name: 'Seasonal ARIMA fit by maximum likelihood',
    summary: 'Nelder–Mead on the exact likelihood of the differenced series under the expanded seasonal ARMA.',
    problem: 'sequence',
    state: fit,
    notes: [...SARIMA, 'box-jenkins-method'],
    cite: ['box2015'],
  },
  sarima.sarimaFitSteps,
)
fn(
  { key: 'fitSarima', name: 'Fit a seasonal ARIMA model', role: 'fit', notes: [...SARIMA, 'box-jenkins-method'] },
  sarima.fitSarima,
)
fn(
  {
    key: 'expandSarima',
    name: 'Expand a seasonal ARIMA',
    summary: 'The multiplicative seasonal polynomials multiplied out into one AR, MA and differencing polynomial.',
    role: 'construction',
    notes: SARIMA,
  },
  sarima.expandSarima,
)
fn(
  { key: 'sarimaSpec', name: 'Seasonal ARIMA specification of a fit', role: 'construction', notes: SARIMA },
  sarima.sarimaSpec,
)
fn(
  { key: 'sarimaLogLikelihood', name: 'Seasonal ARIMA log-likelihood', role: 'estimator', notes: SARIMA },
  sarima.sarimaLogLikelihood,
)
fn(
  { key: 'sarimaResiduals', name: 'Seasonal ARIMA residuals', role: 'transform', notes: [...SARIMA, 'ljung-box-test'] },
  sarima.sarimaResiduals,
)
fn(
  { key: 'simulateSarima', name: 'Simulate a seasonal ARIMA process', role: 'simulation', random: true, notes: SARIMA },
  sarima.simulateSarima,
)
fn(
  {
    key: 'forecastSarima',
    name: 'Seasonal ARIMA forecast',
    role: 'inference',
    notes: [...SARIMA, 'prediction-intervals'],
  },
  sarima.forecastSarima,
)

// ── Differencing and decomposition ───────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'difference',
    name: 'Differencing',
    role: 'transform',
    notes: ['differencing-and-unit-roots', 'autoregressive-integrated-moving-average-model'],
  },
  decompose.difference,
)
fn(
  { key: 'undifference', name: 'Undo differencing', role: 'transform', notes: ['differencing-and-unit-roots'] },
  decompose.undifference,
)
fn(
  {
    key: 'classicalDecomposition',
    name: 'Classical decomposition',
    summary: 'Trend by a centred moving average, seasonal indices by averaging the detrended series per season.',
    role: 'transform',
    returns: 'decomposition',
    notes: ['time-series-decomposition'],
  },
  decompose.classicalDecomposition,
)
fn(
  {
    key: 'stl',
    name: 'STL decomposition',
    summary: 'Seasonal–trend decomposition by loess (inner and robustness loops).',
    role: 'transform',
    returns: 'decomposition',
    notes: ['time-series-decomposition'],
    cite: ['cleveland1990'],
  },
  decompose.stl,
)

// ── Exponential smoothing ────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'exponentialSmoothingFitSteps',
    name: 'Exponential smoothing fit',
    summary: 'Nelder–Mead on the one-step SSE over logit coordinates of the smoothing parameters.',
    problem: 'sequence',
    state: fit,
    notes: SMOOTHING,
    cite: ['hyndman2008'],
  },
  smoothing.exponentialSmoothingFitSteps,
)
fn(
  {
    key: 'exponentialSmoothing',
    name: 'Exponential smoothing',
    summary: 'Level, trend and seasonal recursions (additive or multiplicative, damped trend), with forecasts.',
    role: 'transform',
    notes: SMOOTHING,
    cite: ['hyndman2008', 'holt2004', 'winters1960'],
  },
  smoothing.exponentialSmoothing,
)

// ── GARCH ────────────────────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'garchFitSteps',
    name: 'GARCH(1, 1) fit by maximum likelihood',
    problem: 'sequence',
    state: fit,
    notes: GARCH,
    cite: ['bollerslev1986'],
  },
  garch.garchFitSteps,
)
fn(
  { key: 'fitGarch', name: 'Fit a GARCH(1, 1) model', role: 'fit', notes: GARCH, cite: ['bollerslev1986'] },
  garch.fitGarch,
)
fn(
  { key: 'garchLogLikelihood', name: 'GARCH log-likelihood', role: 'estimator', notes: GARCH },
  garch.garchLogLikelihood,
)
fn(
  {
    key: 'garchProperties',
    name: 'GARCH properties',
    summary: 'Persistence α + β, the unconditional variance and the half-life of a volatility shock.',
    role: 'property',
    notes: GARCH,
  },
  garch.garchProperties,
)
fn(
  { key: 'simulateGarch', name: 'Simulate a GARCH process', role: 'simulation', random: true, notes: GARCH },
  garch.simulateGarch,
)
fn({ key: 'garchForecast', name: 'GARCH variance forecast', role: 'inference', notes: GARCH }, garch.garchForecast)

// ── State-space models ───────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'stateSpaceEm',
    name: 'EM for a linear-Gaussian state-space model',
    summary: 'RTS-smoother E-step and closed-form M-step; the log-likelihood never decreases.',
    problem: 'gaussian-model',
    state: { iterate: 'model', objective: 'logLikelihood', flags: ['converged', 'diverged'] },
    notes: ['linear-gaussian-state-space-models', 'kalman-smoother', 'expectation-maximisation'],
    cite: ['shumway1982', 'ghahramani1996'],
  },
  stateSpace.stateSpaceEm,
)
fn(
  {
    key: 'constantVelocityModel',
    name: 'Constant-velocity model',
    summary: 'The nearly-constant-velocity tracking model with white-noise acceleration, observed through positions.',
    role: 'construction',
    notes: ['kalman-filter', 'linear-gaussian-state-space-models'],
  },
  tracking.constantVelocityModel,
)

/** The algorithms of the module (the fits and state-space EM), keyed by factory name. */
export const timeseriesAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', arma, sarima, smoothing, garch, stateSpace) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

/** The functions of the module, keyed by name. */
export const timeseriesFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', arma, sarima, decompose, smoothing, garch, tracking) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
