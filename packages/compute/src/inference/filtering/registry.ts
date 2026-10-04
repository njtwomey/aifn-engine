/**
 * The algorithms of `aifn-compute/inference/filtering`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as changepoint from './changepoint'
import * as kalman from './kalman'
import * as nonlinear from './nonlinear'
import * as tracking from './tracking'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/filtering')

algorithm(
  {
    key: 'bocpd',
    name: 'Bayesian online changepoint detection',
    problem: 'sequence',
    state: { iterate: 'logPosterior', objective: 'logEvidence', flags: ['terminated'] },
    notes: ['bayesian-online-changepoint-detection'],
    cite: ['adams2007'],
  },
  changepoint.bocpd,
)

algorithm(
  {
    key: 'kalmanFilterSteps',
    name: 'Kalman filter',
    problem: 'sequence',
    state: { iterate: 'mean', objective: 'logLikelihood', flags: ['terminated'] },
    notes: ['kalman-filter'],
    cite: ['kalman1960'],
  },
  kalman.kalmanFilterSteps,
)

algorithm(
  {
    key: 'rtsSmootherSteps',
    name: 'Rauch–Tung–Striebel smoother',
    problem: 'sequence',
    state: { iterate: 'mean', flags: ['terminated'] },
    notes: ['kalman-smoother'],
    cite: ['rauch1965'],
  },
  kalman.rtsSmootherSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const filteringAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', changepoint, kalman) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'inference/filtering')
const KF = ['kalman-filter', 'linear-gaussian-state-space-models']
const BOCPD = ['bayesian-online-changepoint-detection', 'change-point-detection']

fn(
  { key: 'kalmanFilter', name: 'Kalman filter', role: 'inference', notes: KF, cite: ['kalman1960'] },
  kalman.kalmanFilter,
)
fn(
  {
    key: 'rtsSmoother',
    name: 'Rauch–Tung–Striebel smoother',
    role: 'inference',
    notes: ['kalman-smoother', ...KF],
    cite: ['rauch1965'],
  },
  kalman.rtsSmoother,
)
fn({ key: 'kalmanStep', name: 'Kalman predict–update step', role: 'inference', notes: KF }, kalman.kalmanStep)
fn({ key: 'rtsStep', name: 'RTS smoothing step', role: 'inference', notes: ['kalman-smoother'] }, kalman.rtsStep)
fn({ key: 'filterAll', name: 'Kalman filter on arrays', role: 'inference', notes: KF }, kalman.filterAll)
fn(
  { key: 'smoothAll', name: 'RTS smoother on arrays', role: 'inference', notes: ['kalman-smoother'] },
  kalman.smoothAll,
)
fn(
  {
    key: 'simulateStateSpace',
    name: 'Simulate a linear-Gaussian state-space model',
    role: 'simulation',
    random: true,
    notes: KF,
  },
  kalman.simulateStateSpace,
)
fn(
  {
    key: 'steadyStateKalman',
    name: 'Steady-state Kalman gain',
    summary: 'The limiting gain and covariance from the discrete Riccati equation.',
    role: 'solver',
    notes: KF,
  },
  kalman.steadyStateKalman,
)
fn(
  { key: 'normalisedInnovationSquared', name: 'Normalised innovation squared (NIS)', role: 'test', notes: KF },
  kalman.normalisedInnovationSquared,
)
fn(
  {
    key: 'normalisedEstimationErrorSquared',
    name: 'Normalised estimation error squared (NEES)',
    role: 'test',
    notes: KF,
  },
  kalman.normalisedEstimationErrorSquared,
)
fn(
  {
    key: 'extendedKalmanFilter',
    name: 'Extended Kalman filter',
    role: 'inference',
    notes: ['extended-and-unscented-kalman-filters'],
  },
  nonlinear.extendedKalmanFilter,
)
fn(
  {
    key: 'unscentedKalmanFilter',
    name: 'Unscented Kalman filter',
    role: 'inference',
    notes: ['extended-and-unscented-kalman-filters'],
    cite: ['julier1997', 'julier2004'],
  },
  nonlinear.unscentedKalmanFilter,
)
fn(
  { key: 'constantHazard', name: 'Constant hazard', role: 'construction', notes: BOCPD, cite: ['adams2007'] },
  changepoint.constantHazard,
)
fn({ key: 'bocpdInit', name: 'BOCPD initial state', role: 'construction', notes: BOCPD }, changepoint.bocpdInit)
fn(
  {
    key: 'bocpdUpdate',
    name: 'BOCPD update',
    summary: 'One step of the run-length recursion: growth and changepoint probabilities.',
    role: 'inference',
    notes: BOCPD,
    cite: ['adams2007'],
  },
  changepoint.bocpdUpdate,
)
fn(
  { key: 'bocpdForecast', name: 'BOCPD one-step forecast', role: 'inference', notes: BOCPD },
  changepoint.bocpdForecast,
)
fn(
  { key: 'bocpdPredictiveDensity', name: 'BOCPD predictive density', role: 'inference', notes: BOCPD },
  changepoint.bocpdPredictiveDensity,
)
fn({ key: 'runLengthMass', name: 'Run-length mass', role: 'property', notes: BOCPD }, changepoint.runLengthMass)
fn({ key: 'runLengthRow', name: 'Run-length row', role: 'property', notes: BOCPD }, changepoint.runLengthRow)
fn(
  { key: 'detectChangepoints', name: 'Detect changepoints', role: 'inference', notes: BOCPD },
  changepoint.detectChangepoints,
)
fn({ key: 'mapChangepoints', name: 'MAP changepoints', role: 'inference', notes: BOCPD }, changepoint.mapChangepoints)
fn(
  {
    key: 'normalKnownVariance',
    name: 'Normal model with known variance',
    role: 'construction',
    notes: [...BOCPD, 'conjugate-priors'],
  },
  changepoint.normalKnownVariance,
)
fn(
  { key: 'normalGamma', name: 'Normal–gamma model', role: 'construction', notes: [...BOCPD, 'conjugate-priors'] },
  changepoint.normalGamma,
)
fn(
  { key: 'poissonGamma', name: 'Poisson–gamma model', role: 'construction', notes: [...BOCPD, 'conjugate-priors'] },
  changepoint.poissonGamma,
)
fn(
  { key: 'betaBernoulli', name: 'Beta–Bernoulli model', role: 'construction', notes: [...BOCPD, 'conjugate-priors'] },
  changepoint.betaBernoulli,
)
fn(
  {
    key: 'regressionNormalGamma',
    name: 'Bayesian linear regression model',
    role: 'construction',
    notes: [...BOCPD, 'bayesian-linear-regression'],
  },
  changepoint.regressionNormalGamma,
)
fn(
  { key: 'laggedObservations', name: 'Lagged observations', role: 'transform', notes: BOCPD },
  changepoint.laggedObservations,
)

fn(
  {
    key: 'trackingMetrics',
    name: 'Tracking metrics against a known truth',
    summary: 'Lag to follow a step, overshoot, settled noise, RMSE before and after the change, and band coverage.',
    role: 'test',
    notes: [...KF, 'kalman-smoother'],
  },
  tracking.trackingMetrics,
)

/** The functions of the module, keyed by name. */
export const filteringFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', kalman, nonlinear, changepoint, tracking) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
