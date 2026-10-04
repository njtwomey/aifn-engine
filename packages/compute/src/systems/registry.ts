/**
 * The algorithms of `aifn-compute/systems`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as criteria from './criteria'
import * as identification from './identification'
import * as locus from './locus'
import * as placement from './placement'
import * as responses from './responses'
import * as structure from './structure'
import * as system from './system'
import * as transform from './transform'

const algorithm = definer<AlgorithmInfo>('algorithm', 'systems')

algorithm(
  {
    key: 'simulate',
    name: 'LTI simulation',
    summary: 'Steps a discrete LTI system x[k+1] = A x[k] + B u[k] through an input sequence.',
    problem: 'lti-system',
    state: { iterate: 'x', flags: ['diverged', 'terminated'] },
    notes: ['state-space-representation', 'difference-equations', 'linear-time-invariant-systems'],
  },
  responses.simulate,
)

algorithm(
  {
    key: 'predictionErrorMethod',
    name: 'Prediction-error method',
    summary: 'Damped Gauss–Newton on the one-step prediction error of A y = (B/F) u + C e; gradients by filtering.',
    problem: 'least-squares',
    state: { iterate: 'theta', objective: 'loss', flags: ['converged', 'stalled'] },
    notes: ['autoregressive-exogenous-models-and-prediction-error-methods'],
    cite: ['ljung1999'],
  },
  identification.predictionErrorMethod,
)

const fn = definer<FunctionInfo>('function', 'systems')
const LTI = ['linear-time-invariant-systems']
const TF = ['transfer-function-and-block-diagrams', ...LTI]
const SS = ['state-space-representation', ...LTI]
const PZ = ['poles-and-zeros', 'z-transform']

// ── Representations ──────────────────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'transferFunction', name: 'Transfer function', role: 'construction', returns: 'lti', notes: TF },
  system.transferFunction,
)
fn(
  {
    key: 'zerosPolesGain',
    name: 'Zeros, poles and gain',
    role: 'construction',
    returns: 'lti',
    notes: [...PZ, ...LTI],
  },
  system.zerosPolesGain,
)
fn(
  { key: 'stateSpace', name: 'State-space system', role: 'construction', returns: 'lti', notes: SS },
  system.stateSpace,
)
fn(
  {
    key: 'secondOrderSections',
    name: 'Second-order sections',
    role: 'construction',
    returns: 'lti',
    notes: ['infinite-impulse-response-filters'],
  },
  system.secondOrderSections,
)
fn(
  { key: 'toTransferFunction', name: 'To transfer function', role: 'transform', returns: 'lti', notes: TF },
  system.toTransferFunction,
)
fn(
  { key: 'toZerosPolesGain', name: 'To zeros, poles and gain', role: 'transform', returns: 'lti', notes: PZ },
  system.toZerosPolesGain,
)
fn(
  {
    key: 'toStateSpace',
    name: 'To state space (controllable canonical form)',
    role: 'transform',
    returns: 'lti',
    notes: SS,
  },
  system.toStateSpace,
)
fn(
  {
    key: 'toSecondOrderSections',
    name: 'To second-order sections',
    role: 'transform',
    returns: 'lti',
    notes: ['infinite-impulse-response-filters'],
  },
  system.toSecondOrderSections,
)
fn({ key: 'convert', name: 'Convert representation', role: 'transform', returns: 'lti', notes: LTI }, system.convert)
fn({ key: 'poles', name: 'Poles', role: 'property', notes: PZ }, system.poles)
fn({ key: 'systemZeros', name: 'Zeros', role: 'property', notes: PZ }, system.systemZeros)
fn(
  {
    key: 'stability',
    name: 'Stability',
    summary: 'Asymptotic stability from the poles: left half-plane (continuous) or inside the unit circle (discrete).',
    role: 'property',
    notes: ['lyapunov-stability', 'poles-and-zeros', 'routh-hurwitz-criterion'],
  },
  system.stability,
)
fn({ key: 'dimensions', name: 'Dimensions', role: 'property' }, system.dimensions)

// ── Structure ────────────────────────────────────────────────────────────────────────────────────────────────────────

const CO = ['controllability-and-observability']
fn(
  { key: 'controllability', name: 'Controllability (rank test)', role: 'property', notes: CO, cite: ['kalman1960b'] },
  structure.controllability,
)
fn(
  { key: 'observability', name: 'Observability (rank test)', role: 'property', notes: [...CO, 'state-observer'] },
  structure.observability,
)
fn(
  { key: 'controllabilityGramian', name: 'Controllability Gramian', role: 'property', notes: CO },
  structure.controllabilityGramian,
)
fn(
  { key: 'observabilityGramian', name: 'Observability Gramian', role: 'property', notes: CO },
  structure.observabilityGramian,
)

// ── Responses ────────────────────────────────────────────────────────────────────────────────────────────────────────

const FR = ['frequency-response', 'bode-plots-and-stability-margins']
fn({ key: 'responseAt', name: 'Frequency response at a point', role: 'property', notes: FR }, responses.responseAt)
fn(
  { key: 'frequencyResponse', name: 'Frequency response', role: 'property', returns: 'spectrum', notes: FR },
  responses.frequencyResponse,
)
fn({ key: 'frequencyGrid', name: 'Frequency grid', role: 'construction', notes: FR }, responses.frequencyGrid)
fn(
  {
    key: 'bode',
    name: 'Bode plot',
    role: 'property',
    notes: ['bode-plots-and-stability-margins', 'frequency-response'],
    cite: ['bode1945'],
  },
  responses.bode,
)
fn(
  {
    key: 'margins',
    name: 'Gain and phase margins',
    summary: 'Gain and phase margins of a loop transfer function, with their crossover frequencies.',
    role: 'property',
    notes: ['bode-plots-and-stability-margins', 'nyquist-stability-criterion'],
  },
  responses.margins,
)
fn(
  { key: 'respond', name: 'Response to an input', role: 'simulation', notes: [...SS, 'difference-equations'] },
  responses.respond,
)
fn(
  {
    key: 'stepResponse',
    name: 'Step response',
    role: 'simulation',
    notes: [...LTI, 'proportional-integral-derivative-control'],
  },
  responses.stepResponse,
)
fn(
  { key: 'impulseResponse', name: 'Impulse response', role: 'simulation', notes: [...LTI, 'convolution'] },
  responses.impulseResponse,
)
fn(
  { key: 'initialResponse', name: 'Initial-condition response', role: 'simulation', notes: SS },
  responses.initialResponse,
)

// ── Transforms and interconnections ──────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'discretiseSsm',
    name: 'State-space discretisation',
    role: 'transform',
    summary: 'Ā and B̄ of x′ = Ax + Bu sampled with step Δ by zero-order hold, bilinear or Euler; differentiable.',
    notes: ['structured-state-space-models', 'deep-state-space-models', ...SS],
    cite: ['gu2022s4'],
  },
  transform.discretiseSsm,
)
fn(
  {
    key: 'discretise',
    name: 'Discretise',
    summary: 'A continuous system sampled at Δt by zero-order hold, first-order hold, Tustin (bilinear) or Euler.',
    role: 'transform',
    returns: 'lti',
    notes: ['state-space-representation', 'z-transform', 'sampling-theorem'],
  },
  transform.discretise,
)
fn(
  {
    key: 'stateFeedback',
    name: 'Close a state-feedback loop',
    role: 'transform',
    returns: 'lti',
    notes: ['pole-placement', 'linear-quadratic-regulator'],
  },
  transform.stateFeedback,
)
fn({ key: 'series', name: 'Series connection', role: 'transform', returns: 'lti', notes: TF }, transform.series)
fn({ key: 'parallel', name: 'Parallel connection', role: 'transform', returns: 'lti', notes: TF }, transform.parallel)
fn({ key: 'feedback', name: 'Feedback connection', role: 'transform', returns: 'lti', notes: TF }, transform.feedback)
fn(
  {
    key: 'placePoles',
    name: 'Pole placement',
    summary: 'A state-feedback gain K that puts the poles of A − BK at chosen locations (single and multiple inputs).',
    role: 'solver',
    notes: ['pole-placement', 'state-observer'],
    cite: ['ackermann1972'],
  },
  placement.placePoles,
)

// ── Stability criteria ───────────────────────────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'routhArray',
    name: 'Routh array',
    summary: 'Right-half-plane roots of a polynomial from sign changes in the first column of its Routh array.',
    role: 'property',
    notes: ['routh-hurwitz-criterion'],
    cite: ['routh1877', 'hurwitz1895'],
  },
  criteria.routhArray,
)
fn(
  {
    key: 'nyquist',
    name: 'Nyquist plot',
    summary: 'L along the indented Nyquist contour and its encirclements of −1: Z = N + P closed-loop unstable poles.',
    role: 'property',
    notes: ['nyquist-stability-criterion', 'bode-plots-and-stability-margins'],
    cite: ['nyquist1932'],
  },
  criteria.nyquist,
)
fn(
  {
    key: 'rootLocus',
    name: 'Root locus',
    summary: 'Closed-loop poles of 1 + kL as k grows: branches, asymptotes, breakaway points and stability crossings.',
    role: 'property',
    notes: ['root-locus'],
    cite: ['evans1950'],
  },
  locus.rootLocus,
)
fn(
  { key: 'closedLoopPolesAt', name: 'Closed-loop poles at a gain', role: 'property', notes: ['root-locus'] },
  locus.closedLoopPolesAt,
)

// ── Identification ───────────────────────────────────────────────────────────────────────────────────────────────────

const ID = ['autoregressive-exogenous-models-and-prediction-error-methods']
fn({ key: 'arx', name: 'ARX by least squares', role: 'fit', notes: ID, cite: ['ljung1999'] }, identification.arx)
fn(
  { key: 'arxOrderSelection', name: 'ARX order selection', role: 'estimator', notes: ID, cite: ['ljung1999'] },
  identification.arxOrderSelection,
)
fn(
  {
    key: 'polynomialModel',
    name: 'Polynomial model by PEM',
    summary: 'A y = (B/F) u + C e fitted by the prediction-error method (ARX, ARMAX, OE and mixtures).',
    role: 'fit',
    notes: ID,
    cite: ['ljung1999'],
  },
  identification.polynomialModel,
)
fn({ key: 'armax', name: 'ARMAX by PEM', role: 'fit', notes: ID, cite: ['ljung1999'] }, identification.armax)
fn(
  { key: 'outputError', name: 'Output-error model by PEM', role: 'fit', notes: ID, cite: ['ljung1999'] },
  identification.outputError,
)
fn(
  {
    key: 'n4sid',
    name: 'N4SID subspace identification',
    summary: 'A state-space model from the SVD of an oblique projection of future outputs onto past data.',
    role: 'fit',
    notes: ['subspace-identification'],
    cite: ['vanoverschee1994'],
  },
  identification.n4sid,
)

/** The functions of the module, keyed by name. */
export const systemsFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>(
    'function',
    system,
    structure,
    responses,
    transform,
    placement,
    criteria,
    locus,
    identification,
  ) as Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>>

/** Every algorithm of the module, keyed by factory name. */
export const systemsAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', responses, identification) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
