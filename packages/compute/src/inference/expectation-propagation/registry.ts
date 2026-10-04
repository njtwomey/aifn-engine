/**
 * The algorithms of `aifn-compute/inference/expectation-propagation`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as gaussian from './gaussian'
import * as tilted from './tilted'
import * as ep from './ep'
import * as model from './model'
import * as multivariate from './multivariate'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/expectation-propagation')

algorithm(
  {
    key: 'expectationPropagation',
    name: 'Expectation propagation',
    problem: 'gaussian-model',
    state: { iterate: 'posterior', flags: ['converged'] },
    glossary: 'ep',
    notes: ['expectation-propagation'],
    cite: ['minka2001'],
  },
  ep.expectationPropagation,
)
algorithm(
  {
    key: 'assumedDensityFiltering',
    name: 'Assumed density filtering',
    problem: 'gaussian-model',
    state: { iterate: 'posterior', objective: 'logEvidence', flags: ['converged'] },
    notes: ['assumed-density-filtering'],
    cite: ['minka2001'],
  },
  ep.assumedDensityFiltering,
)

algorithm(
  {
    key: 'multivariateExpectationPropagation',
    name: 'Expectation propagation (multivariate, rank-one sites)',
    problem: 'gaussian-model',
    state: { iterate: 'mean', objective: 'logEvidence', flags: ['converged', 'diverged'] },
    glossary: 'ep',
    notes: ['expectation-propagation'],
    cite: ['minka2001'],
  },
  multivariate.multivariateExpectationPropagation,
)

algorithm(
  {
    key: 'modelExpectationPropagation',
    name: 'Expectation propagation over a linear-Gaussian model',
    problem: 'gaussian-model',
    state: { iterate: 'means', flags: ['converged'] },
    glossary: 'ep',
    notes: ['expectation-propagation'],
    cite: ['minka2001', 'herbrich2006'],
  },
  model.modelExpectationPropagation,
)

/** Every algorithm of the module, keyed by factory name. */
export const expectationPropagationAlgorithms: Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
> = entries<AlgorithmInfo>('algorithm', ep, model, multivariate) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
>

const fn = definer<FunctionInfo>('function', 'inference/expectation-propagation')
const EP = ['expectation-propagation']

fn(
  { key: 'epLogEvidence', name: 'EP log evidence', role: 'estimator', notes: EP, cite: ['minka2001'] },
  ep.epLogEvidence,
)
fn(
  {
    key: 'compileGaussianModel',
    name: 'Compile a model for EP',
    role: 'construction',
    notes: [...EP, 'expectation-propagation-on-a-factor-graph'],
  },
  model.compileGaussianModel,
)
fn(
  { key: 'naturalGaussian', name: 'Gaussian in natural parameters', role: 'construction', notes: EP },
  gaussian.naturalGaussian,
)
fn({ key: 'gaussianMoments', name: 'Gaussian moments', role: 'transform', notes: EP }, gaussian.gaussianMoments)
fn({ key: 'multiplyGaussians', name: 'Multiply Gaussians', role: 'transform', notes: EP }, gaussian.multiplyGaussians)
fn(
  { key: 'divideGaussians', name: 'Divide Gaussians (cavity)', role: 'transform', notes: EP },
  gaussian.divideGaussians,
)
fn({ key: 'powerGaussian', name: 'Gaussian to a power', role: 'transform', notes: EP }, gaussian.powerGaussian)
fn(
  {
    key: 'dampGaussian',
    name: 'Damp a Gaussian message',
    role: 'transform',
    notes: [...EP, 'expectation-propagation-failure-modes'],
  },
  gaussian.dampGaussian,
)
fn(
  { key: 'gaussianToNormal', name: 'Gaussian message to a Normal', role: 'transform', notes: EP },
  gaussian.gaussianToNormal,
)
fn(
  { key: 'normalToGaussian', name: 'Normal to a Gaussian message', role: 'transform', notes: EP },
  gaussian.normalToGaussian,
)
fn(
  { key: 'naturalMvGaussian', name: 'Multivariate Gaussian in natural parameters', role: 'construction', notes: EP },
  gaussian.naturalMvGaussian,
)
fn(
  { key: 'mvGaussianMoments', name: 'Multivariate Gaussian moments', role: 'transform', notes: EP },
  gaussian.mvGaussianMoments,
)
fn(
  { key: 'multiplyMvGaussians', name: 'Multiply multivariate Gaussians', role: 'transform', notes: EP },
  gaussian.multiplyMvGaussians,
)
fn(
  { key: 'divideMvGaussians', name: 'Divide multivariate Gaussians', role: 'transform', notes: EP },
  gaussian.divideMvGaussians,
)
fn({ key: 'messageOf', name: 'Message of a distribution', role: 'transform', notes: EP }, gaussian.messageOf)
fn({ key: 'multiplyMessages', name: 'Multiply messages', role: 'transform', notes: EP }, gaussian.multiplyMessages)
fn({ key: 'divideMessages', name: 'Divide messages', role: 'transform', notes: EP }, gaussian.divideMessages)
fn({ key: 'dampMessages', name: 'Damp messages', role: 'transform', notes: EP }, gaussian.dampMessages)
fn({ key: 'powerMessage', name: 'Message to a power', role: 'transform', notes: EP }, gaussian.powerMessage)
fn(
  { key: 'messageToDistribution', name: 'Message to a distribution', role: 'transform', notes: EP },
  gaussian.messageToDistribution,
)
fn(
  {
    key: 'probitTilted',
    name: 'Probit tilted moments',
    role: 'inference',
    notes: ['expectation-propagation-probit-regression', 'expectation-propagation-gaussian-process-classification'],
  },
  tilted.probitTilted,
)
fn(
  {
    key: 'stepTilted',
    name: 'Step-likelihood tilted moments',
    role: 'inference',
    notes: ['expectation-propagation-truncated-gaussian', 'bayes-point-machine'],
  },
  tilted.stepTilted,
)
fn(
  {
    key: 'intervalTilted',
    name: 'Interval tilted moments',
    role: 'inference',
    notes: ['expectation-propagation-truncated-gaussian'],
  },
  tilted.intervalTilted,
)
fn(
  { key: 'tiltedByQuadrature', name: 'Tilted moments by quadrature', role: 'inference', notes: EP },
  tilted.tiltedByQuadrature,
)
fn({ key: 'lift', name: 'Lift tilted moments to power EP', role: 'transform', notes: EP }, tilted.lift)

/** The functions of the module, keyed by name. */
export const expectationPropagationFunctions: Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
> = entries<FunctionInfo>('function', ep, model, gaussian, tilted) as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
>
