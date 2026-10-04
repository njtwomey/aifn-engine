/**
 * The registry of `aifn-methods/inference/mixture-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as clutter from './clutter'
import * as mixture from './mixture'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/mixture-models')
const fn = definer<FunctionInfo>('function', 'inference/mixture-models')
const CLUTTER = ['expectation-propagation-clutter-problem', 'expectation-propagation']

algorithm(
  {
    key: 'caviGaussianMixture',
    name: 'Variational Bayesian Gaussian mixture',
    summary: 'CAVI on q(π) q(μ, Λ) q(z) for a Gaussian mixture; the ELBO never decreases.',
    problem: 'gaussian-model',
    state: { objective: 'elbo', flags: ['converged', 'diverged'] },
    random: true,
    notes: ['gaussian-mixture-model', 'mean-field-variational-inference', 'variational-inference'],
    cite: ['bishop2006'],
  },
  mixture.caviGaussianMixture,
)
fn(
  {
    key: 'mixturePredictiveDensity',
    name: 'Mixture predictive density',
    role: 'inference',
    notes: ['gaussian-mixture-model'],
  },
  mixture.mixturePredictiveDensity,
)
fn(
  { key: 'clutterModel', name: 'Clutter model', role: 'construction', notes: CLUTTER, cite: ['minka2001'] },
  clutter.clutterModel,
)
fn(
  { key: 'clutterTilted', name: 'Clutter tilted moments', role: 'inference', notes: CLUTTER, cite: ['minka2001'] },
  clutter.clutterTilted,
)
fn(
  { key: 'clutterLogLikelihood', name: 'Clutter log-likelihood', role: 'estimator', notes: CLUTTER },
  clutter.clutterLogLikelihood,
)
fn(
  { key: 'sampleClutter', name: 'Sample the clutter problem', role: 'simulation', random: true, notes: CLUTTER },
  clutter.sampleClutter,
)
fn(
  { key: 'clutterEp', name: 'EP options for the clutter problem', role: 'construction', notes: CLUTTER },
  clutter.clutterEp,
)
fn(
  { key: 'clutterPosterior', name: 'Exact clutter posterior on a grid', role: 'inference', notes: CLUTTER },
  clutter.clutterPosterior,
)

/** The algorithms of the module. */
export const mixtureModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  mixture,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const mixtureModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  mixture,
  clutter,
) as Table<FunctionInfo>
