/**
 * The algorithms and functions of `aifn-compute/inference/variational`. `bbvi` is registered with what its factory
 * takes (`problem`) and the roles of its state's fields (`state`: iterate, objective, grad, and the `Status` flags it
 * sets), so a generic trace view picks default series and a worker can address it by key (design S §2.3); the
 * families and the ELBO estimators are registered as functions with the notes they serve.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as elbo from './elbo'
import * as family from './family'
import * as bbvi from './bbvi'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/variational')

algorithm(
  {
    key: 'bbvi',
    name: 'Black-box variational inference',
    problem: 'log-density',
    state: { iterate: 'mean', objective: 'elbo', grad: 'grad', flags: ['diverged'] },
    random: true,
    glossary: 'vi',
    notes: ['black-box-variational-inference', 'variational-inference'],
    cite: ['ranganath2014'],
  },
  bbvi.bbvi,
)

/** Every algorithm of the module, keyed by factory name. */
export const variationalAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', bbvi) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'inference/variational')
const VI = ['variational-inference', 'evidence-lower-bound']

fn(
  {
    key: 'meanFieldGaussian',
    name: 'Mean-field Gaussian family',
    role: 'construction',
    notes: ['mean-field-variational-inference', 'black-box-variational-inference'],
  },
  family.meanFieldGaussian,
)
fn(
  {
    key: 'fullRankGaussian',
    name: 'Full-rank Gaussian family',
    role: 'construction',
    notes: ['black-box-variational-inference'],
  },
  family.fullRankGaussian,
)
fn(
  { key: 'elbo', name: 'ELBO (Monte Carlo)', role: 'estimator', random: true, notes: VI, cite: ['blei2017'] },
  elbo.elbo,
)
fn(
  {
    key: 'elboGradient',
    name: 'ELBO gradient (reparameterised)',
    role: 'estimator',
    random: true,
    notes: ['reparameterisation-trick', 'black-box-variational-inference'],
    cite: ['kucukelbir2017'],
  },
  elbo.elboGradient,
)
fn(
  {
    key: 'gradientVariance',
    name: 'Gradient-estimator variance',
    role: 'estimator',
    random: true,
    notes: ['score-function-estimator', 'reparameterisation-trick'],
  },
  elbo.gradientVariance,
)

/** The functions of the module, keyed by name. */
export const variationalFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', family, elbo) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
