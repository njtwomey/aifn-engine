/**
 * The registry of `aifn-methods/inference/classifier-models`: the catalogue entries (name, summary, role, notes and
 * citations) of the Bayes point machine, as an algorithm, and of its prediction and the AdPredictor functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as adp from './adpredictor'
import * as bpm from './bayesPointMachine'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/classifier-models')
const fn = definer<FunctionInfo>('function', 'inference/classifier-models')
const notes = ['bayes-point-machine', 'expectation-propagation-probit-regression', 'expectation-propagation']

algorithm(
  {
    key: 'bayesPointMachine',
    name: 'Bayes point machine (EP)',
    summary: 'A Gaussian posterior over linear-classifier weights by expectation propagation, one site per example.',
    problem: 'factor-graph',
    state: { iterate: 'mean', flags: ['converged'] },
    notes,
    cite: ['herbrich2001', 'minka2001'],
  },
  bpm.bayesPointMachine,
)
fn(
  { key: 'bayesPointMachinePredict', name: 'Bayes point machine prediction', role: 'inference', notes },
  bpm.bayesPointMachinePredict,
)

const AD = ['adpredictor', 'model-based-machine-learning']
fn(
  {
    key: 'adPredictor',
    name: 'AdPredictor model',
    summary: 'Gaussian weights per sparse feature value for Bayesian probit click-through prediction.',
    role: 'construction',
    notes: AD,
    cite: ['graepel2010adpredictor'],
  },
  adp.adPredictor,
)
fn(
  {
    key: 'adPredictorUpdate',
    name: 'AdPredictor update',
    summary: 'Assumed-density filtering of the active weights after one impression, with truncated-Gaussian v and w.',
    role: 'inference',
    notes: AD,
    cite: ['graepel2010adpredictor'],
  },
  adp.adPredictorUpdate,
)
fn(
  {
    key: 'adPredictorProbability',
    name: 'AdPredictor click probability',
    role: 'property',
    notes: AD,
    cite: ['graepel2010adpredictor'],
  },
  adp.adPredictorProbability,
)

/** The algorithms of the module. */
export const classifierModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  bpm,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const classifierModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  bpm,
  adp,
) as Table<FunctionInfo>
