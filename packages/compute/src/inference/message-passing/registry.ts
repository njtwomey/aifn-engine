/**
 * The algorithms of `aifn-compute/inference/message-passing`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bp from './bp'
import * as gaussianBp from './gaussianBp'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/message-passing')

algorithm(
  {
    key: 'beliefPropagationSteps',
    name: 'Belief propagation',
    problem: 'factor-graph',
    state: { iterate: 'beliefs', objective: 'change', flags: ['converged'] },
    notes: ['belief-propagation', 'loopy-belief-propagation'],
    cite: ['pearl1988'],
  },
  bp.beliefPropagationSteps,
)
algorithm(
  {
    key: 'gaussianBeliefPropagationSteps',
    name: 'Gaussian belief propagation',
    problem: 'gaussian-model',
    state: { iterate: 'means', objective: 'change', flags: ['converged', 'diverged'] },
    notes: ['belief-propagation'],
  },
  gaussianBp.gaussianBeliefPropagationSteps,
)

/** Every algorithm of the module, keyed by factory name. */
export const messagePassingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', bp, gaussianBp) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'inference/message-passing')
const BP = ['belief-propagation', 'loopy-belief-propagation']

fn(
  {
    key: 'beliefPropagation',
    name: 'Belief propagation',
    role: 'inference',
    notes: [...BP, 'factor-graph'],
    cite: ['pearl1988', 'kschischang2001'],
  },
  bp.beliefPropagation,
)
fn({ key: 'factorBeliefs', name: 'Factor beliefs', role: 'inference', notes: BP }, bp.factorBeliefs)
fn(
  {
    key: 'betheLogZ',
    name: 'Bethe free energy (log Z)',
    role: 'estimator',
    notes: ['loopy-belief-propagation'],
    cite: ['yedidia2005'],
  },
  bp.betheLogZ,
)
fn({ key: 'decodeBeliefs', name: 'Decode beliefs (max-marginals)', role: 'inference', notes: BP }, bp.decodeBeliefs)
fn(
  { key: 'gaussianBeliefPropagation', name: 'Gaussian belief propagation', role: 'inference', notes: BP },
  gaussianBp.gaussianBeliefPropagation,
)

/** The functions of the module, keyed by name. */
export const messagePassingFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', bp, gaussianBp) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
