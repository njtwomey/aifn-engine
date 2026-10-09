/**
 * The algorithms and functions of `aifn-compute/inference/exact`, registered with the notes they serve. The stepped
 * engines (enumeration, variable elimination, forward–backward, Viterbi, chain sum–product) are registered with what
 * each factory takes (`problem`) and the roles of its state's fields (`state`: iterate, objective, grad, stepSize,
 * and the `Status` flags it sets), so a generic trace view picks default series and a worker can address an algorithm
 * by key (design S §2.3); the one-call functions with their role.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as chain from './chain'
import * as exact from './exact'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/exact')

algorithm(
  {
    key: 'enumerationSteps',
    name: 'Enumeration',
    summary: 'Exact inference by enumerating every assignment.',
    problem: 'factor-graph',
    state: { iterate: 'logMarginals', objective: 'logZ', flags: [] },
    notes: ['variable-elimination'],
  },
  exact.enumerationSteps,
)
algorithm(
  {
    key: 'variableEliminationSteps',
    name: 'Variable elimination',
    problem: 'factor-graph',
    state: { flags: [] },
    notes: ['variable-elimination'],
    cite: ['zhang1994'],
  },
  exact.variableEliminationSteps,
)
algorithm(
  {
    key: 'forwardBackwardSteps',
    name: 'Forward–backward',
    problem: 'chain',
    state: { iterate: 'marginals', objective: 'logLikelihood', flags: [] },
    notes: ['hidden-markov-model'],
    cite: ['baum1970'],
  },
  chain.forwardBackwardSteps,
)
algorithm(
  {
    key: 'viterbiSteps',
    name: 'Viterbi',
    problem: 'chain',
    state: { iterate: 'path', flags: [] },
    glossary: 'viterbi',
    notes: ['hidden-markov-model'],
    cite: ['viterbi1967'],
  },
  chain.viterbiSteps,
)
algorithm(
  {
    key: 'chainSumProduct',
    name: 'Chain sum–product',
    summary: 'Sum–product (or max–product) on a chain-shaped factor graph.',
    problem: 'factor-graph',
    state: { iterate: 'marginals', objective: 'logZ', flags: [] },
    notes: ['belief-propagation', 'factor-graph'],
  },
  chain.chainSumProduct,
)

/** Every algorithm of the module, keyed by factory name. */
export const exactAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', chain, exact) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >

const fn = definer<FunctionInfo>('function', 'inference/exact')
const HMM = ['hidden-markov-model']

fn(
  {
    key: 'forwardBackward',
    name: 'Forward–backward',
    summary: 'Posterior marginals of a chain from scaled forward and backward messages.',
    role: 'inference',
    notes: HMM,
    cite: ['rabiner1989'],
  },
  chain.forwardBackward,
)
fn(
  {
    key: 'viterbi',
    name: 'Viterbi decoding',
    summary: 'The most probable state path by max-product dynamic programming.',
    role: 'inference',
    notes: HMM,
    cite: ['rabiner1989'],
  },
  chain.viterbi,
)
fn(
  {
    key: 'chainForwardBackward',
    name: 'Forward–backward on log-potentials',
    role: 'inference',
    notes: [...HMM, 'conditional-random-field'],
  },
  chain.chainForwardBackward,
)
fn(
  {
    key: 'chainViterbi',
    name: 'Viterbi on log-potentials',
    role: 'inference',
    notes: [...HMM, 'conditional-random-field'],
  },
  chain.chainViterbi,
)
fn(
  {
    key: 'posteriorDecode',
    name: 'Posterior (max-marginal) decoding',
    summary: 'Each position takes its most probable label under the marginals: the most correct labels in expectation.',
    role: 'inference',
    notes: [...HMM, 'conditional-random-field'],
    cite: ['rabiner1989'],
  },
  chain.posteriorDecode,
)
fn(
  {
    key: 'sampleHiddenPath',
    name: 'Forward filtering, backward sampling',
    role: 'simulation',
    random: true,
    notes: HMM,
  },
  chain.sampleHiddenPath,
)
fn(
  { key: 'factorChain', name: 'Chain as a factor graph', role: 'construction', notes: ['factor-graph'] },
  chain.factorChain,
)
fn(
  {
    key: 'enumerate',
    name: 'Inference by enumeration',
    role: 'inference',
    notes: ['bayesian-network', 'variable-elimination'],
    cite: ['koller2009'],
  },
  exact.enumerate,
)
fn(
  { key: 'jointDistribution', name: 'Joint distribution table', role: 'inference', notes: ['bayesian-network'] },
  exact.jointDistribution,
)
fn(
  {
    key: 'variableElimination',
    name: 'Variable elimination',
    role: 'inference',
    notes: ['variable-elimination'],
    cite: ['zhang1994', 'dechter1999'],
  },
  exact.variableElimination,
)
fn(
  { key: 'eliminationResult', name: 'Variable elimination result', role: 'inference', notes: ['variable-elimination'] },
  exact.eliminationResult,
)

/** The functions of the module, keyed by name. */
export const exactFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', chain, exact) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
