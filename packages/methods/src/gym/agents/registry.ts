/**
 * The registry of the `aifn-methods/gym/agents` group's shared layer besides the agents (registered with
 * `agentRegistry`): dynamic-programming planners as traceable algorithms, and bandit and DQN functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as bandits from './bandits'
import * as dqn from './dqn'
import * as planning from './planning'
import * as tabular from './tabular'

/** A registry table: entries of info kind `I`, keyed by factory or function name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'gym/agents')
const fn = definer<FunctionInfo>('function', 'gym/agents')
const MDP = ['markov-decision-process', 'bellman-equations']
const values = { iterate: 'V', objective: 'residual', flags: ['converged'] } as const

algorithm(
  {
    key: 'valueIteration',
    name: 'Value iteration',
    summary: 'Bellman optimality backups V ← max_a [r + γ P V] until the residual is small.',
    problem: 'dynamic-program',
    state: values,
    notes: ['value-iteration', ...MDP],
    cite: ['bellman1957', 'sutton2018'],
  },
  planning.valueIteration,
)
algorithm(
  {
    key: 'policyIteration',
    name: 'Policy iteration',
    summary: 'Evaluate the policy exactly, then improve it greedily, until it stops changing.',
    problem: 'dynamic-program',
    state: { iterate: 'V', flags: ['converged'] },
    notes: ['policy-iteration', ...MDP],
    cite: ['howard1960', 'sutton2018'],
  },
  planning.policyIteration,
)
algorithm(
  {
    key: 'policyEvaluation',
    name: 'Iterative policy evaluation',
    problem: 'dynamic-program',
    state: values,
    notes: ['bellman-equations', 'policy-iteration'],
    cite: ['sutton2018'],
  },
  planning.policyEvaluation,
)
fn(
  {
    key: 'evaluatePolicy',
    name: 'Evaluate a policy exactly',
    summary: 'V = (I − γP_π)⁻¹ r_π by one linear solve.',
    role: 'solver',
    notes: ['bellman-equations'],
  },
  planning.evaluatePolicy,
)
fn({ key: 'greedyPolicy', name: 'Greedy policy of Q', role: 'transform', notes: MDP }, planning.greedyPolicy)
fn({ key: 'valuesFromQ', name: 'Values from Q', role: 'transform', notes: MDP }, planning.valuesFromQ)
fn(
  { key: 'greedyPath', name: 'Greedy path through a grid', role: 'inference', notes: ['solving-a-maze'] },
  tabular.greedyPath,
)
fn(
  {
    key: 'klBernoulli',
    name: 'Bernoulli KL divergence',
    role: 'property',
    notes: ['kullback-leibler-upper-confidence-bound'],
  },
  bandits.klBernoulli,
)
fn(
  {
    key: 'klUcbIndex',
    name: 'KL-UCB index',
    summary: 'The largest q with n·KL(p̂, q) ≤ log t, by bisection.',
    role: 'solver',
    notes: ['kullback-leibler-upper-confidence-bound'],
    cite: ['garivier2011'],
  },
  bandits.klUcbIndex,
)
fn(
  {
    key: 'laiRobbinsBound',
    name: 'Lai–Robbins regret bound',
    role: 'property',
    notes: ['multi-armed-bandit', 'kullback-leibler-upper-confidence-bound'],
    cite: ['lai1985'],
  },
  bandits.laiRobbinsBound,
)
fn(
  { key: 'replayBuffer', name: 'Replay buffer', role: 'construction', notes: ['deep-q-network'], cite: ['mnih2015'] },
  dqn.replayBuffer,
)
fn(
  {
    key: 'tdTargets',
    name: 'TD targets',
    tex: "r + \\gamma \\max_{a'} Q_{\\bar\\theta}(s', a')",
    role: 'estimator',
    notes: ['deep-q-network', 'q-learning'],
    cite: ['mnih2015'],
  },
  dqn.tdTargets,
)
fn({ key: 'qNetwork', name: 'Q-network', role: 'construction', notes: ['deep-q-network'] }, dqn.qNetwork)

/** The algorithms of the group's shared layer, keyed by factory name. */
export const planningAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  planning,
) as Table<AlgorithmInfo>
/** The functions of the group's shared layer, keyed by name. */
export const agentFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  planning,
  tabular,
  bandits,
  dqn,
) as Table<FunctionInfo>
