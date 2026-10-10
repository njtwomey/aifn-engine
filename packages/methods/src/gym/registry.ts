/**
 * The registry entries of the functions of the `aifn-methods/gym` area's shared layer: MDP tables, rollouts and
 * training, kind `function`.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as mdp from './mdp'
import * as rollout from './rollout'
import * as train from './train'

/** Registers a function of `gym`. */
const fn = definer<FunctionInfo>('function', 'gym')
const MDP = ['markov-decision-process']

fn(
  {
    key: 'tabularMdp',
    name: 'Tabular MDP',
    summary: 'Transition and reward tables of an environment with a model.',
    role: 'construction',
    notes: MDP,
  },
  mdp.tabularMdp,
)
fn(
  { key: 'optimalValues', name: 'Optimal values', role: 'solver', notes: ['value-iteration', 'bellman-equations'] },
  mdp.optimalValues,
)
fn({ key: 'qFromValues', name: 'Q from values', role: 'transform', notes: ['bellman-equations'] }, mdp.qFromValues)
fn({ key: 'policyMatrix', name: 'Policy matrix', role: 'transform', notes: MDP }, mdp.policyMatrix)
fn({ key: 'rollout', name: 'Roll out an agent', role: 'simulation', random: true, notes: MDP }, rollout.rollout)
fn({ key: 'runEpisode', name: 'Run an episode', role: 'simulation', random: true, notes: MDP }, rollout.runEpisode)
fn({ key: 'episodes', name: 'Run episodes', role: 'simulation', random: true, notes: MDP }, rollout.episodes)
fn(
  { key: 'compare', name: 'Compare agents', role: 'simulation', random: true, notes: ['multi-armed-bandit'] },
  rollout.compare,
)
fn({ key: 'train', name: 'Train an agent', role: 'fit', random: true, notes: ['q-learning'] }, train.train)
fn({ key: 'evaluateEpisode', name: 'Evaluate an episode', role: 'estimator', random: true }, train.evaluateEpisode)

/** The functions of the area's shared layer, keyed by name. */
export const gymFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', mdp, rollout, train) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
