/** The environment constructors of the `aifn-methods/gym/environments` group's shared layer. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as gridworlds from './gridworlds'

const fn = definer<FunctionInfo>('function', 'gym/environments')
const MDP = ['markov-decision-process']

fn(
  { key: 'gridworld', name: 'Gridworld', role: 'construction', notes: MDP, cite: ['sutton2018'] },
  gridworlds.gridworld,
)
fn(
  {
    key: 'cliffWalking',
    name: 'Cliff walking',
    role: 'construction',
    notes: [...MDP, 'sarsa', 'q-learning'],
    cite: ['sutton2018'],
  },
  gridworlds.cliffWalking,
)
fn({ key: 'frozenLake', name: 'Frozen lake', role: 'construction', notes: MDP }, gridworlds.frozenLake)
fn({ key: 'maze', name: 'Maze', role: 'construction', notes: ['solving-a-maze', ...MDP] }, gridworlds.maze)
fn(
  { key: 'mdpEnvironment', name: 'Environment from MDP tables', role: 'construction', notes: MDP },
  gridworlds.mdpEnvironment,
)

/** The functions of the group's shared layer, keyed by name. */
export const environmentFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', gridworlds) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
