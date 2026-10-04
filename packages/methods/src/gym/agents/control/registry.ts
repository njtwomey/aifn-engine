/** The functions of `aifn-methods/gym/agents/control`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as swingUp from './swing-up'

const fn = definer<FunctionInfo>('function', 'gym/agents/control')

fn(
  {
    key: 'lineariseDynamics',
    name: 'Linearise the dynamics at a point',
    role: 'transform',
    notes: ['jacobian-linearisation'],
  },
  swingUp.lineariseDynamics,
)
fn(
  { key: 'pendulumLqr', name: 'LQR for the inverted pendulum', role: 'solver', notes: ['linear-quadratic-regulator'] },
  swingUp.pendulumLqr,
)

/** The functions of the module, keyed by name. */
export const controlAgentFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', swingUp) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
