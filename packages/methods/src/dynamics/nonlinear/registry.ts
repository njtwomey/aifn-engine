/**
 * The functions of `aifn-methods/dynamics/nonlinear`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as flow from './flow'
import * as lyapunov from './lyapunov'

const fn = definer<FunctionInfo>('function', 'dynamics/nonlinear')

fn(
  {
    key: 'poincareSection',
    name: 'Poincaré section',
    summary: 'The crossings of a trajectory with a section: the orbit of the first-return map.',
    role: 'solver',
    notes: ['vector-fields-and-flows'],
    cite: ['strogatz2015'],
  },
  flow.poincareSection,
)
fn(
  {
    key: 'limitCycle',
    name: 'Limit cycle',
    summary: 'A periodic orbit found as a fixed point of the return map, with its Floquet multiplier.',
    role: 'solver',
    notes: ['stability-and-equilibria', 'vector-fields-and-flows'],
    cite: ['strogatz2015'],
  },
  flow.limitCycle,
)
fn(
  {
    key: 'lyapunovCheck',
    name: "Lyapunov's conditions on a grid",
    summary: 'V positive definite and V̇ = ∇V·f ≤ 0 checked on a grid about an equilibrium.',
    role: 'property',
    notes: ['lyapunov-stability'],
    cite: ['lyapunov1992'],
  },
  lyapunov.lyapunovCheck,
)

/** The functions of the module, keyed by name. */
export const nonlinearFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', flow, lyapunov) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
