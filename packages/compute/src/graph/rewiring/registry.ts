/** The functions of `aifn-compute/graph/rewiring`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as rewiring from './rewiring'

const fn = definer<FunctionInfo>('function', 'graph/rewiring')
const SQUASHING = ['oversquashing-and-graph-rewiring']

fn(
  {
    key: 'withVirtualNode',
    name: 'Virtual node',
    summary: 'Join one new node to every node, so that every pair of nodes is at most two hops apart.',
    role: 'transform',
    notes: SQUASHING,
    cite: ['gilmer2017'],
  },
  rewiring.withVirtualNode,
)
fn(
  {
    key: 'fosrRewire',
    name: 'First-order spectral rewiring (FoSR)',
    tex: '(i, j) = \\arg\\min_{(i, j) \\notin E} \\frac{x_i x_j}{\\sqrt{(1 + d_i)(1 + d_j)}}',
    summary: 'Add, one by one, the non-edges that to first order most increase the spectral gap.',
    role: 'transform',
    notes: SQUASHING,
    cite: ['karhadkar2023fosr'],
  },
  rewiring.fosrRewire,
)

/** The functions of the module, keyed by name. */
export const rewiringFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', rewiring) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
