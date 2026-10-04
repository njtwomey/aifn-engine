/**
 * The functions of `aifn-methods/algorithms/dynamic-programming`: knapsacks as worked dynamic programs (each
 * `…Program` is the problem for `aifn-compute/optim/programming`'s `dp` algorithm; the plain function solves it).
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as problems from './problems'

const fn = definer<FunctionInfo>('function', 'algorithms/dynamic-programming')

fn(
  { key: 'knapsackProgram', name: '0/1 knapsack as a dynamic program', role: 'construction', cite: ['bellman1957'] },
  problems.knapsackProgram,
)
fn({ key: 'knapsack', name: '0/1 knapsack', role: 'solver' }, problems.knapsack)
fn(
  { key: 'unboundedKnapsackProgram', name: 'Unbounded knapsack as a dynamic program', role: 'construction' },
  problems.unboundedKnapsackProgram,
)
fn({ key: 'unboundedKnapsack', name: 'Unbounded knapsack', role: 'solver' }, problems.unboundedKnapsack)
/** The functions of the module, keyed by name. */
export const dynamicProgrammingFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', problems) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
