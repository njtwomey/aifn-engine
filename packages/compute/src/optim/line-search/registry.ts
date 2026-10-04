/**
 * The functions of `aifn-compute/optim/line-search`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as lineSearch from './lineSearch'

const fn = definer<FunctionInfo>('function', 'optim/line-search')

fn(
  {
    key: 'backtracking',
    name: 'Backtracking (Armijo) line search',
    role: 'solver',
    notes: ['line-search'],
    cite: ['nocedal2006'],
  },
  lineSearch.backtracking,
)
fn(
  {
    key: 'strongWolfe',
    name: 'Strong Wolfe line search',
    role: 'solver',
    notes: ['line-search', 'quasi-newton-methods'],
    cite: ['nocedal2006'],
  },
  lineSearch.strongWolfe,
)
fn(
  { key: 'backtrackingSearch', name: 'Backtracking step size rule', role: 'construction', notes: ['line-search'] },
  lineSearch.backtrackingSearch,
)
fn(
  { key: 'strongWolfeSearch', name: 'Strong Wolfe step size rule', role: 'construction', notes: ['line-search'] },
  lineSearch.strongWolfeSearch,
)

/** The functions of the module, keyed by name. */
export const lineSearchFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', lineSearch) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
