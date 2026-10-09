/**
 * The registry of `aifn-compute/optim/search`: refinement-operator search as a step-through algorithm (`problem:
 * 'search-space'`), the run-to-the-end form and the result-set helpers as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as search from './search'

const MODULE = 'optim/search'
const algorithm = definer<AlgorithmInfo>('algorithm', MODULE)
const fn = definer<FunctionInfo>('function', MODULE)

algorithm(
  {
    key: 'refinementSearchSteps',
    name: 'Refinement search (beam, best-first, depth-first, breadth-first; branch and bound)',
    summary:
      'Search a space given by a root and a refinement operator for the top k nodes by quality; an optimistic estimate prunes branches that cannot enter the top k.',
    problem: 'search-space',
    state: { iterate: 'frontier', objective: 'best', flags: ['terminated'] },
    notes: ['decoding-strategies'],
  },
  search.refinementSearchSteps,
)
fn(
  {
    key: 'refinementSearch',
    name: 'Refinement search, run to the end',
    summary: 'The final state of refinementSearchSteps: the top k nodes, with counts of nodes evaluated and pruned.',
    role: 'solver',
  },
  search.refinementSearch,
)
fn(
  {
    key: 'offerResult',
    name: 'Top-k result set with a redundancy filter',
    summary:
      'Offer a node to a top-k list; a redundancy test keeps only the better of two nodes describing the same pattern.',
    role: 'construction',
  },
  search.offerResult,
)
fn(
  {
    key: 'filterRedundant',
    name: 'Redundancy filter',
    summary: 'The best nodes of a list, skipping any redundant with a better one already kept.',
    role: 'transform',
  },
  search.filterRedundant,
)
fn(
  {
    key: 'searchHistory',
    name: 'Search history',
    summary: 'The search tree recorded by a run of refinementSearchSteps: every node with its fate and step.',
    role: 'transform',
  },
  search.searchHistory,
)

/** A registry table: the entries of one kind, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>

/** The algorithm of the module, keyed by factory name. */
export const searchAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  search,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const searchFunctions: Table<FunctionInfo> = entries<FunctionInfo>('function', search) as Table<FunctionInfo>
