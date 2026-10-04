/**
 * `aifn-compute/optim/search`: search over a space given by a refinement operator (a root, each node's specialisations, a
 * quality and an optional optimistic estimate): beam, best-first, depth-first and breadth-first search as one
 * step-through algorithm, branch-and-bound pruning, exhaustive search to a depth, and a top-k result set with a
 * redundancy filter. Subgroup discovery and rule learning search their description languages with it.
 */

export {
  filterRedundant,
  offerResult,
  refinementSearch,
  refinementSearchSteps,
  searchHistory,
  type RedundancyTest,
  type SearchHistory,
  type SearchOptions,
  type SearchSpace,
  type SearchState,
  type SearchStrategy,
  type SearchVisit,
  type VisitFate,
} from './search'
export { searchAlgorithms, searchFunctions } from './registry'
