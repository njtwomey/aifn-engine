/**
 * `aifn-compute/optim/search`: search over a space given by a refinement operator (a root, each node's
 * specialisations, a quality to maximise and an optional optimistic estimate), for the top $k$ nodes by quality.
 *
 * - Step by step: `refinementSearchSteps` runs beam, best-first, depth-first or breadth-first search as one
 *   step-through algorithm, with branch-and-bound pruning when the space bounds the quality of a node's refinements,
 *   and exhaustive search to a depth without it; each state records what the step expanded, generated, pruned and
 *   discarded, and `searchHistory` assembles the search tree from a trace.
 * - To the end: `refinementSearch` returns the final state, with the top $k$ and the counts of nodes evaluated and
 *   pruned.
 * - Result sets: `offerResult` keeps a top-$k$ list as nodes arrive, and `filterRedundant` thins a list after the
 *   fact; both take a redundancy test, so two nodes describing the same pattern keep only the better.
 * - `searchAlgorithms` and `searchFunctions` register the search and the helpers for generic views and workers.
 *
 * Nodes are plain data, kept in the states, and a node reached twice (by its key) is evaluated once. Invalid options
 * throw `DomainError` when the search is built. Subgroup discovery and rule learning search their description
 * languages with this module.
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
