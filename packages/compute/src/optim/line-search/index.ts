/**
 * `aifn-compute/optim/line-search`: line searches: backtracking (Armijo) and strong Wolfe. `backtracking` and
 * `strongWolfe` take vectors; `backtrackingSearch` and `strongWolfeSearch` are the same searches on float64 working
 * arrays, for the inner loops of the other optimisers.
 */

export {
  backtracking,
  backtrackingSearch,
  strongWolfe,
  strongWolfeSearch,
  type SearchOutcome,
  type BacktrackingOptions,
  type LineSearchResult,
  type LineSearchTrial,
  type StrongWolfeOptions,
} from './lineSearch'
export { lineSearchFunctions } from './registry'
