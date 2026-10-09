/**
 * `aifn-compute/optim/line-search`: line searches, which choose a step length $\alpha$ along a descent direction
 * $\pvec$ from $\xvec$.
 *
 * - Searches on vectors: `backtracking` (shrink $\alpha$ until sufficient decrease, the Armijo condition) and
 *   `strongWolfe` (bracket and zoom until sufficient decrease and strong curvature hold, as quasi-Newton and conjugate
 *   gradient methods need).
 * - The same searches on float64 working arrays, for the inner loops of the other optimisers: `backtrackingSearch` and
 *   `strongWolfeSearch`, returning a `SearchOutcome`.
 *
 * Every trial is recorded in the `LineSearchResult`. Failure (an uphill direction, or no acceptable step within the
 * trial budget) is reported by `converged: false`, never thrown, and the best trial below $f(\xvec)$ is returned.
 * `lineSearchFunctions` registers the searches.
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
