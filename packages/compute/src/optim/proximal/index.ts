/**
 * `aifn-compute/optim/proximal`: proximal and projected methods for composite objectives
 * $F(\xvec) = f(\xvec) + g(\xvec)$, with $f$ smooth and $g$ convex but possibly nonsmooth or an indicator of a
 * constraint set.
 *
 * - Solvers: `proximalGradient`, with `ista` (plain) and `fista` (Nesterov-accelerated, $O(1/k^2)$) as its two forms,
 *   each with a fixed step or backtracking; `projectedGradient` for $f$ over a closed convex set.
 * - Proximal operators, as a `Prox` (the value of $g$ and its prox): `proxL1` (soft thresholding), `proxL2` (block soft
 *   thresholding, as in the group lasso), `proxSquaredL2` (ridge), `proxBox` and `proxNonnegative` (indicators), and
 *   `proxZero` (plain gradient descent).
 * - Projections, as functions from a point to its projection: `projectBox`, `projectBall`, `projectNonnegative`,
 *   `projectSimplex`, `projectSimplexRows` (each row of a matrix onto a simplex) and `projectGroupSums` (fixed sums
 *   over disjoint groups).
 * - Intersections of convex sets: `alternatingProjections` cycles through the projections onto each set until the
 *   point settles, and with Dykstra's correction finds the closest point of the intersection to the start;
 *   `alternatingProjectionsSteps` is the same as a step algorithm.
 * - `proximalAlgorithms` and `proximalFunctions` register the methods and operators for generic views and workers.
 *
 * The solvers are `Algorithm`s started with `{ x0 }`, to be run with `run` or `trace`; the smooth part $f$ returns
 * `{ value, grad }`. Divergence is reported in `diverged`, not thrown. Operators and projections take and return
 * vectors and do not modify their input.
 */

export {
  fista,
  ista,
  projectBall,
  projectBox,
  projectNonnegative,
  projectSimplex,
  projectSimplexRows,
  projectedGradient,
  proximalGradient,
  proxBox,
  proxL1,
  proxL2,
  proxNonnegative,
  proxSquaredL2,
  proxZero,
  type Bound,
  type Prox,
  type ProximalGradientOptions,
  type ProximalGradientState,
} from './proximal'
export {
  alternatingProjections,
  alternatingProjectionsSteps,
  projectGroupSums,
  type AlternatingProjectionsOptions,
  type AlternatingProjectionsResult,
  type AlternatingProjectionsState,
  type Projection,
} from './alternating'
export { proximalAlgorithms, proximalFunctions } from './registry'
