/**
 * `aifn-compute/optim/proximal`: proximal and projected methods: ISTA, FISTA, projected gradient, proximal operators (L1, L2,
 * squared L2, box, non-negative) and projections (ball, box, simplex, non-negative orthant, fixed group sums); alternating projections onto an
 * intersection of convex sets (with Dykstra's correction).
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
