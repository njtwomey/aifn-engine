/**
 * `aifn-compute/optim/second-order`: Newton, quasi-Newton and least-squares methods, which use or build curvature
 * information to take longer, better-scaled steps than gradient descent.
 *
 * - With a supplied Hessian: `newton` (damped by a line search, with a diagonal shift where the Hessian is not
 *   positive definite) and `trustRegion` (dogleg steps within an adaptive radius).
 * - From gradients alone: `bfgs` (a dense $n \times n$ inverse-Hessian approximation), `lbfgs` (the last $m$
 *   `CurvaturePair`s, $O(mn)$ a step, for large $n$), and `owlqn`, L-BFGS for $f(\xvec) + C\lVert \xvec \rVert_1$, with
 *   its `pseudoGradient`.
 * - Nonlinear least squares, $\tfrac12\lVert \rvec(\xvec) \rVert^2$ from residuals and their Jacobian
 *   (`ResidualFunction`): `gaussNewton`, `levenbergMarquardt` (damped, more robust far from the solution), and the
 *   one-call `leastSquares`.
 *
 * Each method is a step-through `Algorithm` started from `{ x0 }`, and reports non-convergence, divergence and a line
 * search that cannot make progress (`stalled`) in its state rather than throwing. `secondOrderAlgorithms` registers
 * them.
 */

export {
  newton,
  trustRegion,
  type DoglegKind,
  type NewtonOptions,
  type NewtonState,
  type TrustRegionOptions,
  type TrustRegionState,
} from './newton'
export {
  bfgs,
  lbfgs,
  owlqn,
  pseudoGradient,
  type BfgsState,
  type CurvaturePair,
  type LbfgsOptions,
  type LbfgsState,
  type OwlqnOptions,
  type OwlqnState,
  type QuasiNewtonOptions,
} from './quasiNewton'
export {
  gaussNewton,
  leastSquares,
  levenbergMarquardt,
  type GaussNewtonOptions,
  type GaussNewtonState,
  type LeastSquaresResult,
  type LeastSquaresState,
  type LevenbergMarquardtOptions,
  type LevenbergMarquardtState,
  type ResidualFunction,
} from './leastSquares'
export { secondOrderAlgorithms } from './registry'
