/**
 * `aifn-compute/optim`: optimisation, as torch.optim, optax and scipy.optimize.
 *
 * The child modules:
 *
 * - `line-search`: step lengths along a descent direction, by backtracking (Armijo) or to the strong Wolfe conditions,
 *   for the inner loops of the other methods.
 * - `first-order`: methods using $f$ and $\nabla f$ only (gradient descent, momentum, Nesterov, AdaGrad, RMSProp,
 *   Adam, conjugate gradients, coordinate descent), and the update rules on parameter pytrees behind them, for
 *   training loops.
 * - `second-order`: Newton and trust-region methods with a Hessian, the quasi-Newton BFGS, L-BFGS and OWL-QN, and
 *   Gauss–Newton and Levenberg–Marquardt for nonlinear least squares.
 * - `proximal`: proximal and projected gradient methods for $f(\xvec) + g(\xvec)$ with $g$ nonsmooth or a constraint,
 *   with proximal operators and projections.
 * - `derivative-free`: minimisation from values alone (Nelder–Mead, simulated annealing, CMA-ES).
 * - `programming`: linear, quadratic and mixed-integer programmes, assignment, and dynamic programming.
 * - `online`: online learning, with expert advice and online convex optimisation, and the regret of each.
 * - `search`: search over a space given by a refinement operator, for the best $k$ nodes.
 * - `minimize`: one entry point that runs any of the methods above by name, as scipy.optimize.minimize.
 *
 * The shared layer, exported here, holds the objective and option types (`ObjectiveFn`, `StoppingOptions`,
 * `StartOptions`, `RunOptions`), `objectiveFn` to differentiate an `Objective` for the gradient methods, and the
 * step-size schedules `inverseTimeDecay`, `exponentialDecay` and `inverseSqrtDecay`. Every iterative method is an
 * `Algorithm` started from `{ x0 }`, run with `run` or `trace`, and reports convergence and divergence in its state.
 */

export {
  type Evaluation,
  type Hessian,
  type IterateState,
  type MatrixLike,
  type ObjectiveFn,
  type Schedule,
  type StoppingOptions,
  type ValueFunction,
  type VectorLike,
} from 'aifn-compute/foundation/contracts'
export { objectiveFn, type RunOptions, type StartOptions } from './options'
export { exponentialDecay, inverseSqrtDecay, inverseTimeDecay, scheduleFunctions } from './schedules'
export { minimize } from './minimize'
export {
  adam,
  adamRule,
  applyUpdates,
  gradientDescent,
  momentum,
  sgdRule,
  type StepSize,
  type UpdateRule,
} from './first-order'
export { newton, lbfgs } from './second-order'
export { nelderMead } from './derivative-free'
export { linprog } from './programming'
