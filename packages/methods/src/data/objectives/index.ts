/**
 * `aifn-methods/data/objectives`: test surfaces for optimisers, with their gradients, Hessians and known minima.
 *
 * - Valleys and ill-conditioning: `rosenbrock` (a curved, flat-bottomed valley, in any dimension $n \ge 2$) and
 *   `quadraticBowl` (a convex quadratic of chosen condition number, rotated in two dimensions).
 * - Several minima: `himmelblau` (four global minima of value 0), `beale` (one minimum in a narrow valley, with flat
 *   plateaus) and `rastrigin` (a grid of local minima around one global minimum).
 *
 * Each returns a `TestFunction`: the value, an `ObjectiveFn` for the gradient-based methods of `aifn-compute/optim`,
 * the Hessian, the global minimisers and minimum value, a conventional start and a plotting box. All are registered
 * as objectives (kind `objective`) and collected in `objectiveRegistry`.
 */

export { beale, himmelblau, quadraticBowl, rastrigin, rosenbrock, type TestFunction } from './surfaces'
