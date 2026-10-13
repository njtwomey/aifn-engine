/**
 * `aifn-methods/data/objectives`: test surfaces for optimisers, with their gradients, Hessians and known minima.
 *
 * - Valleys and ill-conditioning: `rosenbrock` (a curved, flat-bottomed valley, in any dimension $n \ge 2$) and
 *   `quadraticBowl` (a convex quadratic of chosen condition number, rotated in two dimensions).
 * - Several minima: `himmelblau` (four global minima of value 0), `beale` (one minimum in a narrow valley, with flat
 *   plateaus) and `rastrigin` (a grid of local minima around one global minimum).
 * - Global and Bayesian optimisation: `forrester` (one dimension, a deceptive local minimum, with a tilted
 *   low-fidelity version for multi-fidelity methods), `branin` (two dimensions, three global minima) and
 *   `svmTuningSurface` (an RBF SVM's cross-validated error over $\log_{10} C$ and $\log_{10}\gamma$, tabulated once
 *   by `scripts/svm-surface.ts` and interpolated, with optional evaluation noise drawn from a stream).
 * - Several tasks at once: `tuningTables`, the same SVM grid tabulated on seven small classification tasks, as a
 *   task by grid tensor and a configuration by task table, for warm-starting and portfolios (a function, not an
 *   objective).
 *
 * Each returns a `TestFunction`: the value, an `ObjectiveFn` for the gradient-based methods of `aifn-compute/optim`,
 * the Hessian, the global minimisers and minimum value, a conventional start and a plotting box; `svmTuningSurface`
 * returns a `TuningSurface`, a `TestFunction` that also carries its table. All are registered as objectives (kind
 * `objective`) and collected in `objectiveRegistry`.
 */

export {
  beale,
  branin,
  forrester,
  himmelblau,
  quadraticBowl,
  rastrigin,
  rosenbrock,
  type TestFunction,
} from './surfaces'
export { svmTuningSurface, type TuningSurface } from './svm'
export { tuningTables, type TuningTables } from './tuning-tables'
