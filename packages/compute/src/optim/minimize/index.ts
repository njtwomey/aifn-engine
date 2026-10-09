/**
 * `aifn-compute/optim/minimize`: one entry point that runs any method of `aifn-compute/optim` by name, as
 * scipy.optimize.minimize.
 *
 * - Run to the end: `minimize` takes the objective, a start point and a `Method` name (default `'lbfgs'`) with that
 *   method's options (`MethodOptions`), and returns a `MinimizeResult`.
 * - Step through: `methodAlgorithm` returns the named method's `Algorithm`, for `run`, `trace` or a training loop.
 *
 * The objective may be an `Objective` (differentiated by autodiff), a function returning `{ value, grad }`, or a value
 * function for the derivative-free methods. Non-convergence is reported in the result, not thrown.
 * `minimizeFunctions` registers `minimize`.
 */

export { methodAlgorithm, minimize, type Method, type MethodOptions, type MinimizeResult } from './minimize'
export { minimizeFunctions } from './registry'
