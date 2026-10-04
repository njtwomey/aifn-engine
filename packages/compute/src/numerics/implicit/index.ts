/**
 * `aifn-compute/numerics/implicit`: implicit differentiation, the derivative of a solution with respect to the parameters of
 * the equation it solves without differentiating the solver (design K §4.3).
 *
 * - `implicitFixedPoint(solver, F)`: x⋆ = F(p, x⋆).
 * - `implicitRoot(solver, residual)`: residual(p, x⋆) = 0 (a minimiser's gradient, a stationarity condition).
 * - `atConvergence(make, residual, { start, select })`: the converged state of an `Algorithm`.
 *
 * The adjoint system is solved with `aifn-compute/numerics/linalg`'s `solve` for small x (the one dense solve in aifn) and
 * iteratively with vjp/jvp calls only for large x. Built on `defineCustomVjp` from `aifn-compute/foundation/autodiff`.
 */

export { implicitFixedPoint, implicitRoot, type ImplicitOptions } from './implicit'
export { atConvergence, type AtConvergenceOptions } from './convergence'
export { implicitFunctions } from './registry'
