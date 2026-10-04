/**
 * `atConvergence`: differentiating the converged solution of an `Algorithm` by the implicit function theorem (design K
 * §4.3, §7.3). The algorithm runs on raw values (`aifn-compute/foundation/trace`'s `run`); only the equation it solves is
 * differentiated, at the cost of one linear solve and independent of the number of steps. Its counterpart that
 * differentiates through every step, `unrolled`, is in `aifn-compute/foundation/trace`.
 */

import type { Algorithm, Size, Status } from 'aifn-compute/foundation/contracts'
import { NumericalError } from 'aifn-compute/foundation/errors'
import { run, type RunOptions } from 'aifn-compute/foundation/trace'
import { implicitRoot, type ImplicitOptions } from './implicit'

/** Options of `atConvergence`. */
export type AtConvergenceOptions<Start, S, X> = ImplicitOptions &
  RunOptions & {
    /** The start passed to `init`. */
    start: Start
    /** The solution read from the converged state (the x of residual(p, x) = 0). */
    select: (state: S) => X
    /** Steps before giving up with `NumericalError('not-converged')` (default 10 000). */
    maxSteps?: Size
  }

/**
 * The converged solution of the algorithm `make(p)` as a function of $p$, differentiable in $p$ by the implicit
 * function theorem (`implicitRoot`) given the equation it solves, $\text{residual}(p, x) = 0$ (a gradient for a
 * minimiser, $x - F(p, x)$ for a fixed-point iteration). The algorithm runs on raw values and may be written with
 * anything (EM, IRLS, Sinkhorn, an `optim` solver); it must converge (`converged` or `done`), or
 * `NumericalError('not-converged')` is raised.
 *
 * @param make A factory returning the step-through `Algorithm` for parameter $p$.
 * @param residual The stationarity condition or root residual $\text{residual}(p, x) = 0$ satisfied at the solution.
 * @param options Convergence options including initial start, selection function, and step budget.
 * @returns A differentiable function computing the converged solution for parameter $p$.
 *
 * @example Differentiate through a converged square root iteration
 * const stepSqrt = (p) => ({
 *   name: 'stepSqrt',
 *   init: (start) => ({ x: start, converged: false }),
 *   step: (s) => {
 *     const next = 0.5 * (s.x + p / s.x)
 *     return { x: next, converged: Math.abs(next - s.x) < 1e-10 }
 *   },
 * })
 * const sqrtFn = atConvergence(stepSqrt, (p, x) => sub(mul(x, x), p), {
 *   start: 1,
 *   select: (s) => s.x,
 * })
 * print('d(sqrt(4))/dp =', grad(sqrtFn)(4))
 */
export function atConvergence<P, Start, S extends Status, X>(
  make: (p: P) => Algorithm<Start, S>,
  residual: (p: P, x: X) => X,
  options: AtConvergenceOptions<Start, S, X>,
): (p: P) => X {
  const { start, select, maxSteps = 10_000, stream } = options
  const solver = (p: P): X => {
    const alg = make(p)
    const state = run(alg, start, maxSteps, { stream })
    if (!(state.converged === true || alg.done?.(state) === true))
      throw new NumericalError(
        alg.name,
        `atConvergence: ${alg.name} did not converge in ${maxSteps} steps; its solution has no implicit derivative`,
        'not-converged',
      )
    return select(state)
  }
  return implicitRoot(solver, residual, options)
}
