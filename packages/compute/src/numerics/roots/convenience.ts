/** `findRoot` and `solveSystem`: `run` wrappers over the scalar and system algorithms. */

import type { Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import {
  bisection,
  brent,
  type BracketOptions,
  type RootState,
  type RootTolerance,
  type ScalarFunction,
} from './scalar'
import { broyden, newtonSystem, type SystemFunction, type SystemTolerance, type SystemWithJacobian } from './systems'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

/** The result of `findRoot`. */
export type RootResult = {
  /** Root estimate $x$. */
  x: number
  /** Function value $f(x)$ at the root estimate. */
  fx: number
  /** Number of algorithm steps taken. */
  steps: number
  /** Total number of objective function calls performed. */
  evaluations: number
  /** True once convergence criteria are satisfied. */
  converged: boolean
  /** Failure diagnosis string, or `null` on successful convergence. */
  failure: string | null
}

/**
 * Find a root of scalar function $f(x) = 0$ within a sign-changing bracket $[lo, hi]$.
 *
 * Uses Brent's method by default (or bisection) to locate a root where $f(lo)$ and $f(hi)$ have
 * opposite signs. Iterates up to `maxSteps` (default 200). Returns diagnostics in `RootResult` without throwing.
 *
 * @param f Continuous scalar objective function $f(x)$.
 * @param bracket Initial bracket interval boundaries $[lo, hi]$.
 * @param options Solver method choice, step budget, and tolerances.
 * @returns Root result containing the estimated location, residual, and convergence status.
 *
 * @example Find root of cosine minus x
 * const res = findRoot(x => Math.cos(x) - x, [0, 1])
 * print('converged =', res.converged)
 * print('root =', res.x)
 */
export function findRoot(
  f: ScalarFunction,
  bracket: readonly [number, number],
  options: RootTolerance & { method?: 'brent' | 'bisection'; maxSteps?: number } = {},
): RootResult {
  const alg: Algorithm<BracketOptions, RootState> =
    options.method === 'bisection' ? bisection(f, options) : brent(f, options)
  const s = run(alg, { lo: bracket[0], hi: bracket[1] }, options.maxSteps ?? 200)
  return { x: s.x, fx: s.fx, steps: s.t, evaluations: s.evaluations, converged: s.converged, failure: s.failure }
}

/** The result of `solveSystem`. */
export type SystemResult = {
  /** Solution coordinate vector $\xvec$. */
  x: Vector
  /** Residual vector $F(\xvec)$ at the solution estimate. */
  residual: Vector
  /** Euclidean norm of the residual $\|F(\xvec)\|_2$. */
  residualNorm: number
  /** Number of algorithm steps taken. */
  steps: number
  /** Total number of system evaluations performed. */
  evaluations: number
  /** True once convergence criteria are satisfied. */
  converged: boolean
  /** Failure diagnosis string, or `null` on successful convergence. */
  failure: string | null
}

/**
 * Solve a multivariate nonlinear system $F(\xvec) = \mathbf{0}$ from initial guess $\xvec_0$.
 *
 * Solves using damped Newton (default), undamped Newton, or Broyden quasi-Newton methods.
 * Newton methods require $F$ to return both `{ value, jacobian }`, while Broyden accepts functions
 * returning residual vectors only.
 *
 * @param F Nonlinear system returning residual vector (and optional Jacobian matrix).
 * @param x0 Initial estimate vector $\xvec_0$.
 * @param options Method selection, maximum step count, and convergence tolerances.
 * @returns System result containing the solution vector, final residual norm, and status.
 *
 * @example Solve 2D system of equations
 * const F = x => ({
 *   value: [x.data[0] + x.data[1] - 3, x.data[0] ** 2 + x.data[1] ** 2 - 5],
 *   jacobian: [[1, 1], [2 * x.data[0], 2 * x.data[1]]],
 * })
 * const res = solveSystem(F, [2, 0])
 * print('converged =', res.converged)
 * print('solution =', res.x)
 */
export function solveSystem(
  F: SystemWithJacobian | SystemFunction,
  x0: VectorLike,
  options: SystemTolerance & {
    method?: 'newton' | 'damped-newton' | 'broyden'
    maxSteps?: number
    jacobian0?: MatrixLike
  } = {},
): SystemResult {
  const { method = 'damped-newton', maxSteps = 100 } = options
  const s =
    method === 'broyden'
      ? run(broyden(F as SystemFunction, options), { x0 }, maxSteps)
      : run(newtonSystem(F as SystemWithJacobian, { ...options, damped: method === 'damped-newton' }), { x0 }, maxSteps)
  return {
    x: s.x,
    residual: s.residual,
    residualNorm: s.residualNorm,
    steps: s.t,
    evaluations: s.evaluations,
    converged: s.converged,
    failure: s.failure,
  }
}
