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
  x: number
  fx: number
  steps: number
  evaluations: number
  converged: boolean
  failure: string | null
}

/**
 * A root of f in the bracket [lo, hi] (f(lo) and f(hi) of opposite signs) by Brent's method (default) or bisection,
 * for at most `maxSteps` steps (default 200). Failure (no sign change, non-finite f) is reported, not thrown.
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
  x: Vector
  residual: Vector
  residualNorm: number
  steps: number
  evaluations: number
  converged: boolean
  failure: string | null
}

/**
 * Solves F(x) = 0 from `x0` by damped Newton (default; `F` returns `{ value, jacobian }`), Newton, or Broyden (`F`
 * returns the values only), for at most `maxSteps` steps (default 100).
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
