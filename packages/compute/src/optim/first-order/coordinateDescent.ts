/**
 * Coordinate descent: each step changes one coordinate of x, chosen cyclically, uniformly at random, or greedily
 * (the largest gradient component, the Gauss–Southwell rule). Wright (2015), "Coordinate descent algorithms",
 * Mathematical Programming 151, §2–3.
 */

import { integers } from 'aifn-compute/foundation/random'
import type { Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import type { Hessian, IterateState, ObjectiveFn, StoppingOptions } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'

const { data, norm, toF64, toMatrixF64, vec } = dense

/** How the coordinate is chosen. */
export type CoordinateRule = 'cyclic' | 'random' | 'greedy'

/** The state of `coordinateDescent`. */
export type CoordinateDescentState = IterateState & {
  grad: Vector
  gradNorm: number
  /** The coordinate changed on the last step (−1 at t = 0). */
  coordinate: number
  /** The change made to that coordinate. */
  change: number
  /** Completed sweeps: t / n, rounded down. */
  sweep: number
}

/** Options for `coordinateDescent`. */
export type CoordinateDescentOptions = StoppingOptions & {
  /** Default `'cyclic'`. */
  rule?: CoordinateRule
  /**
   * How far to move coordinate i, in order of precedence: `minimizeCoordinate(x, i)` returns the new value of x_i (an
   * exact coordinate minimiser, e.g. a soft threshold for the lasso); `hessian` gives a Newton step −gᵢ/Hᵢᵢ (exact
   * on a quadratic); otherwise a gradient step −lr·gᵢ.
   */
  minimizeCoordinate?: (x: Vector, i: number) => number
  hessian?: Hessian
  /** Step size of the gradient step. Default 0.1. */
  stepSize?: number
}

/**
 * Coordinate descent (Wright, 2015): picks a coordinate i by `rule` and updates x_i alone, by an exact coordinate
 * minimisation, a coordinate Newton step or a coordinate gradient step (see `CoordinateDescentOptions`). On a strictly
 * convex quadratic with exact steps each step is one Gauss–Seidel update. `init` takes `{ x0 }`; the random rule draws
 * step t's coordinate from the runner's step stream.
 */
export function coordinateDescent(
  f: ObjectiveFn,
  options: CoordinateDescentOptions = {},
): Algorithm<StartOptions, CoordinateDescentState> {
  const { rule = 'cyclic', stepSize: lr = 0.1, tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const name = `coordinate-descent-${rule}`
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        coordinate: -1,
        change: 0,
        sweep: 0,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (s, ctx) => {
      const x = data(s.x)
      const g = data(s.grad)
      const n = x.length
      let i: number
      if (rule === 'cyclic') i = s.t % n
      else if (rule === 'random') i = integers(ctx.stream, n)
      else {
        i = 0
        for (let j = 1; j < n; j++) if (Math.abs(g[j]) > Math.abs(g[i])) i = j
      }
      const next = Float64Array.from(x)
      if (options.minimizeCoordinate) next[i] = options.minimizeCoordinate(s.x, i)
      else if (options.hessian) {
        const H = toMatrixF64(options.hessian(s.x), name, n, n).data
        const hii = H[i * n + i]
        // A non-positive curvature gives no Newton step along this axis; fall back to a gradient step.
        next[i] = hii > 0 ? x[i] - g[i] / hii : x[i] - lr * g[i]
      } else next[i] = x[i] - lr * g[i]
      const { value, grad } = evaluate(f, next, name)
      const gradNorm = norm(grad)
      return {
        ...s,
        t: s.t + 1,
        x: vec(next),
        value,
        grad: vec(grad),
        gradNorm,
        coordinate: i,
        change: next[i] - x[i],
        sweep: Math.floor((s.t + 1) / n),
        evaluations: s.evaluations + 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, next, divergeAbove),
      }
    },
  }
}
