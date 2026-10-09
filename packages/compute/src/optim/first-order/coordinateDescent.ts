/**
 * Coordinate descent: each step changes one coordinate of $\xvec$, chosen cyclically, uniformly at random, or greedily
 * (the largest gradient component, the Gauss–Southwell rule). Wright (2015), "Coordinate descent algorithms",
 * Mathematical Programming 151, §2–3.
 *
 * The objective is evaluated (value and full gradient) once per step, and the run stops when
 * $\lVert \nabla f(\xvec) \rVert_2 \le$ `tolerance` or on divergence, which the state reports.
 */

import { integers } from 'aifn-compute/foundation/random'
import type { Vector } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { StartOptions } from '../options'
import type { Hessian, IterateState, ObjectiveFn, StoppingOptions } from 'aifn-compute/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn-compute/foundation/tensor'

const { data, norm, toF64, toMatrixF64, vec } = dense

/**
 * How the coordinate is chosen: in turn (`'cyclic'`), uniformly at random from the step's stream (`'random'`), or as
 * the one with the largest $\lvert \partial f / \partial x_i \rvert$ (`'greedy'`, ties to the lowest index).
 */
export type CoordinateRule = 'cyclic' | 'random' | 'greedy'

/** The state of `coordinateDescent`. */
export type CoordinateDescentState = IterateState & {
  /** $\nabla f(\xvec)$, the full gradient. */
  grad: Vector
  /** $\lVert \nabla f(\xvec) \rVert_2$, compared with `tolerance`. */
  gradNorm: number
  /** The coordinate changed on the last step ($-1$ at $t = 0$). */
  coordinate: number
  /** The change made to that coordinate. */
  change: number
  /** Completed sweeps: $\lfloor t / n \rfloor$. */
  sweep: number
}

/** Options for `coordinateDescent`. */
export type CoordinateDescentOptions = StoppingOptions & {
  /** How the coordinate is chosen. Default `'cyclic'`. */
  rule?: CoordinateRule
  /**
   * How far to move coordinate $i$, in order of precedence: `minimizeCoordinate(x, i)` returns the new value of $x_i$
   * (an exact coordinate minimiser, e.g. a soft threshold for the lasso); `hessian` gives a Newton step
   * $-g_i/H_{ii}$ (exact on a quadratic); otherwise a gradient step $-\eta g_i$, with $\eta$ = `stepSize`.
   */
  minimizeCoordinate?: (x: Vector, i: number) => number
  /**
   * The Hessian $\nabla^2 f(\xvec)$, of which only the diagonal entry $H_{ii}$ is read. Where $H_{ii} \le 0$ the step
   * falls back to a gradient step.
   */
  hessian?: Hessian
  /** Step size $\eta$ of the gradient step. Default 0.1. */
  stepSize?: number
}

/**
 * Coordinate descent (Wright, 2015): picks a coordinate $i$ by `rule` and updates $x_i$ alone, by an exact coordinate
 * minimisation, a coordinate Newton step or a coordinate gradient step (see `CoordinateDescentOptions`). On a strictly
 * convex quadratic with exact steps each step is one Gauss–Seidel update. The random rule draws step $t$'s coordinate
 * from the runner's step stream.
 *
 * @param f The objective, returning `{ value, grad }` at a point.
 * @param options The coordinate rule, how far to move (exact minimiser, Hessian or step size), and the stopping
 *   options.
 * @returns The algorithm; `init` takes `{ x0 }`.
 *
 * @example Exact coordinate steps on a coupled quadratic
 * // f(x) = x₁² + x₁x₂ + 10x₂², with Hessian [[2, 1], [1, 20]]: each step minimises f along one axis.
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + a * b + 10 * b * b, grad: [2 * a + b, a + 20 * b] }
 * }
 * const hessian = () => [[2, 1], [1, 20]]
 * for (const steps of [1, 2, 4, 10]) {
 *   const s = run(coordinateDescent(f, { hessian }), { x0: [1, 1] }, steps)
 *   print(`after ${steps} steps: x =`, s.x, ' last coordinate =', s.coordinate, ' sweeps =', s.sweep)
 * }
 *
 * @example The greedy rule starts with the steepest coordinate
 * // At (1, 1) the gradient is (3, 21), so the greedy rule moves x₂ first.
 * const f = (x) => {
 *   const [a, b] = x.data
 *   return { value: a * a + a * b + 10 * b * b, grad: [2 * a + b, a + 20 * b] }
 * }
 * const hessian = () => [[2, 1], [1, 20]]
 * for (const rule of ['cyclic', 'greedy']) {
 *   const first = run(coordinateDescent(f, { rule, hessian }), { x0: [1, 1] }, 1)
 *   const last = run(coordinateDescent(f, { rule, hessian }), { x0: [1, 1] }, 1000)
 *   print(`${rule}: first coordinate =`, first.coordinate, ' steps to converge =', last.t)
 * }
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
