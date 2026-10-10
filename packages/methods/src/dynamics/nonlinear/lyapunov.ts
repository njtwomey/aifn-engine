/**
 * Lyapunov's direct method checked on a grid, part of `aifn-methods/dynamics/nonlinear` (Khalil, 2002, "Nonlinear
 * Systems", 3rd ed., §4.1).
 *
 * A candidate $V$ certifies an equilibrium $\xvec^*$ of $\dot{\xvec} = \fvec(\xvec)$ as stable when it is positive
 * definite about $\xvec^*$ and $\dot{V} = \nabla V \cdot \fvec \le 0$, and as asymptotically stable when
 * $\dot{V} < 0$ away from $\xvec^*$. Here both are checked at the points of a planar grid, with $\nabla V$ by
 * autodiff: evidence on the sampled region, not a proof.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { dense, fromData, item, isTensor, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import {
  lyapunovDerivative,
  sampleScalar,
  type Grid2,
  type ScalarField,
  type VectorField,
} from 'aifn-compute/dynamics/fields'

/** The result of `lyapunovCheck`. */
export type LyapunovCheck = {
  /** The grid's $x$ coordinates ($n_x$). */
  x: Vector
  /** The grid's $y$ coordinates ($n_y$). */
  y: Vector
  /** $V$ on the grid ($n_y \times n_x$; row $i$ at $y_i$). */
  values: Matrix
  /** $\dot{V} = \nabla V \cdot \fvec$ on the grid ($n_y \times n_x$). */
  derivative: Matrix
  /**
   * $V(\xvec) > V(\xvec^*)$ at every grid point other than $\xvec^*$ ($V$ is positive definite about $\xvec^*$ on
   * the grid).
   */
  positiveDefinite: boolean
  /** $\dot{V} \le$ `tolerance` at every grid point (Lyapunov stability on the region). */
  nonIncreasing: boolean
  /**
   * $\dot{V} < 0$ at every grid point farther than one cell diagonal from $\xvec^*$ (asymptotic stability on the
   * region).
   */
  decreasing: boolean
  /**
   * The largest $\dot{V}$ on the grid farther than one cell diagonal from $\xvec^*$, and where it occurs ($-\infty$
   * and NaN when there is no such point).
   */
  worst: { value: number; at: Vector }
}

/**
 * Checks Lyapunov's conditions for a candidate $V$ about an equilibrium $\xvec^*$ of a planar flow on a grid: $V$
 * positive definite ($V(\xvec) > V(\xvec^*)$ for $\xvec \ne \xvec^*$) and $\dot{V} = \nabla V \cdot \fvec \le 0$
 * (stable) or $< 0$ away from $\xvec^*$ (asymptotically stable). A grid check is evidence on the sampled region, not
 * a proof.
 *
 * @param V The candidate $V$, written with tensor primitives so that autodiff can take $\nabla V$; called with a point
 *   as a rank-1 tensor of 2 values.
 * @param f The vector field $\fvec$ of the planar flow.
 * @param equilibrium The equilibrium $\xvec^*$ (2 values).
 * @param grid The grid: the $x$ and $y$ ranges and the number of points along each.
 * @param options The tolerance.
 * @param options.tolerance The largest $\dot{V}$ still counted as non-increasing (default 1e-12), for rounding at
 *   points where $\dot{V}$ is exactly 0. It does not apply to `decreasing`.
 * @returns $V$ and $\dot{V}$ on the grid, the three verdicts and the worst point.
 *
 * @example Damping: the energy is a Lyapunov function, but not a strict one
 * // x' = y, y' = -x - y with V = x^2 + y^2: dV/dt = -2y^2, zero along the x axis.
 * const V = (x) => sum(mul(x, x))
 * const f = (x) => {
 *   const [a, b] = toFlat(x)
 *   return [b, -a - b]
 * }
 * const r = lyapunovCheck(V, f, [0, 0], { x: [-1, 1], y: [-1, 1], nx: 11, ny: 11 })
 * print('positive definite:', r.positiveDefinite, ' non-increasing:', r.nonIncreasing, ' decreasing:', r.decreasing)
 *
 * @example A spiral sink: strictly decreasing
 * // x' = -x + y, y' = -x - y: dV/dt = -2(x^2 + y^2).
 * const V = (x) => sum(mul(x, x))
 * const f = (x) => {
 *   const [a, b] = toFlat(x)
 *   return [-a + b, -a - b]
 * }
 * const r = lyapunovCheck(V, f, [0, 0], { x: [-1, 1], y: [-1, 1], nx: 11, ny: 11 })
 * print('decreasing:', r.decreasing, ' worst dV/dt:', r.worst.value, 'at', r.worst.at)
 */
export function lyapunovCheck(
  V: ScalarField,
  f: VectorField,
  equilibrium: VectorLike,
  grid: Grid2,
  { tolerance: tol = 1e-12 }: { tolerance?: number } = {},
): LyapunovCheck {
  const eq = dense.toF64(equilibrium, 'lyapunovCheck')
  const values = sampleScalar(V, grid)
  const xs = toFlat(values.x)
  const ys = toFlat(values.y)
  const atEq = V(fromData(Float64Array.of(eq[0], eq[1]), [2]))
  const v0 = typeof atEq === 'number' ? atEq : isTensor(atEq) ? item(atEq) : NaN
  const cell = Math.hypot(
    (grid.x[1] - grid.x[0]) / Math.max(1, grid.nx - 1),
    (grid.y[1] - grid.y[0]) / Math.max(1, grid.ny - 1),
  )
  const vv = toFlat(values.values)
  const dv = new Float64Array(vv.length)
  let positiveDefinite = true
  let nonIncreasing = true
  let decreasing = true
  let worst = -Infinity
  let worstAt = [NaN, NaN]
  for (let i = 0; i < ys.length; i++)
    for (let j = 0; j < xs.length; j++) {
      const k = i * xs.length + j
      const d = lyapunovDerivative(V, f, [xs[j], ys[i]])
      dv[k] = d
      const r = Math.hypot(xs[j] - eq[0], ys[i] - eq[1])
      if (r > 1e-12 && !(vv[k] > v0)) positiveDefinite = false
      if (d > tol) nonIncreasing = false
      if (r > cell) {
        if (!(d < 0)) decreasing = false
        if (d > worst) [worst, worstAt] = [d, [xs[j], ys[i]]]
      }
    }
  return {
    x: values.x,
    y: values.y,
    values: values.values,
    derivative: fromData(dv, [ys.length, xs.length]),
    positiveDefinite,
    nonIncreasing,
    decreasing,
    worst: { value: worst, at: fromData(Float64Array.from(worstAt), [2]) },
  }
}
