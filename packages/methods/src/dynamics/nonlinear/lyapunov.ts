/**
 * Lyapunov's direct method checked on a grid, part of `aifn-methods/dynamics/nonlinear` (Khalil, 2002, "Nonlinear
 * Systems", 3rd ed., §4.1).
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
  x: Vector
  y: Vector
  /** V on the grid (ny × nx). */
  values: Matrix
  /** V̇ = ∇V·f on the grid (ny × nx). */
  derivative: Matrix
  /** V(x) > V(x*) at every grid point other than x* (V is positive definite about x* on the grid). */
  positiveDefinite: boolean
  /** V̇ ≤ tolerance at every grid point (Lyapunov stability on the region). */
  nonIncreasing: boolean
  /** V̇ < 0 at every grid point farther than one cell from x* (asymptotic stability on the region). */
  decreasing: boolean
  /** The largest V̇ on the grid away from x*, and where it occurs. */
  worst: { value: number; at: Vector }
}

/**
 * Checks Lyapunov's conditions for a candidate V about an equilibrium x* of a planar flow on a grid: V positive
 * definite (V(x) > V(x*) for x ≠ x*) and V̇ = ∇V·f ≤ 0 (stable) or < 0 away from x* (asymptotically stable). A grid
 * check is evidence on the sampled region, not a proof.
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
