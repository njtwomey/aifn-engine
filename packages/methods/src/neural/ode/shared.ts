/**
 * Internal helpers shared by the neural ODE runs: work counters fed by `onSolve`, standardised data, grids and flat
 * arrays for a worker to post.
 */

import type { OdeSolveInfo } from 'aifn-compute/dynamics/ode'
import { fromData, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

/** Function evaluations of the solves since the last `take`, split into forward and backward. */
export type WorkCounter = {
  /** The solver's `onSolve` callback: adds each solve's evaluations to the forward or the backward count. */
  onSolve: (info: OdeSolveInfo) => void
  /** The counts since the last call, reset. Backprop's backward pass replays the tape: it counts as the forward. */
  take: (gradient: 'backprop' | 'adjoint' | 'none') => { forward: number; backward: number }
}

/**
 * A fresh counter of function evaluations, both counts at 0.
 *
 * @returns The counter: pass its `onSolve` to the solver, and `take` the counts after each step.
 */
export function workCounter(): WorkCounter {
  let forward = 0
  let backward = 0
  return {
    onSolve: (info) => {
      if (info.phase === 'forward') forward += info.evaluations
      else backward += info.evaluations
    },
    take: (gradient) => {
      const out = { forward, backward: gradient === 'backprop' ? forward : gradient === 'adjoint' ? backward : 0 }
      forward = 0
      backward = 0
      return out
    },
  }
}

/**
 * A value's entries as a `Float64Array`.
 *
 * @param v A number or a tensor (traced values are unwrapped).
 * @returns Its entries in row-major order, a copy.
 */
export const flatOf = (v: Value): Float64Array =>
  typeof v === 'number' ? Float64Array.of(v) : Float64Array.from(toFlat(unwrap(v) as Tensor))

/**
 * The scalar of a rank-0 value or number.
 *
 * @param v A number or a tensor; for a larger tensor, its first entry is read.
 * @returns The number.
 */
export const scalarOf = (v: Value): number => flatOf(v)[0]

/**
 * Rows centred per column and scaled by one factor, so the overall standard deviation (over all $n d$ entries) is
 * `scale`.
 *
 * @param x The data, $[n, d]$; not modified.
 * @param scale The standard deviation wanted.
 * @returns The standardised data, a new $[n, d]$ tensor (only centred when every entry is the same).
 */
export function standardise(x: Tensor, scale = 1): Tensor {
  const [n, d] = x.shape
  const a = Float64Array.from(toFlat(x))
  const mean = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += a[i * d + j] / n
  let ss = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      a[i * d + j] -= mean[j]
      ss += a[i * d + j] ** 2
    }
  const sd = Math.sqrt(ss / (n * d)) || 1
  for (let k = 0; k < a.length; k++) a[k] *= scale / sd
  return fromData(a, [n, d])
}

/**
 * Evenly spaced values from `a` to `b`, both included.
 *
 * @param a The first value.
 * @param b The last value.
 * @param m How many values; with 1, just `a`.
 * @returns The `m` values.
 */
export const spaced = (a: number, b: number, m: number): number[] =>
  Array.from({ length: m }, (_, i) => (m === 1 ? a : a + ((b - a) * i) / (m - 1)))

/**
 * A square grid on $[-b, b]^2$: its axis and its points in row-major order ($y$ outer, $x$ inner).
 *
 * @param box The half-width $b$ of the square.
 * @param g The points per side.
 * @returns The `axis` of $g$ values, and the `points` as a $[g^2, 2]$ tensor of $(x, y)$ rows.
 */
export function planeGrid(box: number, g: number): { axis: Float64Array; points: Tensor } {
  const axis = Float64Array.from(spaced(-box, box, g))
  const pts = new Float64Array(g * g * 2)
  for (let i = 0; i < g; i++)
    for (let j = 0; j < g; j++) {
      pts[2 * (i * g + j)] = axis[j]
      pts[2 * (i * g + j) + 1] = axis[i]
    }
  return { axis, points: fromData(pts, [g * g, 2]) }
}

/**
 * The half-width of a square holding every point, with a margin, rounded up to a multiple of $\frac{1}{2}$.
 *
 * @param x The coordinates of the points, flat (any dimension: the largest absolute entry is used).
 * @param margin The factor the largest absolute coordinate is multiplied by.
 * @returns The half-width, or 1 when every coordinate is 0.
 */
export function boxOf(x: Float64Array, margin = 1.25): number {
  let m = 0
  for (const v of x) m = Math.max(m, Math.abs(v))
  return Math.ceil(m * margin * 2) / 2 || 1
}

/**
 * The wall clock in milliseconds: `performance.now()` where it exists, else `Date.now()`.
 *
 * @returns The time, for differences only.
 */
export const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now())
