/**
 * Andrews curves (Andrews, 1972, "Plots of high-dimensional data", Biometrics 28(1), 125–136): each row
 * x = (x₁, …, x_d) becomes the function
 *
 *   f_x(t) = x₁/√2 + x₂ sin t + x₃ cos t + x₄ sin 2t + x₅ cos 2t + …,   t ∈ [−π, π].
 *
 * The map is linear, and the basis {1/√2, sin kt, cos kt} is orthogonal on [−π, π] with every function of squared norm
 * π, so ∫ (f_x − f_y)² dt = π ‖x − y‖²: rows that are close stay close as curves, and at each t the value is a 1-D
 * projection of x onto (1/√2, sin t, cos t, …).
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrix, values } from '../util'

/** The Andrews basis at t for d features: [1/√2, sin t, cos t, sin 2t, cos 2t, …] (length d). */
function basis(t: number, d: number): Float64Array {
  const b = new Float64Array(d)
  if (d > 0) b[0] = Math.SQRT1_2
  for (let c = 1; c < d; c++) {
    const k = Math.ceil(c / 2)
    b[c] = c % 2 === 1 ? Math.sin(k * t) : Math.cos(k * t)
  }
  return b
}

/**
 * Andrews curves of the rows of x [n, d], evaluated at the points t (a rank-1 tensor or array, default 101 points
 * evenly spaced on [−π, π]). Returns `{ t, curves }` with `curves` [n, m]: curves[i, j] = f_{x_i}(t_j). Scale the
 * features first (e.g. `aifn-methods/learning/preprocess` `standardScaler`) if they are in different units, since the
 * first features carry the lowest frequencies and dominate the shape.
 */
export function andrewsCurves(x: Tensor, t?: Tensor | ArrayLike<number>): { t: Float64Array; curves: Tensor } {
  const { n, d, v } = matrix(x, 'andrewsCurves')
  const ts =
    t === undefined
      ? Float64Array.from({ length: 101 }, (_, j) => -Math.PI + (2 * Math.PI * j) / 100)
      : 'shape' in t
        ? values(t)
        : Float64Array.from(t)
  const m = ts.length
  const out = new Float64Array(n * m)
  for (let j = 0; j < m; j++) {
    const b = basis(ts[j], d)
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let c = 0; c < d; c++) s += v[i * d + c] * b[c]
      out[i * m + j] = s
    }
  }
  return { t: ts, curves: fromData(out, [n, m]) }
}
