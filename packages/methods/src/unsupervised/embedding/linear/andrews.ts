/**
 * Andrews curves (Andrews, 1972, "Plots of high-dimensional data", Biometrics 28(1), 125-136): each row
 * $\xvec = (x_1, \dots, x_d)$ becomes the function
 *
 * $f_{\xvec}(t) = x_1/\sqrt{2} + x_2 \sin t + x_3 \cos t + x_4 \sin 2t + x_5 \cos 2t + \dots$, $t \in [-\pi, \pi]$.
 *
 * The map is linear, and the basis $\{1/\sqrt{2}, \sin kt, \cos kt\}$ is orthogonal on $[-\pi, \pi]$ with every
 * function of squared norm $\pi$, so $\int (f_{\xvec} - f_{\yvec})^2 \, dt = \pi \lVert \xvec - \yvec \rVert^2$: rows
 * that are close stay close as curves, and at each $t$ the value is a one-dimensional projection of $\xvec$ onto
 * $(1/\sqrt{2}, \sin t, \cos t, \dots)$.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrix, values } from '../util'

/**
 * The Andrews basis at $t$ for $d$ features: $(1/\sqrt{2}, \sin t, \cos t, \sin 2t, \cos 2t, \dots)$, cut to $d$
 * entries.
 *
 * @param t The point at which the basis functions are evaluated.
 * @param d The number of features, and so of basis functions.
 * @returns The $d$ basis values; feature $c$ (from 0) is paired with entry $c$.
 */
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
 * Andrews curves of the rows of a matrix, evaluated at the points $t_j$: entry $(i, j)$ of `curves` is
 * $f_{\xvec_i}(t_j)$. Scale the features first (e.g. `standardScaler` of `aifn-methods/learning/preprocessing`) if they
 * are in different units, since the first features carry the lowest frequencies and dominate the shape. Throws
 * `ShapeError` when `x` is not a matrix.
 *
 * @param x The data ($n \times d$), one row per curve.
 * @param t The points at which to evaluate the curves: a tensor (read in row-major order) or an array of $m$ numbers.
 *   Left out, 101 points evenly spaced on $[-\pi, \pi]$, ends included.
 * @returns `t`, the $m$ evaluation points, and `curves`, the $n \times m$ matrix of curve values, one row per row of
 *   `x`.
 *
 * @example Each feature contributes one basis function
 * // The three unit rows trace 1/sqrt(2), sin t and cos t.
 * const x = tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
 * const { t, curves } = andrewsCurves(x, [0, Math.PI / 2, Math.PI])
 * print('t =', t)
 * print('curves =', curves)
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
