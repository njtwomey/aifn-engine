/**
 * Learning-curve extrapolation by a power law, the simplest of the parametric families of Domhan, Springenberg and
 * Hutter (2015): fit
 *
 * $$ y(t) = a + b\,t^{-c}, \qquad c > 0, $$
 *
 * to the losses seen so far and read off the asymptote $a$ or the value at a larger budget. The fit is least squares
 * by variable projection: for a fixed exponent $c$ the best $a$ and $b$ solve a two-column linear least-squares
 * problem, so only the profile residual in $\log c$ is searched, on a grid and then by Brent's method
 * (`minimizeScalar` from `aifn-compute/numerics/roots`).
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { minimizeScalar } from 'aifn-compute/numerics/roots'

/** A fitted power law $y(t) = a + b\,t^{-c}$. */
export interface PowerLawFit {
  /** The asymptote $a$: the loss as the budget grows without bound. */
  a: number
  /** The scale $b$. */
  b: number
  /** The exponent $c > 0$. */
  c: number
  /** The residual sum of squares at the fit. */
  rss: number
  /** The fitted curve at a budget $t > 0$. */
  at: (t: number) => number
}

/**
 * The best $a$, $b$ and residual for a fixed exponent: weighted least squares on the columns $1$ and $t^{-c}$.
 *
 * @param t The budgets.
 * @param y The losses.
 * @param w The weights.
 * @param c The exponent.
 * @returns $a$, $b$ and the weighted residual sum of squares.
 */
function profile(t: Float64Array, y: Float64Array, w: Float64Array, c: number): { a: number; b: number; rss: number } {
  let [sw, sx, sy, sxx, sxy] = [0, 0, 0, 0, 0]
  for (let i = 0; i < t.length; i++) {
    const x = t[i] ** -c
    sw += w[i]
    sx += w[i] * x
    sy += w[i] * y[i]
    sxx += w[i] * x * x
    sxy += w[i] * x * y[i]
  }
  const det = sw * sxx - sx * sx
  const b = det > 1e-300 ? (sw * sxy - sx * sy) / det : 0
  const a = (sy - b * sx) / sw
  let rss = 0
  for (let i = 0; i < t.length; i++) rss += w[i] * (y[i] - a - b * t[i] ** -c) ** 2
  return { a, b, rss }
}

/**
 * Fit $y(t) = a + b\,t^{-c}$ by (weighted) least squares, with $c$ searched in $[10^{-3}, 10]$ (see the file
 * comment). With fewer than three points the fit is not determined: throws `DomainError`, as for a budget that is
 * not positive or a value that is not finite; `ShapeError` when the lengths differ.
 *
 * @param t The budgets $t_i > 0$ at which the losses were seen.
 * @param y The losses $y_i$, as many as budgets.
 * @param options The weights.
 * @param options.weights A non-negative weight per point (default equal), e.g. to favour the later budgets.
 * @returns The fit, as `PowerLawFit`, with the curve `at` for extrapolation.
 *
 * @example Extrapolate a curve seen up to budget 9 to budget 81
 * const t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
 * const y = t.map((s) => 0.2 + 0.5 * s ** -0.7)
 * const fit = fitPowerLaw(t, y)
 * print('a =', fit.a.toFixed(4), ' b =', fit.b.toFixed(4), ' c =', fit.c.toFixed(4))
 * print('at 81:', fit.at(81).toFixed(4), ' true:', (0.2 + 0.5 * 81 ** -0.7).toFixed(4))
 */
export function fitPowerLaw(
  t: Tensor | ArrayLike<number>,
  y: Tensor | ArrayLike<number>,
  options: { weights?: Tensor | ArrayLike<number> } = {},
): PowerLawFit {
  const where = 'fitPowerLaw'
  const read = (v: Tensor | ArrayLike<number>) => Float64Array.from((isTensor(v) ? toFlat(v) : v) as ArrayLike<number>)
  const T = read(t)
  const Y = read(y)
  const W = options.weights === undefined ? new Float64Array(T.length).fill(1) : read(options.weights)
  if (Y.length !== T.length || W.length !== T.length)
    throw new ShapeError(where, `${where}: ${T.length} budgets, ${Y.length} losses and ${W.length} weights`)
  if (T.length < 3) throw new DomainError(where, `${where}: needs at least three points`)
  if (!T.every((v) => v > 0)) throw new DomainError(where, `${where}: the budgets must be positive`)
  if (!Y.every(Number.isFinite) || !W.every((v) => v >= 0))
    throw new DomainError(where, `${where}: the losses must be finite and the weights non-negative`)
  const [lo, hi] = [Math.log(1e-3), Math.log(10)]
  const cost = (u: number) => profile(T, Y, W, Math.exp(u)).rss
  // A grid over log c finds the basin; Brent's method polishes it.
  const grid = 60
  let best = lo
  for (let k = 0; k <= grid; k++) {
    const u = lo + ((hi - lo) * k) / grid
    if (cost(u) < cost(best)) best = u
  }
  const step = (hi - lo) / grid
  const r = minimizeScalar(cost, {
    bounds: [Math.max(lo, best - step), Math.min(hi, best + step)],
    tolerance: 1e-12,
  })
  const u = cost(r.x) < cost(best) ? r.x : best
  const c = Math.exp(u)
  const { a, b, rss } = profile(T, Y, W, c)
  return { a, b, c, rss, at: (s: number) => a + b * s ** -c }
}
