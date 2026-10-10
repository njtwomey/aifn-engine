/**
 * The shared layer of `aifn-methods/theory`: one-dimensional regression problems with a known regression function on
 * $[-1, 1]$, noisy training sets drawn from them, and least-squares fits of a feature map: minimum-norm when the system
 * is underdetermined, and ridge-penalised through the augmented system
 * $[\Phimat; \sqrt{\lambda}\Imat]\wvec \approx [\yvec; \zeros]$.
 *
 * Designs are row-major Float64Arrays with their dimensions passed beside them, as the simulators build them.
 */

import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { lstsq } from 'aifn-compute/numerics/linalg'

/**
 * A regression function on $[-1, 1]$: `sine` $\sin(\pi x)$, `step` ($\pm 0.5$, jumping at 0), `quadratic`
 * $x^2 - 0.3$, or `wiggly` $\sin(4x) + 0.5\cos(9x)$.
 */
export type TargetName = 'sine' | 'step' | 'quadratic' | 'wiggly'

/**
 * The regression function $f$ of a named target (see `TargetName`).
 *
 * @param name The target.
 * @returns $f$, a function of one number.
 *
 * @example The four targets at $x = 0.5$
 * for (const name of ['sine', 'step', 'quadratic', 'wiggly']) print(name, targetFunction(name)(0.5))
 */
export function targetFunction(name: TargetName): (x: number) => number {
  switch (name) {
    case 'sine':
      return (x) => Math.sin(Math.PI * x)
    case 'step':
      return (x) => (x < 0 ? -0.5 : 0.5)
    case 'quadratic':
      return (x) => x * x - 0.3
    case 'wiggly':
      return (x) => Math.sin(4 * x) + 0.5 * Math.cos(9 * x)
  }
}

/**
 * $n$ training pairs: $x$ uniform on $[-1, 1]$ and $y = f(x) + \sigma \varepsilon$ with
 * $\varepsilon \sim \Gauss(0, 1)$.
 *
 * @param s The stream; the inputs and the noise come from its children `x` and `noise`, so `s` is not advanced.
 * @param target The regression function $f$.
 * @param n The number of pairs.
 * @param noise The noise standard deviation $\sigma$.
 * @returns The inputs `x` and targets `y`, $n$ each.
 *
 * @example Five noisy points of the sine
 * const { x, y } = drawTrainingSet(stream(0), 'sine', 5, 0.1)
 * print('x =', x)
 * print('y =', y)
 */
export function drawTrainingSet(
  s: Stream,
  target: TargetName,
  n: number,
  noise: number,
): { x: Float64Array; y: Float64Array } {
  const f = targetFunction(target)
  const u = units(child(s, 'x'), n)
  const e = standardNormals(child(s, 'noise'), n)
  const x = Float64Array.from(u, (v) => 2 * v - 1)
  return { x, y: Float64Array.from(x, (v, i) => f(v) + noise * e[i]) }
}

/**
 * An even grid of $m$ points on $[-1, 1]$, both ends included ($m \ge 2$; a single point is NaN).
 *
 * @param m The number of points.
 * @returns The grid, increasing.
 *
 * @example Five points
 * print(unitGrid(5))
 */
export const unitGrid = (m: number): Float64Array => Float64Array.from({ length: m }, (_, i) => -1 + (2 * i) / (m - 1))

/**
 * Least-squares weights of the design $\Phimat$ $[n, p]$ (row-major) for targets $\yvec$: the minimum-norm solution
 * (`lstsq`, by the SVD) when $\lambda = 0$, else the ridge solution
 * $\argmin_{\wvec} \lVert \Phimat\wvec - \yvec \rVert^2 + \lambda\lVert \wvec \rVert^2$, from the augmented
 * system.
 *
 * @param phi The design $\Phimat$, row-major $[n, p]$; not modified.
 * @param n The number of rows (training points).
 * @param p The number of columns (features).
 * @param y The targets, $n$ values.
 * @param ridge The penalty $\lambda$; 0 or less gives the minimum-norm least-squares solution.
 * @returns The weights $\wvec$, $p$ values.
 *
 * @example A line through three points, and the minimum-norm fit of two points by three features
 * const line = leastSquaresWeights(new Float64Array([1, -1, 1, 0, 1, 1]), 3, 2, new Float64Array([0, 1, 2]))
 * print('intercept and slope =', line)
 * const w = leastSquaresWeights(new Float64Array([1, 0, 1, 0, 1, 1]), 2, 3, new Float64Array([2, 3]))
 * print('minimum-norm weights =', w)
 * print('ridge 1 =', leastSquaresWeights(new Float64Array([1, 0, 1, 0, 1, 1]), 2, 3, new Float64Array([2, 3]), 1))
 */
export function leastSquaresWeights(phi: Float64Array, n: number, p: number, y: Float64Array, ridge = 0): Float64Array {
  if (ridge > 0) {
    const a = new Float64Array((n + p) * p)
    a.set(phi)
    const r = Math.sqrt(ridge)
    for (let j = 0; j < p; j++) a[(n + j) * p + j] = r
    const b = new Float64Array(n + p)
    b.set(y)
    return Float64Array.from(toFlat(lstsq(fromData(a, [n + p, p]), fromData(b, [n + p])).x))
  }
  return Float64Array.from(toFlat(lstsq(fromData(phi, [n, p]), fromData(y, [n])).x))
}

/**
 * The predictions $\Phimat\wvec$ of a row-major design $[m, p]$.
 *
 * @param phi The design $\Phimat$, row-major $[m, p]$.
 * @param m The number of rows.
 * @param p The number of columns.
 * @param w The weights $\wvec$, $p$ values.
 * @returns The $m$ predictions.
 *
 * @example Intercept and slope applied at three points
 * print(applyWeights(new Float64Array([1, -1, 1, 0, 1, 1]), 3, 2, [1, 1]))
 */
export function applyWeights(phi: Float64Array, m: number, p: number, w: ArrayLike<number>): Float64Array {
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let v = 0
    for (let j = 0; j < p; j++) v += phi[i * p + j] * w[j]
    out[i] = v
  }
  return out
}
