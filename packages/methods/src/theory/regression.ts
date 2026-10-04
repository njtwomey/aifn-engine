/**
 * The shared layer of `aifn-methods/theory`: one-dimensional regression problems with a known regression function on
 * [−1, 1], noisy training sets drawn from them, and least-squares fits of a feature map: minimum-norm when the system
 * is underdetermined, and ridge-penalised through the augmented system [Φ; √λ I] w ≈ [y; 0].
 */

import { child, standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { lstsq } from 'aifn-compute/numerics/linalg'

/** A regression function on [−1, 1]. */
export type TargetName = 'sine' | 'step' | 'quadratic' | 'wiggly'

/** f(x) of a named target. */
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

/** n training pairs: x uniform on [−1, 1] and y = f(x) + σ ε with ε ~ N(0, 1). */
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

/** An even grid of m points on [−1, 1]. */
export const unitGrid = (m: number): Float64Array => Float64Array.from({ length: m }, (_, i) => -1 + (2 * i) / (m - 1))

/**
 * Least-squares weights of the design Φ [n, p] (row-major) for targets y: the minimum-norm solution (`lstsq`, by the
 * SVD) when λ = 0, else the ridge solution argmin ‖Φw − y‖² + λ‖w‖².
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

/** Φ·w for a row-major design [m, p]. */
export function applyWeights(phi: Float64Array, m: number, p: number, w: ArrayLike<number>): Float64Array {
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let v = 0
    for (let j = 0; j < p; j++) v += phi[i * p + j] * w[j]
    out[i] = v
  }
  return out
}
