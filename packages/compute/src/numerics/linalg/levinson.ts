/**
 * The Levinson–Durbin recursion for symmetric Toeplitz systems (the Yule–Walker equations), part of
 * `aifn-compute/numerics/linalg`.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'

/** A vector argument as a Float64Array, through tensor's dense kernels (§4.3 #8: no private matrix helpers). */
const toVec = (v: VectorLike, where: string): number[] => Array.from(dense.toF64(v, where))
const vecT = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

/** The result of the Levinson–Durbin recursion. */
export type LevinsonDurbin = {
  /** φ₁ … φ_p of the order-p AR fit x_t = Σ φ_i x_{t−i} + ε_t. */
  ar: Vector
  /** The reflection coefficients φ_kk, k = 1 … p: the partial autocorrelations. */
  reflection: Vector
  /** The prediction-error variances v₀ = γ(0), v₁, …, v_p. */
  variance: Vector
  /** True if some |φ_kk| ≥ 1 (the autocovariance was not positive definite); later orders are then unreliable. */
  singular: boolean
}

/**
 * The Levinson–Durbin recursion (Levinson, 1947; Durbin, 1960): solves the Toeplitz Yule–Walker equations
 * Σ_j φ_j γ(i − j) = γ(i), i = 1 … p, in O(p²) from autocovariances γ(0) … γ(p). Each order k adds the reflection
 * coefficient φ_kk = (γ(k) − Σ_{j<k} φ_{k−1,j} γ(k − j)) / v_{k−1}, updates φ_kj = φ_{k−1,j} − φ_kk φ_{k−1,k−j}, and
 * v_k = v_{k−1}(1 − φ_kk²) (Brockwell & Davis, 1991, Prop. 5.2.1). Autocorrelations work too (v is then relative).
 */
export function levinsonDurbin(acov: VectorLike, order: Size): LevinsonDurbin {
  const g = toVec(acov, 'levinsonDurbin')
  if (order < 0 || order >= g.length)
    throw new ShapeError('levinsonDurbin', `levinsonDurbin: need ${order + 1} autocovariances, got ${g.length}`)
  let phi: number[] = []
  const reflection: number[] = []
  const variance = [g[0]]
  let singular = false
  for (let k = 1; k <= order; k++) {
    const v = variance[k - 1]
    let num = g[k]
    for (let j = 1; j < k; j++) num -= phi[j - 1] * g[k - j]
    const kk = v > 0 ? num / v : NaN
    if (!(Math.abs(kk) < 1)) singular = true
    phi = [...phi.map((p, j) => p - kk * phi[k - 2 - j]), kk]
    reflection.push(kk)
    variance.push(v * (1 - kk * kk))
  }
  return { ar: vecT(phi), reflection: vecT(reflection), variance: vecT(variance), singular }
}
