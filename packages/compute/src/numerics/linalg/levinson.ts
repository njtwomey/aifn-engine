/**
 * The Levinson–Durbin recursion for symmetric Toeplitz systems (the Yule–Walker equations), part of
 * `aifn-compute/numerics/linalg`.
 */

import type { Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'

/**
 * A vector argument as a plain array of numbers, read through tensor's dense kernels (§4.3 #8: no private matrix
 * helpers).
 *
 * @param v The vector to read, as a tensor or an array of numbers; not modified.
 * @param where The caller's name, used in the error message when `v` is not a vector.
 * @returns The entries of `v` in order, copied into a new plain array of numbers.
 */
const toVec = (v: VectorLike, where: string): number[] => Array.from(dense.toF64(v, where))
/**
 * An array of numbers as a float64 vector tensor.
 *
 * @param v The values of the vector, in order; copied, not modified.
 * @returns A float64 tensor of shape $[\text{length}]$ holding the values of `v`.
 */
const vecT = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

/** The result of the Levinson–Durbin recursion. */
export type LevinsonDurbin = {
  /** $\phi_1, \dots, \phi_p$ of the order-$p$ AR fit $x_t = \sum_i \phi_i x_{t-i} + \varepsilon_t$. */
  ar: Vector
  /** The reflection coefficients $\phi_{kk}$, $k = 1, \dots, p$: the partial autocorrelations. */
  reflection: Vector
  /** The prediction-error variances $v_0 = \gamma(0), v_1, \dots, v_p$. */
  variance: Vector
  /**
   * True if some $|\phi_{kk}| \ge 1$ (the autocovariance was not positive definite); later orders are then
   * unreliable.
   */
  singular: boolean
}

/**
 * The Levinson–Durbin recursion (Levinson, 1947; Durbin, 1960): solves the Toeplitz Yule–Walker equations
 * $\sum_j \phi_j \gamma(i - j) = \gamma(i)$, $i = 1, \dots, p$, in $O(p^2)$ from autocovariances
 * $\gamma(0), \dots, \gamma(p)$. Each order $k$ adds the reflection coefficient
 * $\phi_{kk} = (\gamma(k) - \sum_{j<k} \phi_{k-1,j} \gamma(k - j)) / v_{k-1}$, updates
 * $\phi_{kj} = \phi_{k-1,j} - \phi_{kk} \phi_{k-1,k-j}$, and $v_k = v_{k-1}(1 - \phi_{kk}^2)$ (Brockwell & Davis, 1991,
 * Prop. 5.2.1). Autocorrelations work too ($v$ is then relative).
 *
 * @param acov The autocovariances $\gamma(0), \gamma(1), \dots$ indexed by lag, at least `order` + 1 of them; only the
 *   first `order` + 1 are read. Autocorrelations (with $\gamma(0) = 1$) may be given instead.
 * @param order The order $p$ of the AR fit: the number of recursion steps, from 0 (no coefficients) up to one less than
 *   the number of autocovariances. Outside that range throws `ShapeError`.
 * @returns `ar`, the $p$ coefficients of the order-$p$ fit; `reflection`, the $p$ reflection coefficients;
 *   `variance`, the $p + 1$ prediction-error variances of orders 0 to $p$; and `singular`, true when some reflection
 *   coefficient was not below 1 in magnitude.
 *
 * @example Fit an AR(2) model from autocovariances
 * // Autocovariances at lags 0, 1, 2 of an AR process.
 * const { ar, reflection, variance } = levinsonDurbin(tensor([1, 0.5, 0.2]), 2)
 * print('AR coefficients =', ar)
 * print('reflection coefficients =', reflection)
 * print('innovation variance by order =', variance)
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
