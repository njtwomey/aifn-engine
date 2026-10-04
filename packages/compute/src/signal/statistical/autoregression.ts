/**
 * Autoregressive estimation, part of `aifn-compute/signal/statistical`: Yule–Walker (through `aifn-compute/numerics/linalg`'s
 * Levinson–Durbin recursion) and Burg's method.
 */

import { levinsonDurbin } from 'aifn-compute/numerics/linalg'
import { autocovariance } from 'aifn-compute/probability/stats'
import { dense, fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { isSignal, type SignalInput } from '../signal'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A vector argument as a Float64Array, through tensor's dense kernels (§4.3 #8: no private matrix helpers). */
const toVec = (v: SignalInput, where: string): number[] => Array.from(dense.toF64(isSignal(v) ? v.data : v, where))
const vecT = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
const meanOf = (x: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return s / x.length
}

/** An autoregressive fit x_t − μ = Σ φ_i (x_{t−i} − μ) + ε_t, ε_t ~ N(0, σ²). */
export type AutoregressiveFit = {
  ar: Vector
  /** Innovation variance σ². */
  sigma2: Scalar
  /** The mean μ that was subtracted (the sample mean, or 0 with `demean: false`). */
  mean: Scalar
  /** The partial autocorrelations (reflection coefficients) of orders 1 … p. */
  reflection: Vector
}

/**
 * Yule–Walker estimates of an AR(p): solve the Yule–Walker equations with the biased (÷ n) sample autocovariances by
 * Levinson–Durbin, so the fitted polynomial is always stationary; σ² is the order-p prediction-error variance
 * (statsmodels' `yule_walker(x, p, method='mle')`).
 */
export function yuleWalker(
  x: SignalInput,
  order: Size,
  { demean = true }: { demean?: boolean } = {},
): AutoregressiveFit {
  const xs = toVec(x, 'yuleWalker')
  const mean = demean ? meanOf(xs) : 0
  const g = autocovariance(xs, { maxLag: order, demean })
  const ld = levinsonDurbin(g, order)
  return { ar: ld.ar, sigma2: (ld.variance.data as Float64Array)[order], mean, reflection: ld.reflection }
}

/**
 * Burg's method (Burg, 1968; Kay, 1988, §7.4): at each order choose the reflection coefficient that minimises the sum
 * of forward and backward prediction-error powers, k_m = 2Σ f_t b_{t−1} / Σ (f_t² + b_{t−1}²), then update the errors
 * and the coefficients by the Levinson step. The estimate is always stationary and, unlike Yule–Walker, uses no
 * zero-padded lags, so it resolves sharp spectral peaks in short series. σ² is the final mean error power.
 */
export function burg(x: SignalInput, order: Size, { demean = true }: { demean?: boolean } = {}): AutoregressiveFit {
  const xs = toVec(x, 'burg')
  const n = xs.length
  if (order >= n) throw new DomainError('burg', `burg: order ${order} needs more than ${n} values`)
  const mean = demean ? meanOf(xs) : 0
  let f = xs.map((v) => v - mean)
  let b = [...f]
  let power = f.reduce((s, v) => s + v * v, 0) / n
  let phi: number[] = []
  const reflection: number[] = []
  for (let m = 1; m <= order; m++) {
    // Forward errors f_t for t = m … n−1 against backward errors b_{t−1}.
    let num = 0
    let den = 0
    for (let t = m; t < n; t++) {
      num += f[t] * b[t - 1]
      den += f[t] * f[t] + b[t - 1] * b[t - 1]
    }
    const k = den > 0 ? (2 * num) / den : 0
    const nf = [...f]
    const nb = [...b]
    for (let t = m; t < n; t++) {
      nf[t] = f[t] - k * b[t - 1]
      nb[t] = b[t - 1] - k * f[t]
    }
    f = nf
    b = nb
    phi = [...phi.map((p, j) => p - k * phi[m - 2 - j]), k]
    reflection.push(k)
    power *= 1 - k * k
  }
  return { ar: vecT(phi), sigma2: power, mean, reflection: vecT(reflection) }
}
