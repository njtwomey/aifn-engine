/**
 * Autoregressive (AR) estimation by the Levinson recursion, part of `aifn-compute/signal/statistical`: Yule–Walker
 * (through the Levinson–Durbin recursion of `aifn-compute/numerics/linalg`) and Burg's method.
 *
 * Both fit the AR($p$) model $x_t - \mu = \sum_{i=1}^{p} \phi_i (x_{t-i} - \mu) + \varepsilon_t$ order by order, one
 * reflection coefficient $k_m$ per order, so both return the partial autocorrelations with the coefficients, and
 * since $\lvert k_m \rvert < 1$ at every order (barring a degenerate series) the fitted model is stationary.
 * Yule–Walker works from the sample autocovariances, Burg from the forward and backward prediction errors of the
 * series itself. The least-squares fit, which does not guarantee stationarity, is `leastSquaresAr` in the parametric
 * file.
 */

import { levinsonDurbin } from 'aifn-compute/numerics/linalg'
import { autocovariance } from 'aifn-compute/probability/stats'
import { dense, fromData, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { isSignal, type SignalInput } from '../signal'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The samples of a signal or vector as a plain array of numbers (not a Float64Array), read through the tensor
 * package's dense kernels (§4.3 #8: no private matrix helpers). A signal's channels are not checked.
 *
 * @param v The series: a signal (its data is read) or its samples.
 * @param where The caller's name for error messages.
 * @returns A fresh array of the samples.
 */
const toVec = (v: SignalInput, where: string): number[] => Array.from(dense.toF64(isSignal(v) ? v.data : v, where))
/**
 * A rank-1 float64 tensor holding a copy of the values.
 *
 * @param v The values.
 * @returns The tensor, of length `v.length`.
 */
const vecT = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
/**
 * The arithmetic mean (NaN for no values).
 *
 * @param x The values.
 * @returns Their mean.
 */
const meanOf = (x: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return s / x.length
}

/**
 * An autoregressive fit $x_t - \mu = \sum_{i=1}^{p} \phi_i (x_{t-i} - \mu) + \varepsilon_t$,
 * $\varepsilon_t \sim \Gauss(0, \sigma^2)$.
 */
export type AutoregressiveFit = {
  /** The coefficients $\phi_1, \dots, \phi_p$, by lag. */
  ar: Vector
  /** Innovation variance $\sigma^2$. */
  sigma2: Scalar
  /** The mean $\mu$ that was subtracted (the sample mean, or 0 with `demean: false`). */
  mean: Scalar
  /** The partial autocorrelations (reflection coefficients) of orders $1, \dots, p$. */
  reflection: Vector
}

/**
 * Yule–Walker estimates of an AR($p$): solve the Yule–Walker equations with the biased (divisor $n$) sample
 * autocovariances by Levinson–Durbin, so the fitted polynomial is always stationary; $\sigma^2$ is the order-$p$
 * prediction-error variance (statsmodels' `yule_walker(x, p, method='mle')`). Throws `ShapeError` (from
 * `levinsonDurbin`) unless $0 \le p < n$.
 *
 * @param x The series, of length $n$: a signal or its samples.
 * @param order The order $p$.
 * @param options The fitting options.
 * @param options.demean Subtract the sample mean first (default true); with false the series is taken to have mean 0.
 * @returns The coefficients, innovation variance, the mean subtracted and the reflection coefficients.
 *
 * @example An AR(1) process: its lag-1 autocorrelation is the coefficient
 * // x_t = 0.8 x_{t-1} + e_t with unit-variance noise, 500 samples.
 * const e = toArray(normals(stream(1), 500))
 * const x = [e[0]]
 * for (let t = 1; t < 500; t++) x.push(0.8 * x[t - 1] + e[t])
 * const fit = yuleWalker(x, 1)
 * print('phi =', fit.ar)
 * print('sigma2 =', fit.sigma2)
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
 * of forward and backward prediction-error powers,
 * $k_m = 2\sum_t f_t b_{t-1} / \sum_t (f_t^2 + b_{t-1}^2)$ over $t = m, \dots, n - 1$, then update the errors and
 * the coefficients by the Levinson step. The estimate is always stationary and, unlike Yule–Walker, uses no
 * zero-padded lags, so it resolves sharp spectral peaks in short series. $\sigma^2$ is the order-0 power
 * $\frac{1}{n}\sum_t (x_t - \mu)^2$ times $\prod_m (1 - k_m^2)$. Throws `DomainError` unless $p < n$.
 *
 * @param x The series, of length $n$: a signal or its samples.
 * @param order The order $p$.
 * @param options The fitting options.
 * @param options.demean Subtract the sample mean first (default true); with false the series is taken to have mean 0.
 * @returns The coefficients, innovation variance, the mean subtracted and the reflection coefficients.
 *
 * @example A short AR(2) series, by Burg and by Yule–Walker
 * // x_t = 1.2 x_{t-1} - 0.6 x_{t-2} + e_t, 60 samples.
 * const e = toArray(normals(stream(2), 60))
 * const x = [e[0], e[1]]
 * for (let t = 2; t < 60; t++) x.push(1.2 * x[t - 1] - 0.6 * x[t - 2] + e[t])
 * const fit = burg(x, 2)
 * print('Burg phi =', fit.ar)
 * print('Burg reflection =', fit.reflection)
 * print('Yule–Walker phi =', yuleWalker(x, 2).ar)
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
