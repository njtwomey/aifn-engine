/**
 * Peaks over threshold: the generalised Pareto law of exceedances over a high threshold (Pickands, 1975; Balkema and de
 * Haan, 1974), fitted by maximum likelihood, and the tail estimates it gives: high quantiles beyond the data, tail
 * probabilities, and the mean excess function used to choose the threshold. Anomaly detectors use it to turn a score
 * into a calibrated threshold at a chosen risk (Siffer et al., 2017).
 *
 * Above a threshold $u$ the excesses $y = x - u$ follow, approximately, the generalised Pareto law with shape $\xi$
 * and scale $\sigma$, whose survival function is $(1 + \xi y/\sigma)^{-1/\xi}$ ($e^{-y/\sigma}$ at $\xi = 0$).
 * The data may be plain arrays or tensors of any shape (read as their flat values). Invalid input throws
 * `DomainError`.
 */

import type { DataLike, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { minimizeScalar } from 'aifn-compute/numerics/roots'
import { GeneralisedPareto, type Univariate } from 'aifn-compute/probability/distributions'
import { quantile } from 'aifn-compute/probability/stats'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The values of an array or tensor as a fresh `Float64Array` (a tensor read in row-major order, whatever its shape).
 *
 * @param x The data: a plain numeric array or a tensor. Not modified.
 * @returns A copy of its values.
 */
const valuesOf = (x: DataLike): Float64Array => Float64Array.from(isTensor(x) ? toFlat(x) : x)

/** A generalised Pareto fit to excesses $y > 0$ (location 0). */
export type GeneralisedParetoFit = {
  /**
   * The shape $\xi$ (tail index): $\xi > 0$ a heavy tail, $\xi = 0$ exponential, $\xi < 0$ bounded above (at
   * $-\sigma/\xi$). The fit keeps $\xi \ge -1$.
   */
  shape: Scalar
  /** The scale $\sigma > 0$. */
  scale: Scalar
  /** The maximised log-likelihood. */
  logLikelihood: Scalar
  /** The number $n$ of excesses fitted. */
  n: Size
}

/**
 * The maximum-likelihood fit of a generalised Pareto law with location 0 to positive excesses $y_1, \dots, y_n$, by
 * Grimshaw's (1993) reduction to one dimension: with $\theta = \xi/\sigma$ the likelihood equations give
 * $\xi(\theta) = \frac{1}{n} \sum_i \log(1 + \theta y_i)$ and $\sigma = \xi/\theta$, so the profile
 * log-likelihood is $\ell(\theta) = -n [\log(\xi(\theta)/\theta) + \xi(\theta) + 1]$ on
 * $\theta > -1/\max_i y_i$ ($\theta \to 0$ is the exponential fit, $\sigma = \bar{y}$). It is maximised by a grid
 * search over $\theta$ (141 points, from just above $-1/\max_i y_i$ to $10^4/\bar{y}$) followed by Brent's method
 * between the best grid point's neighbours. As $\theta \to -1/\max_i y_i$ the likelihood grows without bound when
 * $\xi < -1$ (no maximum exists), so the search is restricted to $\xi \ge -1$; when the likelihood still rises towards
 * that edge (excesses with a sharp upper end), the fit ends at $\xi \approx -1$, the uniform law on
 * $[0, \sigma]$, where a local search such as scipy's may stop at an interior stationary point instead. Throws
 * `DomainError` for fewer than two excesses or one that is not positive.
 *
 * @param excesses The excesses $y_i > 0$ over the threshold (not the raw observations): an array or a tensor of any
 *   shape, read as its flat values. At least two.
 * @returns The fitted `shape` $\xi$ and `scale` $\sigma$, the maximised `logLikelihood` and the number `n` of
 *   excesses.
 *
 * @example Fit excesses, as scipy's genpareto.fit with floc=0
 * const fit = fitGeneralisedPareto([0.5, 1, 1.5, 2, 3, 5, 8])
 * print('shape =', fit.shape)
 * print('scale =', fit.scale)
 * print('log-likelihood =', fit.logLikelihood)
 *
 * @example Excesses spread over orders of magnitude have a heavy tail
 * const { shape, scale } = fitGeneralisedPareto([0.2, 0.5, 1, 2, 5, 10, 20, 50])
 * print('shape =', shape)
 * print('scale =', scale)
 */
export function fitGeneralisedPareto(excesses: DataLike): GeneralisedParetoFit {
  const y = valuesOf(excesses)
  const n = y.length
  if (n < 2) throw new DomainError('fitGeneralisedPareto', 'fitGeneralisedPareto: needs at least two excesses')
  let max = 0
  let total = 0
  for (const v of y) {
    if (!(v > 0))
      throw new DomainError('fitGeneralisedPareto', `fitGeneralisedPareto: excesses must be positive, got ${v}`)
    max = Math.max(max, v)
    total += v
  }
  const mean = total / n
  const xiOf = (theta: number) => {
    let s = 0
    for (const v of y) s += Math.log1p(theta * v)
    return s / n
  }
  // The negative profile log-likelihood per point; the limit at θ = 0 is log ȳ + 1.
  const f = (theta: number) => {
    if (Math.abs(theta) * mean < 1e-10) return Math.log(mean) + 1
    const xi = xiOf(theta)
    const sigma = xi / theta
    // Below ξ = −1 the likelihood is unbounded at the largest point (no maximum); the search stays at ξ ≥ −1.
    return sigma > 0 && Number.isFinite(sigma) && xi >= -1 ? Math.log(sigma) + xi + 1 : Infinity
  }
  // A grid on θ: negative values up to just above −1/max y, and positive values on a log scale up to 1e4/ȳ.
  const lower = -1 / max
  const grid: number[] = [0]
  for (let k = 1; k <= 60; k++) grid.push(lower * (1 - Math.pow(10, -6 + (5.9 * (60 - k)) / 60)))
  for (let k = 0; k <= 80; k++) grid.push(Math.pow(10, -6 + (10 * k) / 80) / mean)
  grid.sort((a, b) => a - b)
  let best = 0
  grid.forEach((theta, i) => {
    if (f(theta) < f(grid[best])) best = i
  })
  const lo = grid[Math.max(0, best - 1)]
  const hi = grid[Math.min(grid.length - 1, best + 1)]
  const r =
    lo < hi ? minimizeScalar(f, { bounds: [lo, hi], tolerance: 1e-12 }) : { x: grid[best], value: f(grid[best]) }
  const theta = r.value <= f(grid[best]) ? r.x : grid[best]
  const exponential = Math.abs(theta) * mean < 1e-10
  const shape = exponential ? 0 : xiOf(theta)
  const scale = exponential ? mean : shape / theta
  return { shape, scale, logLikelihood: -n * f(theta), n }
}

/** Options of `peaksOverThreshold`. */
export type PeaksOverThresholdOptions = {
  /** The threshold $u$. Default: the `quantile` of the data. */
  threshold?: Scalar
  /**
   * The level of the empirical quantile used as the threshold when none is given, in $(0, 1)$ (default 0.9). The
   * quantile is the linear-interpolation one (numpy's default).
   */
  quantile?: Scalar
}

/**
 * A peaks-over-threshold fit: the generalised Pareto fit of the excesses (`shape`, `scale`, `logLikelihood` and `n`,
 * the number of excesses), with the threshold, the exceedance rate and the fitted tail law.
 */
export type PeaksOverThreshold = GeneralisedParetoFit & {
  /** The threshold $u$ used, given or chosen as an empirical quantile. */
  threshold: Scalar
  /** The number of observations $N$. */
  total: Size
  /** $\zeta_u = n/N$, the share of observations strictly above the threshold. */
  rate: Scalar
  /** The excesses $x - u$ of the observations above $u$, in the order of the data. */
  excesses: Float64Array
  /**
   * The fitted law of the observations above $u$: `GeneralisedPareto` with shape $\xi$, location $u$ and scale
   * $\sigma$.
   */
  tail: Univariate<number>
}

/**
 * Fit the tail of a sample by peaks over threshold: keep the observations strictly above a threshold $u$, fit a
 * generalised Pareto law to their excesses $x - u$ by maximum likelihood (`fitGeneralisedPareto`), and record the
 * exceedance rate $\zeta_u = n/N$. Above $u$ the survival function is then estimated by
 * $P(X > x) \approx \zeta_u (1 + \xi(x - u)/\sigma)^{-1/\xi}$ (`tailProbability`), and high quantiles beyond the
 * data by inverting it (`tailQuantile`). Throws `DomainError` for fewer than three observations, a `quantile` outside
 * $(0, 1)$, or fewer than two observations above the threshold.
 *
 * @param values The observations: an array or a tensor of any shape, read as its flat values. At least three.
 * @param options The threshold, or the quantile level that chooses it (default: the 0.9 quantile of the data).
 * @returns The fit of the excesses with the `threshold`, the `total` count $N$, the `rate` $\zeta_u$, the `excesses`
 *   themselves and the fitted `tail` law.
 *
 * @example Fit the top 40% of a sample
 * // The threshold is numpy.quantile(x, 0.6), and the excesses are fitted as by scipy's genpareto.fit with floc=0.
 * const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]
 * const fit = peaksOverThreshold(x, { quantile: 0.6 })
 * print('threshold =', fit.threshold, ' rate =', fit.rate)
 * print('excesses =', fit.excesses)
 * print('shape =', fit.shape, ' scale =', fit.scale)
 *
 * @example A fixed threshold
 * const fit = peaksOverThreshold([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50], { threshold: 10 })
 * print('excesses =', fit.excesses)
 * print('rate =', fit.rate)
 * print('P(X > 20 | X > 10) =', fit.tail.survival(20))
 */
export function peaksOverThreshold(values: DataLike, options: PeaksOverThresholdOptions = {}): PeaksOverThreshold {
  const x = valuesOf(values)
  const N = x.length
  if (N < 3) throw new DomainError('peaksOverThreshold', 'peaksOverThreshold: needs at least three observations')
  let u = options.threshold
  if (u === undefined) {
    const q = options.quantile ?? 0.9
    if (!(q > 0 && q < 1))
      throw new DomainError('peaksOverThreshold', `peaksOverThreshold: quantile must lie in (0, 1), got ${q}`)
    // The linear-interpolation quantile (NumPy's default).
    u = quantile(x, q)
  }
  const threshold = u
  const excesses = Float64Array.from(x.filter((v) => v > threshold).map((v) => v - threshold))
  if (excesses.length < 2)
    throw new DomainError('peaksOverThreshold', 'peaksOverThreshold: fewer than two observations exceed the threshold')
  const fit = fitGeneralisedPareto(excesses)
  return {
    ...fit,
    threshold,
    total: N,
    rate: excesses.length / N,
    excesses,
    tail: GeneralisedPareto(fit.shape, threshold, fit.scale),
  }
}

/**
 * The estimated tail probability $P(X > x) \approx \zeta_u (1 + \xi(x - u)/\sigma)^{-1/\xi}$ for $x \ge u$ under a
 * peaks-over-threshold fit. Below the threshold the model says nothing, and the exceedance rate $\zeta_u$ itself is
 * returned.
 *
 * @param fit The fit, as `peaksOverThreshold` returns it.
 * @param x The value whose exceedance probability is wanted.
 * @returns $P(X > x)$ for $x \ge u$; $\zeta_u$ for $x < u$.
 *
 * @example Probabilities beyond the largest observation
 * // At the threshold it is the exceedance rate, 6 of 15 observations.
 * const fit = peaksOverThreshold([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50], { quantile: 0.6 })
 * print('P(X > 9.4) =', tailProbability(fit, 9.4))
 * print('P(X > 40) =', tailProbability(fit, 40))
 * print('P(X > 100) =', tailProbability(fit, 100))
 * print('P(X > 5), below the threshold =', tailProbability(fit, 5))
 */
export function tailProbability(fit: PeaksOverThreshold, x: Scalar): Scalar {
  if (x < fit.threshold) return fit.rate
  return fit.rate * (fit.tail.survival(x) as number)
}

/**
 * The quantile $x_p$ with $P(X > x_p) = 1 - p$, for $p \ge 1 - \zeta_u$:
 * $x_p = u + (\sigma/\xi)[((1 - p)/\zeta_u)^{-\xi} - 1]$ ($u - \sigma \log((1 - p)/\zeta_u)$ at $\xi = 0$). With
 * $p = 1 - q$ this is the anomaly threshold at risk $q$ of Siffer et al. (2017). Throws `DomainError` when $p$ is
 * outside $[0, 1)$ or below $1 - \zeta_u$ (a quantile under the threshold, which the tail model does not describe).
 *
 * @param fit The fit, as `peaksOverThreshold` returns it.
 * @param p The probability level, in $[1 - \zeta_u, 1)$. At $p = 1 - \zeta_u$ the quantile is the threshold $u$.
 * @returns The quantile $x_p$.
 *
 * @example The 99% quantile, beyond the data
 * // The largest observation is 50; p = 0.6 = 1 - rate gives the threshold back.
 * const fit = peaksOverThreshold([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50], { quantile: 0.6 })
 * print('x at p = 0.6 =', tailQuantile(fit, 0.6))
 * print('x at p = 0.99 =', tailQuantile(fit, 0.99))
 * print('P(X > x) back =', tailProbability(fit, tailQuantile(fit, 0.99)))
 */
export function tailQuantile(fit: PeaksOverThreshold, p: Scalar): Scalar {
  if (!(p >= 0 && p < 1)) throw new DomainError('tailQuantile', `tailQuantile: p must lie in [0, 1), got ${p}`)
  const r = (1 - p) / fit.rate
  if (r > 1)
    throw new DomainError('tailQuantile', `tailQuantile: p = ${p} lies below the threshold (needs p ≥ 1 − ζ_u)`)
  return fit.tail.isf(r) as number
}

/**
 * The empirical mean excess $e(u)$, the mean of $x - u$ over the observations with $x > u$, at each threshold (the mean
 * residual life plot). For a generalised Pareto tail it is linear in $u$ with slope $\xi/(1 - \xi)$, so the threshold
 * is chosen where the plot becomes linear. NaN where fewer than `minimum` observations exceed $u$.
 *
 * @param values The observations: an array or a tensor of any shape, read as its flat values.
 * @param thresholds The thresholds $u$ at which to evaluate $e(u)$, in any order.
 * @param minimum The least number of observations that must exceed a threshold for $e(u)$ to be reported; below it the
 *   entry is NaN (as it is whenever none exceed it).
 * @returns $e(u)$ for each threshold, in the order of `thresholds`.
 *
 * @example Mean excesses of 1 to 10
 * // Over 2 the excesses are 1, ..., 8; nothing exceeds 10.
 * print('e(u) =', meanExcess([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 2, 5, 9, 10]))
 *
 * @example Require enough exceedances
 * print('e(u) =', meanExcess([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 5, 7, 8], 3))
 */
export function meanExcess(values: DataLike, thresholds: DataLike, minimum: Size = 1): Float64Array {
  const x = valuesOf(values)
  return valuesOf(thresholds).map((u) => {
    let s = 0
    let k = 0
    for (const v of x)
      if (v > u) {
        s += v - u
        k++
      }
    return k >= minimum && k > 0 ? s / k : NaN
  })
}
