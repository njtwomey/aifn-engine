/**
 * Peaks over threshold: the generalised Pareto law of exceedances over a high threshold (Pickands, 1975; Balkema and de
 * Haan, 1974), fitted by maximum likelihood, and the tail estimates it gives: high quantiles beyond the data, tail
 * probabilities, and the mean excess function used to choose the threshold. Anomaly detectors use it to turn a score
 * into a calibrated threshold at a chosen risk (Siffer et al., 2017).
 */

import type { DataLike, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { minimizeScalar } from 'aifn-compute/numerics/roots'
import { GeneralisedPareto, type Univariate } from 'aifn-compute/probability/distributions'
import { quantile } from 'aifn-compute/probability/stats'
import { DomainError } from 'aifn-compute/foundation/errors'

const valuesOf = (x: DataLike): Float64Array => Float64Array.from(isTensor(x) ? toFlat(x) : x)

/** A generalised Pareto fit to exceedances y > 0 (location 0). */
export type GeneralisedParetoFit = {
  /** The shape ξ (tail index): > 0 heavy tail, 0 exponential, < 0 bounded. */
  shape: Scalar
  /** The scale σ > 0. */
  scale: Scalar
  /** The maximised log-likelihood. */
  logLikelihood: Scalar
  /** The number of exceedances. */
  n: Size
}

/**
 * The maximum-likelihood fit of a generalised Pareto law with location 0 to positive excesses y₁ … yₙ, by Grimshaw's
 * (1993) reduction to one dimension: with θ = ξ/σ the likelihood equations give ξ(θ) = (1/n) Σ log(1 + θyᵢ) and
 * σ = ξ/θ, so the profile log-likelihood is ℓ(θ) = −n [log(ξ(θ)/θ) + ξ(θ) + 1] on θ > −1/max y (θ → 0 is the exponential
 * fit, σ = ȳ). It is maximised by a grid search over θ followed by Brent's method. As θ → −1/max y the likelihood
 * grows without bound when ξ < −1 (no maximum exists), so the search is restricted to ξ ≥ −1.
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
  /** The threshold u. Default: the `quantile` of the data. */
  threshold?: Scalar
  /** The empirical quantile used as the threshold when none is given (default 0.9). */
  quantile?: Scalar
}

/** A peaks-over-threshold fit: the threshold, the exceedance rate and the GPD of the excesses. */
export type PeaksOverThreshold = GeneralisedParetoFit & {
  threshold: Scalar
  /** The number of observations N. */
  total: Size
  /** ζ_u = n/N, the share of observations above the threshold. */
  rate: Scalar
  /** The excesses x − u of the observations above u. */
  excesses: Float64Array
  /** The fitted law of the observations above u: GeneralisedPareto(ξ, u, σ). */
  tail: Univariate<number>
}

/**
 * Fit the tail of a sample by peaks over threshold: keep the observations above a threshold u, fit a generalised
 * Pareto law to their excesses x − u by maximum likelihood, and record the exceedance rate ζ_u = n/N. Above u the
 * survival function is then estimated by P(X > x) ≈ ζ_u (1 + ξ(x − u)/σ)^{−1/ξ} (`tailProbability`), and high quantiles
 * beyond the data by inverting it (`tailQuantile`).
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

/** P(X > x) ≈ ζ_u (1 + ξ(x − u)/σ)^{−1/ξ} for x ≥ u under a peaks-over-threshold fit (the exceedance rate below u). */
export function tailProbability(fit: PeaksOverThreshold, x: Scalar): Scalar {
  if (x < fit.threshold) return fit.rate
  return fit.rate * (fit.tail.survival(x) as number)
}

/**
 * The quantile x_p with P(X > x_p) = 1 − p, for p ≥ 1 − ζ_u: x_p = u + (σ/ξ)[((1 − p)/ζ_u)^{−ξ} − 1] (u − σ log((1 − p)/ζ_u)
 * at ξ = 0). With p = 1 − q this is the anomaly threshold at risk q of Siffer et al. (2017).
 */
export function tailQuantile(fit: PeaksOverThreshold, p: Scalar): Scalar {
  if (!(p >= 0 && p < 1)) throw new DomainError('tailQuantile', `tailQuantile: p must lie in [0, 1), got ${p}`)
  const r = (1 - p) / fit.rate
  if (r > 1)
    throw new DomainError('tailQuantile', `tailQuantile: p = ${p} lies below the threshold (needs p ≥ 1 − ζ_u)`)
  return fit.tail.isf(r) as number
}

/**
 * The empirical mean excess e(u) = mean(x − u | x > u) at each threshold (the mean residual life plot). For a
 * generalised Pareto tail it is linear in u with slope ξ/(1 − ξ), so the threshold is chosen where the plot becomes
 * linear. NaN where fewer than `minimum` observations (default 1) exceed u.
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
