/**
 * Concentration inequalities against simulated truth, and the limit theorems. For the mean X̄ₙ of n independent
 * variables in [0, 1] with mean μ and variance σ², the two-sided tail P(|X̄ₙ − μ| ≥ t) is estimated by Monte Carlo and
 * compared with:
 *
 * - Chebyshev: σ²/(n t²);
 * - Hoeffding (1963): 2 exp(−2 n t²);
 * - Bernstein: 2 exp(−n t² / (2σ² + 2t/3));
 * - Chernoff–Hoeffding (relative entropy): exp(−n KL(μ + t ‖ μ)) + exp(−n KL(μ − t ‖ μ)), with KL between Bernoulli
 *   laws, valid for any law on [0, 1] (Hoeffding's Theorem 1).
 *
 * McDiarmid's inequality is checked on a function of many independent variables that is not a sum: the fraction of
 * empty bins after n balls fall into m bins, whose value changes by at most 1/m when one ball moves, so
 * P(|f − E f| ≥ t) ≤ 2 exp(−2 t² m² / n). The law of large numbers is shown by running means, the central limit
 * theorem by standardised sums √n (X̄ₙ − μ)/σ, including a heavy-tailed law where it fails.
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Laws to average: bounded ones for the inequalities; exponential and Pareto (α = 1.5, infinite variance) for the CLT. */
export type SummandLaw = 'bernoulli' | 'uniform' | 'arcsine' | 'exponential' | 'pareto'

/** The mean and variance of a law (Bernoulli with parameter p); variance Infinity for the Pareto law. */
export function lawMoments(law: SummandLaw, p = 0.3): { mean: number; variance: number; bounded: boolean } {
  switch (law) {
    case 'bernoulli':
      return { mean: p, variance: p * (1 - p), bounded: true }
    case 'uniform':
      return { mean: 0.5, variance: 1 / 12, bounded: true }
    case 'arcsine':
      return { mean: 0.5, variance: 1 / 8, bounded: true }
    case 'exponential':
      return { mean: 1, variance: 1, bounded: false }
    case 'pareto':
      // Pareto with x_m = 1 and α = 1.5: mean α/(α − 1) = 3, infinite variance.
      return { mean: 3, variance: Infinity, bounded: false }
  }
}

/** A draw of the law from a uniform u. */
function inverse(law: SummandLaw, u: number, p: number): number {
  switch (law) {
    case 'bernoulli':
      return u < p ? 1 : 0
    case 'uniform':
      return u
    case 'arcsine':
      return Math.sin((Math.PI * u) / 2) ** 2
    case 'exponential':
      return -Math.log(1 - u)
    case 'pareto':
      return Math.pow(1 - u, -1 / 1.5)
  }
}

/** `trials` sample means of n draws each. */
export function sampleMeans(s: Stream, law: SummandLaw, n: number, trials: number, p = 0.3): Float64Array {
  const out = new Float64Array(trials)
  for (let r = 0; r < trials; r++) {
    const u = units(child(s, 'trial', r), n)
    let sum = 0
    for (let i = 0; i < n; i++) sum += inverse(law, u[i], p)
    out[r] = sum / n
  }
  return out
}

/** KL(a ‖ b) between Bernoulli laws (Infinity outside [0, 1]). */
export function bernoulliKl(a: number, b: number): number {
  if (a < 0 || a > 1) return Infinity
  const term = (x: number, y: number) => (x === 0 ? 0 : x * Math.log(x / y))
  return term(a, b) + term(1 - a, 1 - b)
}

/** Every bound at deviations t for the mean of n variables in [0, 1] with mean μ and variance σ² (module docs). */
export function tailBounds(
  t: ArrayLike<number>,
  n: number,
  mean: number,
  variance: number,
): { chebyshev: Float64Array; hoeffding: Float64Array; bernstein: Float64Array; chernoff: Float64Array } {
  const cap = (v: number) => Math.min(1, v)
  return {
    chebyshev: Float64Array.from(t, (v) => cap(variance / (n * v * v))),
    hoeffding: Float64Array.from(t, (v) => cap(2 * Math.exp(-2 * n * v * v))),
    bernstein: Float64Array.from(t, (v) => cap(2 * Math.exp((-n * v * v) / (2 * variance + (2 * v) / 3)))),
    chernoff: Float64Array.from(t, (v) =>
      cap(Math.exp(-n * bernoulliKl(mean + v, mean)) + Math.exp(-n * bernoulliKl(mean - v, mean))),
    ),
  }
}

/** Options of `concentrationStudy`. */
export interface ConcentrationOptions {
  law?: SummandLaw
  /** Bernoulli parameter (default 0.3). */
  p?: number
  /** Variables per mean (default 50). */
  n?: number
  /** Monte Carlo trials (default 20 000). */
  trials?: number
  /** Deviations t at which tails are read (default 60 points on (0, 0.5]). */
  thresholds?: readonly number[]
}

/** The empirical tail of |X̄ₙ − μ| against the bounds, and the sample means themselves. */
export interface ConcentrationStudy {
  readonly t: Float64Array
  readonly empirical: Float64Array
  readonly chebyshev: Float64Array
  readonly hoeffding: Float64Array
  readonly bernstein: Float64Array
  readonly chernoff: Float64Array
  readonly means: Float64Array
  readonly mean: number
  readonly variance: number
}

/** Monte Carlo tails of a bounded law's sample mean against Chebyshev, Hoeffding, Bernstein and Chernoff. */
export function concentrationStudy(s: Stream, options: ConcentrationOptions = {}): ConcentrationStudy {
  const { law = 'bernoulli', p = 0.3, n = 50, trials = 20_000 } = options
  const { mean, variance, bounded } = lawMoments(law, p)
  if (!bounded)
    throw new DomainError('concentrationStudy', `concentrationStudy: the ${law} law is not bounded in [0, 1]`)
  const t = Float64Array.from(options.thresholds ?? Array.from({ length: 60 }, (_, i) => (0.5 * (i + 1)) / 60))
  const means = sampleMeans(s, law, n, trials, p)
  const dev = Float64Array.from(means, (m) => Math.abs(m - mean)).sort()
  // P(|X̄ − μ| ≥ t) by counting (with a tolerance so that lattice values of a Bernoulli mean count at their t).
  const empirical = Float64Array.from(t, (v) => {
    let lo = 0
    let hi = dev.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (dev[mid] < v - 1e-12) lo = mid + 1
      else hi = mid
    }
    return (dev.length - lo) / dev.length
  })
  return { t, empirical, ...tailBounds(t, n, mean, variance), means, mean, variance }
}

/** Options of `mcdiarmidStudy`. */
export interface McdiarmidOptions {
  balls?: number
  bins?: number
  trials?: number
  thresholds?: readonly number[]
}

/** The fraction of empty bins after `balls` uniform throws into `bins` bins, its tail and McDiarmid's bound. */
export function mcdiarmidStudy(
  s: Stream,
  options: McdiarmidOptions = {},
): { t: Float64Array; empirical: Float64Array; bound: Float64Array; values: Float64Array; expected: number } {
  const { balls = 100, bins = 100, trials = 5000 } = options
  const t = Float64Array.from(options.thresholds ?? Array.from({ length: 50 }, (_, i) => (0.25 * (i + 1)) / 50))
  const values = new Float64Array(trials)
  const filled = new Uint8Array(bins)
  for (let r = 0; r < trials; r++) {
    filled.fill(0)
    const u = units(child(s, 'trial', r), balls)
    for (const v of u) filled[Math.floor(v * bins)] = 1
    let empty = 0
    for (const v of filled) empty += 1 - v
    values[r] = empty / bins
  }
  // E f = (1 − 1/m)ⁿ exactly.
  const expected = Math.pow(1 - 1 / bins, balls)
  const empirical = Float64Array.from(
    t,
    (v) => values.filter((x) => Math.abs(x - expected) >= v - 1e-12).length / trials,
  )
  const bound = Float64Array.from(t, (v) => Math.min(1, 2 * Math.exp((-2 * v * v * bins * bins) / balls)))
  return { t, empirical, bound, values, expected }
}

/** Running means X̄₁, …, X̄ₙ of `paths` independent sequences, row-major [paths × n]. */
export function runningMeans(s: Stream, law: SummandLaw, n: number, paths: number, p = 0.3): Float64Array {
  const out = new Float64Array(paths * n)
  for (let r = 0; r < paths; r++) {
    const u = units(child(s, 'path', r), n)
    let sum = 0
    for (let i = 0; i < n; i++) {
      sum += inverse(law, u[i], p)
      out[r * n + i] = sum / (i + 1)
    }
  }
  return out
}

/**
 * Standardised sums √n (X̄ₙ − μ)/σ over `trials` samples. For the Pareto law (infinite variance) the sums are centred
 * and scaled as its stable limit needs, (Sₙ − nμ)/n^{1/α} = n^{1 − 1/α} (X̄ₙ − μ) = n^{1/3} (X̄ₙ − μ), so the histogram
 * stays finite but skewed instead of normal.
 */
export function standardisedSums(s: Stream, law: SummandLaw, n: number, trials: number, p = 0.3): Float64Array {
  const { mean, variance } = lawMoments(law, p)
  const means = sampleMeans(s, law, n, trials, p)
  const scale = Number.isFinite(variance) ? Math.sqrt(n / variance) : Math.pow(n, 1 - 1 / 1.5)
  return Float64Array.from(means, (m) => (m - mean) * scale)
}
