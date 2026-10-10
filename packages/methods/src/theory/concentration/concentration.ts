/**
 * Concentration inequalities against simulated truth, and the limit theorems. For the mean $\bar{X}_n$ of $n$
 * independent variables in $[0, 1]$ with mean $\mu$ and variance $\sigma^2$, the two-sided tail
 * $P(\lvert \bar{X}_n - \mu \rvert \ge t)$ is estimated by Monte Carlo and compared with:
 *
 * - Chebyshev: $\sigma^2/(n t^2)$;
 * - Hoeffding (1963): $2 \exp(-2 n t^2)$;
 * - Bernstein: $2 \exp(-n t^2 / (2\sigma^2 + 2t/3))$;
 * - Chernoff–Hoeffding (relative entropy): $\exp(-n \KL(\mu + t \Vert \mu)) + \exp(-n \KL(\mu - t \Vert \mu))$, with
 *   $\KL$ between Bernoulli laws, valid for any law on $[0, 1]$ (Hoeffding's Theorem 1).
 *
 * Every bound is capped at 1. McDiarmid's inequality is checked on a function of many independent variables that is
 * not a sum: the fraction of empty bins after $n$ balls fall into $m$ bins, whose value changes by at most $1/m$ when
 * one ball moves, so $P(\lvert f - \expect f \rvert \ge t) \le 2 \exp(-2 t^2 m^2 / n)$. The law of large numbers is
 * shown by running means, the central limit theorem by standardised sums $\sqrt{n} (\bar{X}_n - \mu)/\sigma$,
 * including a heavy-tailed law where it fails. Every simulation draws trial $r$ from a child stream of the one given,
 * so the given stream is never advanced.
 */

import { child, units, type Stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Laws to average: bounded ones on $[0, 1]$ for the inequalities (Bernoulli with parameter $p$, uniform, and arcsine,
 * the law of $\sin^2(\pi U/2)$); exponential (rate 1) and Pareto ($x_m = 1$, $\alpha = 1.5$, infinite variance) for
 * the CLT.
 */
export type SummandLaw = 'bernoulli' | 'uniform' | 'arcsine' | 'exponential' | 'pareto'

/**
 * The mean and variance of a law; variance `Infinity` for the Pareto law.
 *
 * @param law The law.
 * @param p The Bernoulli parameter; ignored by the other laws.
 * @returns The `mean`, the `variance`, and whether the law is `bounded` in $[0, 1]$ (so the inequalities apply).
 *
 * @example Every law's moments
 * for (const law of ['bernoulli', 'uniform', 'arcsine', 'exponential', 'pareto']) print(law, lawMoments(law))
 */
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

/**
 * A draw of the law from a uniform $u$, by inverting its distribution function.
 *
 * @param law The law.
 * @param u A uniform draw in $[0, 1)$.
 * @param p The Bernoulli parameter; ignored by the other laws.
 * @returns The draw.
 */
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

/**
 * `trials` sample means $\bar{X}_n$ of $n$ draws each.
 *
 * @param s The stream; trial $r$ draws from `child(s, 'trial', r)`.
 * @param law The law of each draw.
 * @param n The draws per mean.
 * @param trials The number of means.
 * @param p The Bernoulli parameter; ignored by the other laws.
 * @returns The `trials` sample means.
 *
 * @example The spread of a mean shrinks as $1/\sqrt{n}$
 * for (const n of [10, 100]) {
 *   const m = sampleMeans(stream(0), 'uniform', n, 1000)
 *   const avg = m.reduce((a, b) => a + b, 0) / m.length
 *   const sd = Math.sqrt(m.reduce((a, b) => a + (b - avg) ** 2, 0) / m.length)
 *   print('n =', n, ' mean of means =', avg, ' sd =', sd, ' theory =', Math.sqrt(1 / 12 / n))
 * }
 */
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

/**
 * $\KL(a \Vert b) = a \ln(a/b) + (1 - a) \ln((1 - a)/(1 - b))$ between Bernoulli laws, in nats (`Infinity` for $a$
 * outside $[0, 1]$, so that a deviation past the end of $[0, 1]$ has bound 0).
 *
 * @param a The first law's parameter.
 * @param b The second law's parameter, in $(0, 1)$.
 * @returns The divergence, at least 0.
 *
 * @example Divergence grows with the gap, and is asymmetric
 * print('KL(0.4 || 0.3) =', bernoulliKl(0.4, 0.3))
 * print('KL(0.3 || 0.4) =', bernoulliKl(0.3, 0.4))
 * print('KL(0.6 || 0.3) =', bernoulliKl(0.6, 0.3))
 */
export function bernoulliKl(a: number, b: number): number {
  if (a < 0 || a > 1) return Infinity
  const term = (x: number, y: number) => (x === 0 ? 0 : x * Math.log(x / y))
  return term(a, b) + term(1 - a, 1 - b)
}

/**
 * Every bound at deviations $t$ for the mean of $n$ variables in $[0, 1]$ with mean $\mu$ and variance $\sigma^2$ (see
 * the file's introduction), each capped at 1.
 *
 * @param t The deviations $t > 0$.
 * @param n The number of variables averaged.
 * @param mean Their mean $\mu$, used by the Chernoff bound.
 * @param variance Their variance $\sigma^2$, used by the Chebyshev and Bernstein bounds.
 * @returns The Chebyshev, Hoeffding, Bernstein and Chernoff bounds on $P(\lvert \bar{X}_n - \mu \rvert \ge t)$, one
 *   per deviation.
 *
 * @example The four bounds for 100 Bernoulli(0.3) variables
 * const b = tailBounds([0.05, 0.1, 0.15], 100, 0.3, 0.21)
 * print('chebyshev', b.chebyshev)
 * print('hoeffding', b.hoeffding)
 * print('bernstein', b.bernstein)
 * print('chernoff ', b.chernoff)
 */
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
  /** The law averaged (default `bernoulli`); it must be bounded in $[0, 1]$. */
  law?: SummandLaw
  /** Bernoulli parameter (default 0.3). */
  p?: number
  /** Variables per mean (default 50). */
  n?: number
  /** Monte Carlo trials (default 20 000). */
  trials?: number
  /** Deviations $t$ at which tails are read (default 60 even points on $(0, 0.5]$). */
  thresholds?: readonly number[]
}

/** The empirical tail of $\lvert \bar{X}_n - \mu \rvert$ against the bounds, and the sample means themselves. */
export interface ConcentrationStudy {
  /** The deviations $t$. */
  readonly t: Float64Array
  /** The Monte Carlo estimate of $P(\lvert \bar{X}_n - \mu \rvert \ge t)$ at each $t$. */
  readonly empirical: Float64Array
  /** Chebyshev's bound at each $t$. */
  readonly chebyshev: Float64Array
  /** Hoeffding's bound at each $t$. */
  readonly hoeffding: Float64Array
  /** Bernstein's bound at each $t$. */
  readonly bernstein: Float64Array
  /** The Chernoff (relative entropy) bound at each $t$. */
  readonly chernoff: Float64Array
  /** The simulated sample means, one per trial. */
  readonly means: Float64Array
  /** The law's mean $\mu$. */
  readonly mean: number
  /** The law's variance $\sigma^2$. */
  readonly variance: number
}

/**
 * Monte Carlo tails of a bounded law's sample mean against Chebyshev, Hoeffding, Bernstein and Chernoff. An unbounded
 * law throws `DomainError`. A sample mean at exactly $t$ from $\mu$ counts in the tail, up to a tolerance of
 * $10^{-12}$ that keeps the lattice values of a Bernoulli mean.
 *
 * @param s The stream; trial $r$ draws from `child(s, 'trial', r)`.
 * @param options The law, sizes and deviations (see `ConcentrationOptions`).
 * @returns The empirical tail and the four bounds at each deviation (see `ConcentrationStudy`).
 *
 * @example Hoeffding's bound against the empirical tail of a Bernoulli mean
 * const r = concentrationStudy(stream(0), { n: 50, trials: 2000, thresholds: [0.05, 0.1, 0.15, 0.2] })
 * print('t        ', r.t)
 * print('empirical', r.empirical)
 * print('hoeffding', r.hoeffding)
 * print('chernoff ', r.chernoff)
 */
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
  /** The number of balls $n$ thrown (default 100). */
  balls?: number
  /** The number of bins $m$ (default 100). */
  bins?: number
  /** Monte Carlo trials (default 5000). */
  trials?: number
  /** Deviations $t$ at which tails are read (default 50 even points on $(0, 0.25]$). */
  thresholds?: readonly number[]
}

/**
 * The fraction of empty bins after `balls` uniform throws into `bins` bins, its tail and McDiarmid's bound
 * $2 \exp(-2 t^2 m^2 / n)$ (capped at 1).
 *
 * @param s The stream; trial $r$ draws from `child(s, 'trial', r)`.
 * @param options The numbers of balls and bins, trials and deviations (see `McdiarmidOptions`).
 * @returns The deviations `t`, the `empirical` tail $P(\lvert f - \expect f \rvert \ge t)$ and the `bound` at each,
 *   the simulated `values` of $f$, and the exact `expected` value $(1 - 1/m)^n$.
 *
 * @example The fraction of empty bins concentrates far inside McDiarmid's bound
 * const r = mcdiarmidStudy(stream(0), { trials: 1000, thresholds: [0.02, 0.05, 0.1] })
 * print('E f =', r.expected)
 * print('empirical', r.empirical)
 * print('bound    ', r.bound)
 */
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

/**
 * Running means $\bar{X}_1, \dots, \bar{X}_n$ of `paths` independent sequences, the law of large numbers in action.
 *
 * @param s The stream; path $r$ draws from `child(s, 'path', r)`.
 * @param law The law of each draw.
 * @param n The length of each sequence.
 * @param paths The number of sequences.
 * @param p The Bernoulli parameter; ignored by the other laws.
 * @returns The running means, row-major $[\text{paths}, n]$: entry `r * n + i` is the mean of the first $i + 1$ draws
 *   of path $r$.
 *
 * @example Three paths settle on the mean 1 of an exponential law
 * const n = 2000
 * const m = runningMeans(stream(0), 'exponential', n, 3)
 * for (let r = 0; r < 3; r++) print('path', r, ' after 10:', m[r * n + 9], ' after 2000:', m[r * n + n - 1])
 */
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
 * Standardised sums $\sqrt{n} (\bar{X}_n - \mu)/\sigma$ over `trials` samples. For the Pareto law (infinite variance)
 * the sums are centred and scaled as its stable limit needs,
 * $(S_n - n\mu)/n^{1/\alpha} = n^{1 - 1/\alpha} (\bar{X}_n - \mu) = n^{1/3} (\bar{X}_n - \mu)$, so the histogram
 * stays finite but skewed instead of normal.
 *
 * @param s The stream; trial $r$ draws from `child(s, 'trial', r)`.
 * @param law The law of each draw.
 * @param n The draws per sum.
 * @param trials The number of sums.
 * @param p The Bernoulli parameter; ignored by the other laws.
 * @returns The `trials` standardised sums: approximately $\Gauss(0, 1)$ for a law of finite variance.
 *
 * @example Exponential sums look normal; Pareto sums stay skewed
 * const summary = (z) => {
 *   const m = z.reduce((a, b) => a + b, 0) / z.length
 *   const sd = Math.sqrt(z.reduce((a, b) => a + (b - m) ** 2, 0) / z.length)
 *   const below = z.filter((v) => v < m).length / z.length
 *   return { mean: m, sd, fractionBelowMean: below }
 * }
 * print('exponential:', summary(standardisedSums(stream(0), 'exponential', 50, 2000)))
 * print('pareto:', summary(standardisedSums(stream(0), 'pareto', 50, 2000)))
 */
export function standardisedSums(s: Stream, law: SummandLaw, n: number, trials: number, p = 0.3): Float64Array {
  const { mean, variance } = lawMoments(law, p)
  const means = sampleMeans(s, law, n, trials, p)
  const scale = Number.isFinite(variance) ? Math.sqrt(n / variance) : Math.pow(n, 1 - 1 / 1.5)
  return Float64Array.from(means, (m) => (m - mean) * scale)
}
