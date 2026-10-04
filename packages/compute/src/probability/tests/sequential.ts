/**
 * Sequential tests over a data stream, each a step-through `Algorithm` whose step absorbs one observation, so a
 * player can show the statistic and its boundaries as the data arrive:
 *
 * - `sprt`: Wald's sequential probability ratio test of a simple null against a simple alternative;
 * - `msprt`: the mixture SPRT of a normal mean (Robbins, 1970; Johari et al., 2017), with its always-valid p-value;
 * - `confidenceSequence`: the normal-mixture confidence sequence for a mean and the e-value of a null mean;
 * - `groupSequentialTest`: a z-test examined at planned looks against group-sequential boundaries (see
 *   `groupSequentialBoundaries` for boundaries from an alpha-spending function);
 * - `cusum`: Page's cumulative-sum control chart, with `cusumAverageRunLength` for its average run length.
 *
 * Every state carries `t` (observations absorbed) and `terminated` (a decision was reached, or the data ran out).
 */

import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, tensor, toFlat, type VectorLike } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import type { Univariate } from 'aifn-compute/probability/distributions'

const Phi = (x: number) => normalCdf(x) as number
const PhiInv = (p: number) => normalQuantile(p) as number

function stream(data: VectorLike, where: string): Float64Array {
  const v = dense.toF64(data, where)
  for (const a of v) if (!Number.isFinite(a)) throw new DomainError(where, `${where}: the data have a non-finite value`)
  return v
}

function rate(x: number, name: string, where: string): void {
  if (!(x > 0 && x < 1)) throw new DomainError(where, `${where}: ${name} must be in (0, 1)`)
}

// ── Wald's SPRT ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Wald's (1945) boundaries for the log-likelihood ratio of a sequential test with type-I error α and type-II error β:
 * accept H₀ at log(β/(1 − α)) or below, reject it at log((1 − β)/α) or above. They hold the errors at about α and β
 * (at most α/(1 − β) and β/(1 − α), ignoring the overshoot).
 */
export function waldBoundaries(alpha: number, beta: number): { lower: number; upper: number } {
  rate(alpha, 'α', 'waldBoundaries')
  rate(beta, 'β', 'waldBoundaries')
  return { lower: Math.log(beta / (1 - alpha)), upper: Math.log((1 - beta) / alpha) }
}

/** The decision of a sequential test so far. */
export type Decision = 'continue' | 'accept-null' | 'reject-null'

/** A state of `sprt`. */
export type SprtState = Status & {
  /** The log-likelihood ratio Σ log p₁(xᵢ)/p₀(xᵢ) of the observations so far. */
  llr: number
  lower: number
  upper: number
  decision: Decision
  terminated: boolean
}

/**
 * Wald's sequential probability ratio test (Wald, 1945) of H₀: x ~ `h0` against H₁: x ~ `h1` (any two univariate
 * distributions with the same support), over the observations `data`: each step adds log p₁(x) − log p₀(x) to the
 * log-likelihood ratio and stops at Wald's boundaries for errors α (default 0.05) and β (default 0.2). Among tests with
 * the same error rates it has the smallest expected sample size under both hypotheses (Wald and Wolfowitz, 1948).
 */
export function sprt(
  data: VectorLike,
  { h0, h1, alpha = 0.05, beta = 0.2 }: { h0: Univariate; h1: Univariate; alpha?: number; beta?: number },
): Algorithm<void, SprtState> {
  const x = stream(data, 'sprt')
  const { lower, upper } = waldBoundaries(alpha, beta)
  return {
    name: 'sprt',
    init: () => ({ t: 0, llr: 0, lower, upper, decision: 'continue', terminated: x.length === 0 }),
    step: (s) => {
      const v = x[s.t]
      const llr = s.llr + (h1.logProb(v) as number) - (h0.logProb(v) as number)
      const decision: Decision = llr >= upper ? 'reject-null' : llr <= lower ? 'accept-null' : 'continue'
      return { ...s, t: s.t + 1, llr, decision, terminated: decision !== 'continue' || s.t + 1 >= x.length }
    },
  }
}

// ── The normal mixture: mSPRT, e-values and confidence sequences ────────────────────────────────────────────────────

/** The parameters of the normal mixture: the known standard deviation σ and the mixing standard deviation τ. */
export type NormalMixture = { sigma: number; tau: number }

function mixture({ sigma, tau }: NormalMixture, where: string): void {
  if (!(sigma > 0)) throw new DomainError(where, `${where}: σ must be positive`)
  if (!(tau > 0)) throw new DomainError(where, `${where}: τ must be positive`)
}

/**
 * The log of the normal-mixture likelihood ratio after n observations with mean x̄ from N(θ, σ²), against H₀: θ = θ₀,
 * mixing the alternative over θ ~ N(θ₀, τ²) (Robbins, 1970):
 * log Λₙ = ½ log(σ²/(σ² + nτ²)) + n²τ²(x̄ − θ₀)²/(2σ²(σ² + nτ²)). Under H₀, Λₙ is a nonnegative martingale with mean 1,
 * so Λₙ is an e-value at every n and P(supₙ Λₙ ≥ 1/α) ≤ α (Ville's inequality).
 */
export function normalMixtureLogLikelihoodRatio(n: number, mean: number, theta0: number, m: NormalMixture): number {
  mixture(m, 'normalMixtureLogLikelihoodRatio')
  if (n === 0) return 0
  const s2 = m.sigma ** 2
  const v = s2 + n * m.tau ** 2
  return 0.5 * Math.log(s2 / v) + (n * n * m.tau ** 2 * (mean - theta0) ** 2) / (2 * s2 * v)
}

/**
 * The half-width of the normal-mixture confidence sequence after n observations: the θ₀ with log Λₙ(θ₀) < log(1/α) are
 * |x̄ − θ₀| < √(2σ²(σ² + nτ²)/(n²τ²) · (log(1/α) + ½ log((σ² + nτ²)/σ²))). The intervals x̄ ± radius cover θ at every n
 * simultaneously with probability at least 1 − α. The radius shrinks like √(log n / n), so peeking costs nothing.
 */
export function normalMixtureRadius(n: number, alpha: number, m: NormalMixture): number {
  mixture(m, 'normalMixtureRadius')
  rate(alpha, 'α', 'normalMixtureRadius')
  if (n === 0) return Infinity
  const s2 = m.sigma ** 2
  const v = s2 + n * m.tau ** 2
  return Math.sqrt(((2 * s2 * v) / (n * n * m.tau ** 2)) * (Math.log(1 / alpha) + 0.5 * Math.log(v / s2)))
}

/** A state of `msprt`. */
export type MsprtState = Status & {
  n: number
  mean: number
  /** log Λₙ, the log mixture likelihood ratio (the log e-value). */
  logLikelihoodRatio: number
  /** The always-valid p-value minₖ≤ₙ min(1, 1/Λₖ). */
  pValue: number
  rejected: boolean
  terminated: boolean
}

/**
 * The mixture sequential probability ratio test (Robbins, 1970; Johari, Koomen, Pekelis and Walsh, 2017) of
 * H₀: θ = `theta0` (default 0) for observations from N(θ, σ²) with σ known: after each observation the mixture
 * likelihood ratio Λₙ (`normalMixtureLogLikelihoodRatio`) is compared with 1/α, and the test stops when it crosses. The
 * running p-value min(1, 1/maxₖ Λₖ) is valid at any stopping time. For an A/B test, feed the paired differences (or the
 * per-observation differences of means) with their σ.
 */
export function msprt(
  data: VectorLike,
  { theta0 = 0, alpha = 0.05, ...m }: NormalMixture & { theta0?: number; alpha?: number },
): Algorithm<void, MsprtState> {
  const x = stream(data, 'msprt')
  mixture(m, 'msprt')
  rate(alpha, 'α', 'msprt')
  return {
    name: 'msprt',
    init: () => ({
      t: 0,
      n: 0,
      mean: 0,
      logLikelihoodRatio: 0,
      pValue: 1,
      rejected: false,
      terminated: x.length === 0,
    }),
    step: (s) => {
      const n = s.n + 1
      const mean = s.mean + (x[s.t] - s.mean) / n
      const llr = normalMixtureLogLikelihoodRatio(n, mean, theta0, m)
      const pValue = Math.min(s.pValue, Math.min(1, Math.exp(-llr)))
      const rejected = pValue <= alpha
      return { t: s.t + 1, n, mean, logLikelihoodRatio: llr, pValue, rejected, terminated: rejected || n >= x.length }
    },
  }
}

/** A state of `confidenceSequence`. */
export type ConfidenceSequenceState = Status & {
  n: number
  mean: number
  radius: number
  /** The running intersection of the intervals so far (each is valid, so their intersection is too). */
  lower: number
  upper: number
  /** log Λₙ(μ₀): the log e-value against the null mean μ₀. */
  logEValue: number
  /** True once the e-value has reached 1/α (μ₀ left the sequence). */
  rejected: boolean
  terminated: boolean
}

/**
 * The normal-mixture confidence sequence for the mean of N(μ, σ²) observations (Robbins, 1970; Howard, Ramdas,
 * McAuliffe and Sekhon, 2021): after n observations, x̄ₙ ± `normalMixtureRadius(n, α, { σ, τ })`, intersected over
 * time. With `mu0` (default 0), the state also carries the e-value Λₙ(μ₀) of that null mean; μ₀ leaves the sequence
 * exactly when the e-value reaches 1/α. The sequence runs to the end of the data.
 */
export function confidenceSequence(
  data: VectorLike,
  { mu0 = 0, alpha = 0.05, ...m }: NormalMixture & { mu0?: number; alpha?: number },
): Algorithm<void, ConfidenceSequenceState> {
  const x = stream(data, 'confidenceSequence')
  mixture(m, 'confidenceSequence')
  rate(alpha, 'α', 'confidenceSequence')
  return {
    name: 'confidenceSequence',
    init: () => ({
      t: 0,
      n: 0,
      mean: 0,
      radius: Infinity,
      lower: -Infinity,
      upper: Infinity,
      logEValue: 0,
      rejected: false,
      terminated: x.length === 0,
    }),
    step: (s) => {
      const n = s.n + 1
      const mean = s.mean + (x[s.t] - s.mean) / n
      const radius = normalMixtureRadius(n, alpha, m)
      const logEValue = normalMixtureLogLikelihoodRatio(n, mean, mu0, m)
      return {
        t: s.t + 1,
        n,
        mean,
        radius,
        lower: Math.max(s.lower, mean - radius),
        upper: Math.min(s.upper, mean + radius),
        logEValue,
        rejected: s.rejected || logEValue >= Math.log(1 / alpha),
        terminated: n >= x.length,
      }
    },
  }
}

// ── Group-sequential designs ─────────────────────────────────────────────────────────────────────────────────────────

/**
 * An alpha-spending function α*(t), the type-I error spent by information fraction t ∈ [0, 1] (Lan and DeMets, 1983):
 * `obrien-fleming`, 2 − 2Φ(z_{1−α/2}/√t), which spends almost nothing early; `pocock`, α log(1 + (e − 1)t), close to
 * Pocock's constant boundary; `power`, α t^ρ (Kim and DeMets, 1987).
 */
export type Spending = { family: 'obrien-fleming' | 'pocock' | 'power'; rho?: number }

/** α*(t) for a spending function, total α (one- or two-sided as the design is). */
export function spentAlpha(spending: Spending, alpha: number, t: number): number {
  rate(alpha, 'α', 'spentAlpha')
  if (t <= 0) return 0
  const u = Math.min(1, t)
  if (spending.family === 'obrien-fleming') return 2 - 2 * Phi(PhiInv(1 - alpha / 2) / Math.sqrt(u))
  if (spending.family === 'pocock') return alpha * Math.log(1 + (Math.E - 1) * u)
  return alpha * u ** (spending.rho ?? 1)
}

/** Group-sequential boundaries on the z scale at each look, with the type-I error each look spends. */
export type GroupSequentialBoundaries = {
  readonly kind: 'group-sequential-boundaries'
  /** Information fractions t₁ < … < t_K = 1. */
  readonly information: Float64Array
  /** Critical values c_k: reject at look k when Z_k ≥ c_k (one-sided) or |Z_k| ≥ c_k (two-sided). Infinity: no stop. */
  readonly z: Float64Array
  /** The probability under H₀ of first crossing at each look; they sum to α. */
  readonly crossing: Float64Array
  readonly alpha: number
  readonly sides: 1 | 2
}

const SQRT_2PI = Math.sqrt(2 * Math.PI)
const phi = (x: number) => Math.exp(-0.5 * x * x) / SQRT_2PI

/**
 * The recursive numerical integration of Armitage, McPherson and Rowe (1969) for the score process S_k = Z_k √t_k, a
 * Brownian motion in information time: `density` holds the sub-density of S_k on a Simpson grid over the continuation
 * region (paths not yet stopped). `crossing(c)` is the probability of first crossing at the next look with critical
 * value c; `advance(c)` moves the density to that look.
 */
class ScoreDensity {
  grid: Float64Array = new Float64Array(0)
  density: Float64Array = new Float64Array(0)
  weights: Float64Array = new Float64Array(0)
  readonly sides: 1 | 2
  readonly points: number
  constructor(sides: 1 | 2, points: number) {
    this.sides = sides
    this.points = points
  }

  /** The continuation region's grid at information t with critical value c (truncated at ±10 sd). */
  private region(t: number, c: number): Float64Array {
    const sd = Math.sqrt(t)
    const hi = Math.min(c, 10) * sd
    const lo = this.sides === 2 ? -hi : -10 * sd
    const n = this.points
    return Float64Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
  }

  private simpson(grid: Float64Array): Float64Array {
    const n = grid.length
    const h = (grid[n - 1] - grid[0]) / (n - 1)
    return Float64Array.from({ length: n }, (_, i) => ((i === 0 || i === n - 1 ? 1 : i % 2 === 1 ? 4 : 2) * h) / 3)
  }

  /** P(first crossing at the look at information t with value c), from the previous look at information tPrev. */
  crossing(tPrev: number, t: number, c: number): number {
    const tail = (u: number, sd: number) =>
      1 - Phi((c * Math.sqrt(t) - u) / sd) + (this.sides === 2 ? Phi((-c * Math.sqrt(t) - u) / sd) : 0)
    if (tPrev === 0) return tail(0, Math.sqrt(t))
    const sd = Math.sqrt(t - tPrev)
    let p = 0
    for (let i = 0; i < this.grid.length; i++) p += this.weights[i] * this.density[i] * tail(this.grid[i], sd)
    return p
  }

  /** Move the sub-density to the look at information t, keeping paths inside (−c√t, c√t) (or below c√t). */
  advance(tPrev: number, t: number, c: number): void {
    const grid = this.region(t, c)
    const out = new Float64Array(grid.length)
    if (tPrev === 0) for (let i = 0; i < grid.length; i++) out[i] = phi(grid[i] / Math.sqrt(t)) / Math.sqrt(t)
    else {
      const sd = Math.sqrt(t - tPrev)
      for (let i = 0; i < grid.length; i++) {
        let s = 0
        for (let j = 0; j < this.grid.length; j++)
          s += this.weights[j] * this.density[j] * phi((grid[i] - this.grid[j]) / sd)
        out[i] = s / sd
      }
    }
    this.grid = grid
    this.density = out
    this.weights = this.simpson(grid)
  }
}

/** Bisection for the c with crossing(c) = target (crossing decreases in c). */
function criticalValue(f: (c: number) => number, target: number): number {
  if (!(target > 1e-15)) return Infinity
  let lo = 0
  let hi = 40
  for (let i = 0; i < 100 && hi - lo > 1e-10; i++) {
    const mid = (lo + hi) / 2
    if (f(mid) > target) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

function fractions(information: VectorLike | number, where: string): Float64Array {
  const raw =
    typeof information === 'number'
      ? Float64Array.from({ length: information }, (_, k) => k + 1)
      : dense.toF64(information, where)
  if (raw.length === 0) throw new DomainError(where, `${where}: needs at least one look`)
  for (let k = 0; k < raw.length; k++)
    if (!(raw[k] > (k ? raw[k - 1] : 0))) throw new DomainError(where, `${where}: information must increase from > 0`)
  const last = raw[raw.length - 1]
  return raw.map((v) => v / last)
}

/**
 * Group-sequential boundaries from an alpha-spending function (Lan and DeMets, 1983): at each look k, the critical
 * value c_k is chosen so that the probability under H₀ of first crossing at look k is α*(t_k) − α*(t_{k−1}).
 * `information` is the number of equally spaced looks or the information (or sample size) at each look; `sides` 2
 * (default) gives symmetric two-sided boundaries |Z| ≥ c_k with total α, spending α/2 on each side by the one-sided
 * function (the convention of gsDesign and ldbounds), and 1 an upper boundary spending α. Crossing probabilities
 * are computed by recursive numerical integration (Armitage, McPherson and Rowe, 1969) on a Simpson grid of `points`
 * (default 401) per look.
 */
export function groupSequentialBoundaries(
  information: VectorLike | number,
  {
    alpha = 0.05,
    spending = { family: 'obrien-fleming' },
    sides = 2,
    points = 401,
  }: { alpha?: number; spending?: Spending; sides?: 1 | 2; points?: number } = {},
): GroupSequentialBoundaries {
  const t = fractions(information, 'groupSequentialBoundaries')
  rate(alpha, 'α', 'groupSequentialBoundaries')
  const K = t.length
  const z = new Float64Array(K)
  const crossing = new Float64Array(K)
  const S = new ScoreDensity(sides, points | 1)
  let spent = 0
  for (let k = 0; k < K; k++) {
    const tPrev = k ? t[k - 1] : 0
    // A two-sided design spends α/2 on each side with the one-sided function (Lan and DeMets, as gsDesign and ldbounds).
    const target = (sides === 2 ? 2 * spentAlpha(spending, alpha / 2, t[k]) : spentAlpha(spending, alpha, t[k])) - spent
    z[k] = criticalValue((c) => S.crossing(tPrev, t[k], c), target)
    crossing[k] = Number.isFinite(z[k]) ? S.crossing(tPrev, t[k], z[k]) : 0
    spent += crossing[k]
    S.advance(tPrev, t[k], z[k])
  }
  return { kind: 'group-sequential-boundaries', information: t, z, crossing, alpha, sides }
}

/**
 * The classical group-sequential boundaries with a fixed shape: `pocock` (Pocock, 1977), one critical value c at every
 * look; `obrien-fleming` (O'Brien and Fleming, 1979), c/√t_k, wide early and near the fixed-sample value at the end.
 * c is found so that the total probability of crossing under H₀ is α.
 */
export function constantBoundaries(
  shape: 'pocock' | 'obrien-fleming',
  information: VectorLike | number,
  { alpha = 0.05, sides = 2, points = 401 }: { alpha?: number; sides?: 1 | 2; points?: number } = {},
): GroupSequentialBoundaries {
  const t = fractions(information, 'constantBoundaries')
  rate(alpha, 'α', 'constantBoundaries')
  const at = (c: number, k: number) => (shape === 'pocock' ? c : c / Math.sqrt(t[k]))
  const run = (c: number) => {
    const S = new ScoreDensity(sides, points | 1)
    const crossing = new Float64Array(t.length)
    for (let k = 0; k < t.length; k++) {
      const tPrev = k ? t[k - 1] : 0
      crossing[k] = S.crossing(tPrev, t[k], at(c, k))
      S.advance(tPrev, t[k], at(c, k))
    }
    return crossing
  }
  const c = criticalValue((v) => run(v).reduce((a, b) => a + b, 0), alpha)
  return {
    kind: 'group-sequential-boundaries',
    information: t,
    z: Float64Array.from(t, (_, k) => at(c, k)),
    crossing: run(c),
    alpha,
    sides,
  }
}

/** A state of `groupSequentialTest`. */
export type GroupSequentialState = Status & {
  n: number
  mean: number
  /** The z statistic at the most recent look (NaN before the first). */
  z: number
  /** The index of the most recent look (−1 before the first). */
  look: number
  rejected: boolean
  terminated: boolean
}

/**
 * A group-sequential z-test of H₀: μ = `mu0` (default 0) for N(μ, σ²) observations with σ known: each step absorbs
 * one observation; at the planned look sizes `looks` (cumulative counts, one per boundary) the statistic
 * Z = (x̄ − μ₀)√n/σ is compared with the boundary, and the test stops at the first crossing (|Z| ≥ c_k two-sided, Z ≥ c_k
 * one-sided) or after the last look.
 */
export function groupSequentialTest(
  data: VectorLike,
  {
    looks,
    boundaries,
    sigma,
    mu0 = 0,
  }: { looks: readonly number[]; boundaries: GroupSequentialBoundaries; sigma: number; mu0?: number },
): Algorithm<void, GroupSequentialState> {
  const x = stream(data, 'groupSequentialTest')
  if (looks.length !== boundaries.z.length)
    throw new DomainError('groupSequentialTest', 'groupSequentialTest: one look size per boundary')
  if (!(sigma > 0)) throw new DomainError('groupSequentialTest', 'groupSequentialTest: σ must be positive')
  const last = Math.min(looks[looks.length - 1], x.length)
  return {
    name: 'groupSequentialTest',
    init: () => ({ t: 0, n: 0, mean: 0, z: NaN, look: -1, rejected: false, terminated: last === 0 }),
    step: (s) => {
      const n = s.n + 1
      const mean = s.mean + (x[s.t] - s.mean) / n
      const k = looks.indexOf(n)
      if (k < 0) return { ...s, t: s.t + 1, n, mean, terminated: n >= last }
      const z = ((mean - mu0) * Math.sqrt(n)) / sigma
      const c = boundaries.z[k]
      const rejected = boundaries.sides === 2 ? Math.abs(z) >= c : z >= c
      return { t: s.t + 1, n, mean, z, look: k, rejected, terminated: rejected || n >= last }
    },
  }
}

// ── CUSUM ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A state of `cusum`. */
export type CusumState = Status & {
  /** C⁺, the upper cumulative sum (detects an increase). */
  upper: number
  /** C⁻, the lower cumulative sum (detects a decrease). */
  lower: number
  /** True when this step signalled. */
  alarm: boolean
  /** The number of signals so far. */
  alarms: number
  /** The step of the first signal (−1 before it). */
  firstAlarm: number
  terminated: boolean
}

/**
 * Page's (1954) tabular CUSUM for a shift in the mean of N(`target`, σ²) observations, in standardised units
 * zₜ = (xₜ − target)/σ: C⁺ₜ = max(0, C⁺ₜ₋₁ + zₜ − k) and C⁻ₜ = max(0, C⁻ₜ₋₁ − zₜ − k), signalling when either exceeds h.
 * k (default ½, half the shift to detect) is the reference value and h (default 5) the decision interval. After a
 * signal both sums restart at 0 unless `reset` is false. `sides` 'upper' or 'lower' watches one direction only.
 */
export function cusum(
  data: VectorLike,
  {
    target = 0,
    sigma = 1,
    k = 0.5,
    h = 5,
    sides = 'both',
    reset = true,
  }: {
    target?: number
    sigma?: number
    k?: number
    h?: number
    sides?: 'both' | 'upper' | 'lower'
    reset?: boolean
  } = {},
): Algorithm<void, CusumState> {
  const x = stream(data, 'cusum')
  if (!(sigma > 0) || !(h > 0) || !(k >= 0)) throw new DomainError('cusum', 'cusum: needs σ > 0, h > 0 and k ≥ 0')
  return {
    name: 'cusum',
    init: () => ({ t: 0, upper: 0, lower: 0, alarm: false, alarms: 0, firstAlarm: -1, terminated: x.length === 0 }),
    step: (s) => {
      const zt = (x[s.t] - target) / sigma
      let upper = sides === 'lower' ? 0 : Math.max(0, s.upper + zt - k)
      let lower = sides === 'upper' ? 0 : Math.max(0, s.lower - zt - k)
      const alarm = upper > h || lower > h
      const t = s.t + 1
      const out = {
        t,
        upper,
        lower,
        alarm,
        alarms: s.alarms + (alarm ? 1 : 0),
        firstAlarm: s.firstAlarm < 0 && alarm ? t : s.firstAlarm,
        terminated: t >= x.length,
      }
      if (alarm && reset) {
        upper = 0
        lower = 0
        return { ...out, upper, lower }
      }
      return out
    },
  }
}

/**
 * The average run length of a one-sided upper CUSUM (k, h in standard-deviation units) on normal data whose mean has
 * shifted by `shift` standard deviations (0: the in-control ARL₀, the mean time between false alarms).
 *
 * - `markov-chain` (default; Brook and Evans, 1972): [0, h] is cut into `states` cells of width w = 2h/(2·states − 1),
 *   the chart moves between cells with normal probabilities, and the ARL from 0 is the first entry of (I − R)⁻¹𝟏 for
 *   the transient block R. Converges to the exact ARL as the cells shrink.
 * - `siegmund` (Siegmund, 1985): (e^{−2Δb} + 2Δb − 1)/(2Δ²) with Δ = shift − k and b = h + 1.166 (b² when Δ = 0).
 *
 * `sides: 'both'` combines the upper and lower charts by 1/ARL = 1/ARL⁺ + 1/ARL⁻, a close approximation for the
 * two-sided chart (the two sums rarely are both positive).
 */
export function cusumAverageRunLength({
  k = 0.5,
  h = 5,
  shift = 0,
  method = 'markov-chain',
  states = 100,
  sides = 'upper',
}: {
  k?: number
  h?: number
  shift?: number
  method?: 'markov-chain' | 'siegmund'
  states?: number
  sides?: 'upper' | 'both'
} = {}): number {
  if (!(h > 0) || !(k >= 0)) throw new DomainError('cusumAverageRunLength', 'needs h > 0 and k ≥ 0')
  const one = (delta: number) => {
    if (method === 'siegmund') {
      const d = delta - k
      const b = h + 1.166
      return Math.abs(d) < 1e-12 ? b * b : (Math.exp(-2 * d * b) + 2 * d * b - 1) / (2 * d * d)
    }
    const m = states
    const w = (2 * h) / (2 * m - 1)
    const A = Array.from({ length: m }, (_, i) =>
      Array.from({ length: m }, (_, j) => {
        // From cell i (centre i·w), the next value is i·w + z − k with z ~ N(δ, 1).
        const p =
          j === 0
            ? Phi(w / 2 - i * w + k - delta)
            : Phi((j + 0.5) * w - i * w + k - delta) - Phi((j - 0.5) * w - i * w + k - delta)
        return (i === j ? 1 : 0) - p
      }),
    )
    return toFlat(solve(tensor(A), tensor(new Array<number>(m).fill(1))))[0]
  }
  if (sides === 'upper') return one(shift)
  return 1 / (1 / one(shift) + 1 / one(-shift))
}
