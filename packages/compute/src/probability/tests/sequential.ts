/**
 * Sequential tests over a data stream, each a step-through `Algorithm` whose step absorbs one observation, so a
 * player can show the statistic and its boundaries as the data arrive:
 *
 * - `sprt`: Wald's sequential probability ratio test of a simple null against a simple alternative;
 * - `msprt`: the mixture SPRT of a normal mean (Robbins, 1970; Johari et al., 2017), with its always-valid p-value;
 * - `confidenceSequence`: the normal-mixture confidence sequence for a mean and the e-value of a null mean;
 * - `groupSequentialTest`: a $z$-test examined at planned looks against group-sequential boundaries (see
 *   `groupSequentialBoundaries` for boundaries from an alpha-spending function);
 * - `cusum`: Page's cumulative-sum control chart, with `cusumAverageRunLength` for its average run length.
 *
 * Every state carries `t` (observations absorbed) and `terminated` (a decision was reached, or the data ran out), so
 * `run(test, undefined, steps)` stops at the decision. The data must be finite (a `DomainError` otherwise), and
 * error rates must be in $(0, 1)$. Unlike a fixed-sample test, each of these keeps its type-I error at $\alpha$
 * however often the data are looked at.
 */

import type { Algorithm, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, tensor, toFlat, type VectorLike } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'
import { normalCdf, normalQuantile } from 'aifn-compute/numerics/special'
import type { Univariate } from 'aifn-compute/probability/distributions'

/**
 * The standard normal cdf $\Phi$.
 *
 * @param x The point.
 * @returns $\Phi(x)$.
 */
const Phi = (x: number) => normalCdf(x) as number
/**
 * The standard normal quantile $\Phi^{-1}$.
 *
 * @param p A probability in $(0, 1)$.
 * @returns $\Phi^{-1}(p)$.
 */
const PhiInv = (p: number) => normalQuantile(p) as number

/**
 * The observations as a Float64Array, checked to be finite (a `DomainError` otherwise). An empty stream is allowed:
 * the test then starts terminated.
 *
 * @param data The observations, in arrival order.
 * @param where The caller's name, for error messages.
 * @returns A new array of them.
 */
function stream(data: VectorLike, where: string): Float64Array {
  const v = dense.toF64(data, where)
  for (const a of v) if (!Number.isFinite(a)) throw new DomainError(where, `${where}: the data have a non-finite value`)
  return v
}

/**
 * Check an error rate: throws `DomainError` unless it is in $(0, 1)$.
 *
 * @param x The rate.
 * @param name Its symbol in the message, such as $\alpha$.
 * @param where The caller's name, for the error message.
 */
function rate(x: number, name: string, where: string): void {
  if (!(x > 0 && x < 1)) throw new DomainError(where, `${where}: ${name} must be in (0, 1)`)
}

// ── Wald's SPRT ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Wald's (1945) boundaries for the log-likelihood ratio of a sequential test with type-I error $\alpha$ and type-II
 * error $\beta$: accept $H_0$ at $\log(\beta/(1 - \alpha))$ or below, reject it at $\log((1 - \beta)/\alpha)$ or
 * above. They hold the errors at about $\alpha$ and $\beta$ (at most $\alpha/(1 - \beta)$ and $\beta/(1 - \alpha)$,
 * ignoring the overshoot).
 *
 * @param alpha The type-I error $\alpha$, in $(0, 1)$.
 * @param beta The type-II error $\beta$, in $(0, 1)$.
 * @returns The `lower` (accept) and `upper` (reject) boundaries of the log-likelihood ratio.
 *
 * @example The boundaries for 5% and 20% errors
 * print(waldBoundaries(0.05, 0.2))
 * print('as likelihood ratios:', Math.exp(waldBoundaries(0.05, 0.2).lower), Math.exp(waldBoundaries(0.05, 0.2).upper))
 */
export function waldBoundaries(alpha: number, beta: number): { lower: number; upper: number } {
  rate(alpha, 'α', 'waldBoundaries')
  rate(beta, 'β', 'waldBoundaries')
  return { lower: Math.log(beta / (1 - alpha)), upper: Math.log((1 - beta) / alpha) }
}

/**
 * The decision of a sequential test so far: `continue` sampling, `accept-null` or `reject-null`.
 */
export type Decision = 'continue' | 'accept-null' | 'reject-null'

/** A state of `sprt`. */
export type SprtState = Status & {
  /** The log-likelihood ratio $\sum_i \log p_1(x_i)/p_0(x_i)$ of the observations so far. */
  llr: number
  /** Wald's lower boundary: accept $H_0$ at or below it. */
  lower: number
  /** Wald's upper boundary: reject $H_0$ at or above it. */
  upper: number
  /** The decision so far. */
  decision: Decision
  /** True once a boundary is crossed or the data run out. */
  terminated: boolean
}

/**
 * Wald's sequential probability ratio test (Wald, 1945) of $H_0: x \sim p_0$ (`h0`) against $H_1: x \sim p_1$ (`h1`)
 * (any two univariate distributions with the same support), over the observations `data`: each step adds
 * $\log p_1(x) - \log p_0(x)$ to the log-likelihood ratio and stops at Wald's boundaries (`waldBoundaries`) for errors
 * $\alpha$ (default 0.05) and $\beta$ (default 0.2). Among tests with the same error rates it has the smallest expected
 * sample size under both hypotheses (Wald and Wolfowitz, 1948).
 *
 * @param data The observations, in arrival order.
 * @param options The test.
 * @param options.h0 The distribution of an observation under the null.
 * @param options.h1 The distribution of an observation under the alternative.
 * @param options.alpha The type-I error $\alpha$.
 * @param options.beta The type-II error $\beta$.
 * @returns The test as an algorithm: run it with `run(test, undefined, steps)`.
 *
 * @example A coin with P(heads) = 0.5 against 0.8, on a biased coin and a fair one
 * // The Bernoulli laws: the null laws of the exact binomial test of one toss.
 * const h0 = binomialTest(0, 1, { p: 0.5 }).null
 * const h1 = binomialTest(0, 1, { p: 0.8 }).null
 * const biased = toArray(bernoulli(stream(3), 0.8, { shape: [100] }))
 * const s = run(sprt(biased, { h0, h1 }), undefined, 100)
 * print('biased coin:', s.decision, 'after', s.t, 'tosses, llr =', s.llr)
 * const fair = toArray(bernoulli(stream(4), 0.5, { shape: [100] }))
 * const f = run(sprt(fair, { h0, h1 }), undefined, 100)
 * print('fair coin:', f.decision, 'after', f.t, 'tosses, llr =', f.llr)
 * print('boundaries:', [f.lower, f.upper])
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

/**
 * The parameters of the normal mixture: `sigma`, the known standard deviation $\sigma$ of an observation, and `tau`,
 * the mixing standard deviation $\tau$ of the alternative means (of the order of the effect expected).
 */
export type NormalMixture = { sigma: number; tau: number }

/**
 * Check a normal mixture: throws `DomainError` unless $\sigma$ and $\tau$ are positive.
 *
 * @param options The mixture.
 * @param options.sigma The observations' standard deviation $\sigma$.
 * @param options.tau The mixing standard deviation $\tau$.
 * @param where The caller's name, for error messages.
 */
function mixture({ sigma, tau }: NormalMixture, where: string): void {
  if (!(sigma > 0)) throw new DomainError(where, `${where}: σ must be positive`)
  if (!(tau > 0)) throw new DomainError(where, `${where}: τ must be positive`)
}

/**
 * The log of the normal-mixture likelihood ratio after $n$ observations with mean $\bar x$ from
 * $\Gauss(\theta, \sigma^2)$, against $H_0: \theta = \theta_0$, mixing the alternative over
 * $\theta \sim \Gauss(\theta_0, \tau^2)$ (Robbins, 1970):
 * $\log\Lambda_n = \frac12\log\frac{\sigma^2}{v_n} + \frac{n^2\tau^2(\bar x - \theta_0)^2}{2\sigma^2 v_n}$ with
 * $v_n = \sigma^2 + n\tau^2$. Under $H_0$, $\Lambda_n$ is a nonnegative martingale with mean 1, so $\Lambda_n$ is an
 * e-value at every $n$ and $\pr(\sup_n \Lambda_n \ge 1/\alpha) \le \alpha$ (Ville's inequality).
 *
 * @param n The number of observations (0 gives 0).
 * @param mean Their mean $\bar x$.
 * @param theta0 The null mean $\theta_0$.
 * @param m The mixture's $\sigma$ and $\tau$.
 * @returns $\log\Lambda_n$.
 *
 * @example The evidence of 25 observations grows with their distance from 0
 * const m = { sigma: 1, tau: 1 }
 * for (const mean of [0, 0.3, 0.6]) print('mean =', mean, ' log LR =', normalMixtureLogLikelihoodRatio(25, mean, 0, m))
 * print('reject at 5% once log LR >=', Math.log(1 / 0.05))
 */
export function normalMixtureLogLikelihoodRatio(n: number, mean: number, theta0: number, m: NormalMixture): number {
  mixture(m, 'normalMixtureLogLikelihoodRatio')
  if (n === 0) return 0
  const s2 = m.sigma ** 2
  const v = s2 + n * m.tau ** 2
  return 0.5 * Math.log(s2 / v) + (n * n * m.tau ** 2 * (mean - theta0) ** 2) / (2 * s2 * v)
}

/**
 * The half-width of the normal-mixture confidence sequence after $n$ observations: the $\theta_0$ with
 * $\log\Lambda_n(\theta_0) < \log(1/\alpha)$ are those with $\lvert \bar x - \theta_0 \rvert < r_n$, where
 * $r_n = \sqrt{\frac{2\sigma^2 v_n}{n^2\tau^2}\left(\log\frac{1}{\alpha} + \frac12\log\frac{v_n}{\sigma^2}\right)}$
 * and $v_n = \sigma^2 + n\tau^2$. The intervals $\bar x \pm r_n$ cover $\theta$ at every $n$ simultaneously with
 * probability at least $1 - \alpha$. The radius shrinks like $\sqrt{\log n / n}$, so peeking costs nothing.
 *
 * @param n The number of observations (0 gives $\infty$).
 * @param alpha The error $\alpha$, in $(0, 1)$.
 * @param m The mixture's $\sigma$ and $\tau$.
 * @returns The radius $r_n$.
 *
 * @example Wider than a fixed-n interval, but valid at every n at once
 * const m = { sigma: 1, tau: 1 }
 * for (const n of [10, 100, 1000, 10000])
 *   print('n =', n, ' radius =', normalMixtureRadius(n, 0.05, m), ' fixed-n 1.96/sqrt(n) =', 1.96 / Math.sqrt(n))
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
  /** The number of observations so far. */
  n: number
  /** Their mean. */
  mean: number
  /** $\log\Lambda_n$, the log mixture likelihood ratio (the log e-value). */
  logLikelihoodRatio: number
  /** The always-valid p-value $\min_{k \le n} \min(1, 1/\Lambda_k)$. */
  pValue: number
  /** True once the p-value is at most $\alpha$. */
  rejected: boolean
  /** True once rejected or the data run out. */
  terminated: boolean
}

/**
 * The mixture sequential probability ratio test (Robbins, 1970; Johari, Koomen, Pekelis and Walsh, 2017) of
 * $H_0: \theta = \theta_0$ (`theta0`, default 0) for observations from $\Gauss(\theta, \sigma^2)$ with $\sigma$
 * known: after each observation the mixture likelihood ratio $\Lambda_n$ (`normalMixtureLogLikelihoodRatio`) is
 * compared with $1/\alpha$, and the test stops when it crosses. The running p-value
 * $\min(1, 1/\max_{k \le n} \Lambda_k)$ is valid at any stopping time. For an A/B test, feed the paired differences
 * (or the per-observation differences of means) with their $\sigma$.
 *
 * @param data The observations, in arrival order.
 * @param options The test.
 * @param options.theta0 The null mean $\theta_0$.
 * @param options.alpha The level $\alpha$, in $(0, 1)$: the test rejects once $\Lambda_n \ge 1/\alpha$.
 * @param options.m The remaining fields, the mixture's `sigma` ($\sigma$, known) and `tau` ($\tau$).
 * @returns The test as an algorithm: run it with `run(test, undefined, steps)`.
 *
 * @example A mean of 0.3 is found early; a mean of 0 is never rejected
 * const shifted = run(msprt(normals(stream(1), 500, 0.3, 1), { sigma: 1, tau: 1 }), undefined, 500)
 * print('mean 0.3: rejected =', shifted.rejected, ' after', shifted.n, 'observations, p =', shifted.pValue)
 * const centred = run(msprt(normals(stream(2), 500, 0, 1), { sigma: 1, tau: 1 }), undefined, 500)
 * print('mean 0: rejected =', centred.rejected, ' after', centred.n, 'observations, p =', centred.pValue)
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
  /** The number of observations so far. */
  n: number
  /** Their mean $\bar x_n$. */
  mean: number
  /** The half-width $r_n$ of the current interval ($\infty$ before the first observation). */
  radius: number
  /** The running intersection of the intervals so far (each is valid, so their intersection is too): its lower end. */
  lower: number
  /** The upper end of the running intersection. */
  upper: number
  /** $\log\Lambda_n(\mu_0)$: the log e-value against the null mean $\mu_0$. */
  logEValue: number
  /** True once the e-value has reached $1/\alpha$ ($\mu_0$ left the sequence). */
  rejected: boolean
  /** True when the data run out. */
  terminated: boolean
}

/**
 * The normal-mixture confidence sequence for the mean of $\Gauss(\mu, \sigma^2)$ observations (Robbins, 1970;
 * Howard, Ramdas, McAuliffe and Sekhon, 2021): after $n$ observations, $\bar x_n \pm r_n$ with $r_n$ =
 * `normalMixtureRadius(n, alpha, { sigma, tau })`, intersected over time. The state also carries the e-value
 * $\Lambda_n(\mu_0)$ of the null mean $\mu_0$ (`mu0`, default 0); $\mu_0$ leaves the sequence exactly when the
 * e-value reaches $1/\alpha$. The sequence runs to the end of the data.
 *
 * @param data The observations, in arrival order.
 * @param options The sequence.
 * @param options.mu0 The null mean $\mu_0$ whose e-value is tracked.
 * @param options.alpha The error $\alpha$, in $(0, 1)$: the sequence covers the mean with probability $1 - \alpha$.
 * @param options.m The remaining fields, the mixture's `sigma` ($\sigma$, known) and `tau` ($\tau$).
 * @returns The sequence as an algorithm: run it with `run(seq, undefined, steps)`.
 *
 * @example 400 observations with mean 0.3
 * const s = run(confidenceSequence(normals(stream(1), 400, 0.3, 1), { sigma: 1, tau: 1 }), undefined, 400)
 * print('n =', s.n, ' mean =', s.mean, ' interval:', [s.lower, s.upper])
 * print('mean 0 rejected:', s.rejected, ' log e-value =', s.logEValue)
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
 * An alpha-spending function $\alpha^*(t)$, the type-I error spent by information fraction $t \in [0, 1]$ (Lan and
 * DeMets, 1983): `obrien-fleming`, $2 - 2\Phi(z_{1-\alpha/2}/\sqrt t)$, which spends almost nothing early; `pocock`,
 * $\alpha\log(1 + (e - 1)t)$, close to Pocock's constant boundary; `power`, $\alpha t^\rho$ (Kim and DeMets, 1987),
 * with $\rho$ = `rho` (default 1).
 */
export type Spending = { family: 'obrien-fleming' | 'pocock' | 'power'; rho?: number }

/**
 * $\alpha^*(t)$ for a spending function, total $\alpha$ (one- or two-sided as the design is). It is 0 at $t \le 0$
 * and $\alpha$ at $t \ge 1$.
 *
 * @param spending The spending function.
 * @param alpha The total error $\alpha$, in $(0, 1)$.
 * @param t The information fraction (clamped to at most 1).
 * @returns The error spent by $t$.
 *
 * @example Three ways to spend 5% over four looks
 * for (const t of [0.25, 0.5, 0.75, 1])
 *   print('t =', t, ' OBF:', spentAlpha({ family: 'obrien-fleming' }, 0.05, t),
 *     ' Pocock:', spentAlpha({ family: 'pocock' }, 0.05, t),
 *     ' t^2:', spentAlpha({ family: 'power', rho: 2 }, 0.05, t))
 */
export function spentAlpha(spending: Spending, alpha: number, t: number): number {
  rate(alpha, 'α', 'spentAlpha')
  if (t <= 0) return 0
  const u = Math.min(1, t)
  if (spending.family === 'obrien-fleming') return 2 - 2 * Phi(PhiInv(1 - alpha / 2) / Math.sqrt(u))
  if (spending.family === 'pocock') return alpha * Math.log(1 + (Math.E - 1) * u)
  return alpha * u ** (spending.rho ?? 1)
}

/** Group-sequential boundaries on the $z$ scale at each look, with the type-I error each look spends. */
export type GroupSequentialBoundaries = {
  /** Always `'group-sequential-boundaries'`. */
  readonly kind: 'group-sequential-boundaries'
  /** Information fractions $t_1 < \dots < t_K = 1$. */
  readonly information: Float64Array
  /**
   * Critical values $c_k$: reject at look $k$ when $Z_k \ge c_k$ (one-sided) or $\lvert Z_k \rvert \ge c_k$
   * (two-sided). Infinity: no stop.
   */
  readonly z: Float64Array
  /** The probability under $H_0$ of first crossing at each look; they sum to $\alpha$. */
  readonly crossing: Float64Array
  /** The total type-I error $\alpha$. */
  readonly alpha: number
  /** 2 for symmetric two-sided boundaries, 1 for an upper boundary only. */
  readonly sides: 1 | 2
}

const SQRT_2PI = Math.sqrt(2 * Math.PI)
/**
 * The standard normal density $\phi$.
 *
 * @param x The point.
 * @returns $\phi(x)$.
 */
const phi = (x: number) => Math.exp(-0.5 * x * x) / SQRT_2PI

/**
 * The recursive numerical integration of Armitage, McPherson and Rowe (1969) for the score process
 * $S_k = Z_k\sqrt{t_k}$, a Brownian motion in information time: `density` holds the sub-density of $S_k$ on a Simpson
 * grid (`grid`, with quadrature `weights`) over the continuation region (paths not yet stopped). `crossing` is the
 * probability of first crossing at the next look with critical value $c$; `advance` moves the density to that look.
 * `sides` is 2 for two-sided boundaries and 1 for an upper one; `points` is the grid size (odd, for Simpson's rule).
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

  /**
   * The continuation region's grid at information $t$ with critical value $c$, truncated at $\pm 10$ standard
   * deviations: $(-c\sqrt t, c\sqrt t)$ two-sided, $(-10\sqrt t, c\sqrt t)$ one-sided.
   *
   * @param t The information fraction of the look.
   * @param c The critical value on the $z$ scale (at most 10 is used).
   * @returns `points` equally spaced score values.
   */
  private region(t: number, c: number): Float64Array {
    const sd = Math.sqrt(t)
    const hi = Math.min(c, 10) * sd
    const lo = this.sides === 2 ? -hi : -10 * sd
    const n = this.points
    return Float64Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
  }

  /**
   * Simpson's rule weights for an equally spaced grid of an odd number of points.
   *
   * @param grid The grid.
   * @returns The weights $h/3 \cdot (1, 4, 2, 4, \dots, 4, 1)$.
   */
  private simpson(grid: Float64Array): Float64Array {
    const n = grid.length
    const h = (grid[n - 1] - grid[0]) / (n - 1)
    return Float64Array.from({ length: n }, (_, i) => ((i === 0 || i === n - 1 ? 1 : i % 2 === 1 ? 4 : 2) * h) / 3)
  }

  /**
   * $\pr(\text{first crossing at the look at information } t \text{ with value } c)$, from the current sub-density
   * at the previous look.
   *
   * @param tPrev The information fraction of the previous look (0 before the first).
   * @param t The information fraction of this look.
   * @param c The critical value on the $z$ scale.
   * @returns The crossing probability.
   */
  crossing(tPrev: number, t: number, c: number): number {
    const tail = (u: number, sd: number) =>
      1 - Phi((c * Math.sqrt(t) - u) / sd) + (this.sides === 2 ? Phi((-c * Math.sqrt(t) - u) / sd) : 0)
    if (tPrev === 0) return tail(0, Math.sqrt(t))
    const sd = Math.sqrt(t - tPrev)
    let p = 0
    for (let i = 0; i < this.grid.length; i++) p += this.weights[i] * this.density[i] * tail(this.grid[i], sd)
    return p
  }

  /**
   * Move the sub-density to the look at information $t$, keeping paths inside $(-c\sqrt t, c\sqrt t)$ (or below
   * $c\sqrt t$).
   *
   * @param tPrev The information fraction of the previous look (0 before the first).
   * @param t The information fraction of this look.
   * @param c The critical value on the $z$ scale at this look.
   */
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

/**
 * Bisection on $[0, 40]$ for the $c$ with $f(c) = \text{target}$, for a crossing probability $f$ decreasing in $c$.
 *
 * @param f The crossing probability as a function of the critical value.
 * @param target The probability to spend; at most $10^{-15}$ gives $\infty$ (no stop at this look).
 * @returns The critical value.
 */
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

/**
 * Information fractions $t_k$ from looks: $K$ equally spaced looks for a number, or the given information (or sample
 * sizes), divided by the last. Throws `DomainError` unless there is a look and the information increases from above
 * 0.
 *
 * @param information The number of looks, or the information at each look.
 * @param where The caller's name, for error messages.
 * @returns The fractions, ending at 1.
 */
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
 * Group-sequential boundaries from an alpha-spending function (Lan and DeMets, 1983): at each look $k$, the critical
 * value $c_k$ is chosen so that the probability under $H_0$ of first crossing at look $k$ is
 * $\alpha^*(t_k) - \alpha^*(t_{k-1})$. `information` is the number of equally spaced looks or the information (or
 * sample size) at each look; `sides` 2 (default) gives symmetric two-sided boundaries $\lvert Z \rvert \ge c_k$ with
 * total $\alpha$, spending $\alpha/2$ on each side by the one-sided function (the convention of gsDesign and
 * ldbounds), and 1 an upper boundary spending $\alpha$. Crossing probabilities are computed by recursive numerical
 * integration (Armitage, McPherson and Rowe, 1969) on a Simpson grid of `points` (default 401) per look.
 *
 * @param information The number of equally spaced looks, or the increasing information (or sample size) at each.
 * @param options The design.
 * @param options.alpha The total type-I error $\alpha$, in $(0, 1)$.
 * @param options.spending The alpha-spending function (default O'Brien–Fleming-like).
 * @param options.sides 2 for two-sided boundaries, 1 for an upper one.
 * @param options.points The grid size per look (made odd); more is slower and more accurate.
 * @returns The boundaries, with the crossing probability of each look.
 *
 * @example Four equally spaced looks, as gsDesign's Lan–DeMets O'Brien–Fleming design
 * const b = groupSequentialBoundaries(4)
 * print('z:', b.z)
 * print('alpha spent at each look:', b.crossing)
 * print('Pocock-like spending:', groupSequentialBoundaries(4, { spending: { family: 'pocock' } }).z)
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
 * The classical group-sequential boundaries with a fixed shape: `pocock` (Pocock, 1977), one critical value $c$ at
 * every look; `obrien-fleming` (O'Brien and Fleming, 1979), $c/\sqrt{t_k}$, wide early and near the fixed-sample value
 * at the end. $c$ is found so that the total probability of crossing under $H_0$ is $\alpha$.
 *
 * @param shape `pocock` or `obrien-fleming`.
 * @param information The number of equally spaced looks, or the increasing information (or sample size) at each.
 * @param options The design.
 * @param options.alpha The total type-I error $\alpha$, in $(0, 1)$.
 * @param options.sides 2 for two-sided boundaries, 1 for an upper one.
 * @param options.points The grid size per look (made odd); more is slower and more accurate.
 * @returns The boundaries, with the crossing probability of each look.
 *
 * @example Pocock's and O'Brien–Fleming's boundaries for four looks
 * // A grid of 101 points (default 401) keeps the cell fast; the values agree to six digits.
 * print('Pocock:', constantBoundaries('pocock', 4, { points: 101 }).z)
 * print("O'Brien-Fleming:", constantBoundaries('obrien-fleming', 4, { points: 101 }).z)
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
  /** The number of observations so far. */
  n: number
  /** Their mean. */
  mean: number
  /** The $z$ statistic at the most recent look (NaN before the first). */
  z: number
  /** The index of the most recent look ($-1$ before the first). */
  look: number
  /** True once a look's statistic crossed its boundary. */
  rejected: boolean
  /** True once rejected, after the last look, or when the data run out. */
  terminated: boolean
}

/**
 * A group-sequential $z$-test of $H_0: \mu = \mu_0$ (`mu0`, default 0) for $\Gauss(\mu, \sigma^2)$ observations with
 * $\sigma$ known: each step absorbs one observation; at the planned look sizes `looks` (cumulative counts, one per
 * boundary) the statistic $Z = (\bar x - \mu_0)\sqrt n/\sigma$ is compared with the boundary, and the test stops at
 * the first crossing ($\lvert Z \rvert \ge c_k$ two-sided, $Z \ge c_k$ one-sided) or after the last look. Throws
 * `DomainError` when the number of looks differs from the boundaries' or $\sigma \le 0$.
 *
 * @param data The observations, in arrival order.
 * @param options The test.
 * @param options.looks The cumulative sample sizes at which to look, increasing, one per boundary.
 * @param options.boundaries The critical values, from `groupSequentialBoundaries` or `constantBoundaries`.
 * @param options.sigma The known standard deviation $\sigma$.
 * @param options.mu0 The null mean $\mu_0$.
 * @returns The test as an algorithm: run it with `run(test, undefined, steps)`.
 *
 * @example A mean of 0.5 stops the trial at the second of four looks
 * const boundaries = groupSequentialBoundaries(4)
 * const x = normals(stream(5), 100, 0.5, 1)
 * const s = run(groupSequentialTest(x, { looks: [25, 50, 75, 100], boundaries, sigma: 1 }), undefined, 100)
 * print('rejected:', s.rejected, ' at look', s.look, '(from 0), n =', s.n)
 * print('z =', s.z, ' boundary =', boundaries.z[s.look])
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
  /** $C^+$, the upper cumulative sum (detects an increase). */
  upper: number
  /** $C^-$, the lower cumulative sum (detects a decrease). */
  lower: number
  /** True when this step signalled. */
  alarm: boolean
  /** The number of signals so far. */
  alarms: number
  /** The step of the first signal, counted from 1 ($-1$ before it). */
  firstAlarm: number
  /** True when the data run out (a signal does not stop the chart). */
  terminated: boolean
}

/**
 * Page's (1954) tabular CUSUM for a shift in the mean of $\Gauss(\mu_0, \sigma^2)$ observations ($\mu_0$ =
 * `target`), in standardised units $z_t = (x_t - \mu_0)/\sigma$: $C^+_t = \max(0, C^+_{t-1} + z_t - k)$ and
 * $C^-_t = \max(0, C^-_{t-1} - z_t - k)$, signalling when either exceeds $h$. $k$ (default $\tfrac12$, half the
 * shift to detect) is the reference value and $h$ (default 5) the decision interval. After a signal both sums restart
 * at 0 unless `reset` is false. `sides` `upper` or `lower` watches one direction only. Throws `DomainError` unless
 * $\sigma > 0$, $h > 0$ and $k \ge 0$.
 *
 * @param data The observations, in time order.
 * @param options The chart.
 * @param options.target The in-control mean $\mu_0$.
 * @param options.sigma The in-control standard deviation $\sigma$.
 * @param options.k The reference value $k$, in standard deviations.
 * @param options.h The decision interval $h$, in standard deviations.
 * @param options.sides `both`, `upper` (increases only) or `lower` (decreases only).
 * @param options.reset Restart both sums at 0 after a signal.
 * @returns The chart as an algorithm: run it with `run(chart, undefined, steps)` over the whole series.
 *
 * @example The mean shifts by one standard deviation after 30 observations
 * const before = toArray(normals(stream(6), 30, 0, 1))
 * const after = toArray(normals(stream(7), 30, 1, 1))
 * const s = run(cusum([...before, ...after]), undefined, 60)
 * print('first alarm at observation', s.firstAlarm, ' alarms:', s.alarms)
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
 * The average run length of a one-sided upper CUSUM ($k$, $h$ in standard-deviation units) on normal data whose mean
 * has shifted by `shift` standard deviations (0: the in-control $\mathrm{ARL}_0$, the mean time between false
 * alarms).
 *
 * - `markov-chain` (default; Brook and Evans, 1972): $[0, h]$ is cut into `states` cells of width
 *   $w = 2h/(2s - 1)$ ($s$ = `states`), the chart moves between cells with normal probabilities, and the ARL from 0 is
 *   the first entry of $(\Imat - \Rmat)^{-1}\ones$ for the transient block $\Rmat$. Converges to the exact ARL as the
 *   cells shrink.
 * - `siegmund` (Siegmund, 1985): $(e^{-2\Delta b} + 2\Delta b - 1)/(2\Delta^2)$ with $\Delta = \text{shift} - k$ and
 *   $b = h + 1.166$ ($b^2$ when $\Delta = 0$).
 *
 * `sides: 'both'` combines the upper and lower charts by $1/\mathrm{ARL} = 1/\mathrm{ARL}^+ + 1/\mathrm{ARL}^-$, a
 * close approximation for the two-sided chart (the two sums rarely are both positive). Throws `DomainError` unless
 * $h > 0$ and $k \ge 0$.
 *
 * @param options The chart and the shift.
 * @param options.k The reference value $k$.
 * @param options.h The decision interval $h$.
 * @param options.shift The shift of the mean, in standard deviations.
 * @param options.method `markov-chain` or `siegmund`.
 * @param options.states The number of cells of the Markov chain (a dense $s \times s$ solve).
 * @param options.sides `upper` for the one-sided chart, `both` for the two-sided one.
 * @returns The average run length, in observations.
 *
 * @example The standard chart (k = 0.5, h = 5) in control and after a one-sd shift
 * print('ARL0 =', cusumAverageRunLength(), ' two-sided:', cusumAverageRunLength({ sides: 'both' }))
 * print('ARL1 =', cusumAverageRunLength({ shift: 1 }))
 * print('Siegmund: ARL0 =', cusumAverageRunLength({ method: 'siegmund' }),
 *   ' ARL1 =', cusumAverageRunLength({ method: 'siegmund', shift: 1 }))
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
  if (!(h > 0) || !(k >= 0))
    throw new DomainError('cusumAverageRunLength', 'cusumAverageRunLength: needs h > 0 and k ≥ 0')
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
