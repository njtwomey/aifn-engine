/**
 * Goodness-of-fit tests of a sample's distribution: the Kolmogorov–Smirnov test (one sample against a continuous cdf,
 * or two samples; two- and one-sided, as scipy's `ks_1samp` and `ks_2samp`) with its exact null laws, and the
 * Shapiro–Wilk test of normality (Royston's 1995 algorithm AS R94, as scipy's `shapiro`).
 *
 * - One sample: D = supₓ |Fₙ(x) − F(x)|, attained at a sorted draw x₍ᵢ₎ as max(i/n − F(x₍ᵢ₎), F(x₍ᵢ₎) − (i − 1)/n);
 *   the one-sided D⁺ and D⁻ are the two halves. The exact law of D for continuous F is computed by Marsaglia, Tsang
 *   and Wang's matrix-power algorithm (2003, J. Stat. Softw. 8(18)); far in the tail (n d² > 7.24, or > 3.76 with
 *   n > 99) their asymptotic expansion is used, as in their paper, and for n d ≥ 100 Kolmogorov's limit with Stephens'
 *   correction (≈0.3% relative). The one-sided D⁺ has the exact Birnbaum–Tingey law (Smirnov's).
 * - Two samples of sizes m and n: D = supₓ |F_m(x) − G_n(x)| (D⁺ = sup (F_m − G_n) one-sided). Under the null every
 *   interleaving of the pooled sorted samples is equally likely, so P(D ≥ d) is one minus the probability that a
 *   uniformly random monotone lattice path from (0, 0) to (m, n) keeps i/m − j/n inside the band, computed exactly by
 *   dynamic programming (Hodges, 1958, Ark. Mat. 3). Above m·n = 10⁶ the one-sample law at n = round(mn/(m + n)) is
 *   used for two-sided tests (scipy's 'asymp'), and Hodges' corrected limit (scipy's) for one-sided ones.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { VectorLike } from 'aifn-compute/foundation/tensor'
import { affineBijector, chainBijectors, expBijector } from 'aifn-compute/probability/bijectors'
import { Normal, Transformed, type Univariate } from 'aifn-compute/probability/distributions'
import { logChoose, normalQuantile } from 'aifn-compute/numerics/special'
import { continuousLaw, pValueOf, result, sample, type Alternative, type TestResult } from './protocol'

// ── Statistics ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** The Kolmogorov–Smirnov statistic with where it is attained. */
export type KsStatistic = {
  /** D (two-sided), D⁺ (greater) or D⁻ (less). */
  statistic: number
  /** The x at which it is attained. */
  location: number
  /** +1 when the (first) empirical cdf is above the reference there, −1 when below. */
  sign: 1 | -1
}

/** A reference distribution: a continuous cdf (one-sample test) or a second sample (two-sample test). */
export type KsReference = ((x: number) => number) | VectorLike

function sortedValues(x: VectorLike, what: string): Float64Array {
  return sample(x, what).sort()
}

/** D (or D⁺, D⁻) for one sorted sample against a cdf. */
function oneSample(x: Float64Array, cdf: (x: number) => number, alternative: Alternative): KsStatistic {
  const n = x.length
  let best: KsStatistic = { statistic: -1, location: x[0], sign: 1 }
  for (let i = 0; i < n; i++) {
    const f = cdf(x[i])
    const above = (i + 1) / n - f
    const below = f - i / n
    if (alternative !== 'less' && above > best.statistic) best = { statistic: above, location: x[i], sign: 1 }
    if (alternative !== 'greater' && below > best.statistic) best = { statistic: below, location: x[i], sign: -1 }
  }
  return best
}

/** D (or D⁺, D⁻) for two sorted samples: the gaps between the empirical cdfs, read just after each distinct value. */
function twoSample(x: Float64Array, y: Float64Array, alternative: Alternative): KsStatistic {
  const m = x.length
  const n = y.length
  let i = 0
  let j = 0
  let best: KsStatistic = { statistic: 0, location: Math.min(x[0], y[0]), sign: 1 }
  while (i < m || j < n) {
    const v = Math.min(i < m ? x[i] : Infinity, j < n ? y[j] : Infinity)
    while (i < m && x[i] === v) i++
    while (j < n && y[j] === v) j++
    const d = i / m - j / n
    const s = alternative === 'greater' ? d : alternative === 'less' ? -d : Math.abs(d)
    if (s > best.statistic) best = { statistic: s, location: v, sign: d > 0 ? 1 : -1 }
  }
  return best
}

/**
 * The Kolmogorov–Smirnov statistic of a sample against a continuous cdf, or of two samples, with the x at which it is
 * attained (scipy's `statistic_location` and `statistic_sign`): D for `two-sided` (default), D⁺ = sup (Fₙ − F) for
 * `greater`, D⁻ = sup (F − Fₙ) for `less`.
 */
export function ksStatistic(
  x: VectorLike,
  reference: KsReference,
  alternative: Alternative = 'two-sided',
): KsStatistic {
  const xs = sortedValues(x, 'ksStatistic')
  return typeof reference === 'function'
    ? oneSample(xs, reference, alternative)
    : twoSample(xs, sortedValues(reference, 'ksStatistic'), alternative)
}

// ── Null laws ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Kolmogorov's limit P(K > λ) = 2 Σₖ (−1)ᵏ⁻¹ e^{−2k²λ²} (scipy's `kstwobign.sf`). */
export function kolmogorovLimitSf(lambda: number): number {
  if (!(lambda > 0)) return 1
  // For small λ the alternating series converges slowly; use the Jacobi-transformed form √(2π)/λ Σ e^{−(2k−1)²π²/(8λ²)}.
  if (lambda < 1) {
    let s = 0
    for (let k = 1; k <= 20; k++) s += Math.exp((-((2 * k - 1) ** 2) * Math.PI ** 2) / (8 * lambda * lambda))
    return 1 - (Math.sqrt(2 * Math.PI) / lambda) * s
  }
  let s = 0
  for (let k = 1; k <= 100; k++) {
    const term = Math.exp(-2 * k * k * lambda * lambda)
    s += (k % 2 === 1 ? 2 : -2) * term
    if (term < 1e-17) break
  }
  return s
}

/**
 * P(Dₙ ≥ d) for the one-sample statistic of n draws from a continuous distribution (scipy's `kstwo.sf`), by
 * Marsaglia, Tsang and Wang (2003): P(Dₙ < d) = n!/nⁿ · (Hⁿ)ₖₖ with k = ⌊nd⌋ + 1 and H the (2k − 1)-square matrix of
 * their paper, its power taken by squaring with a decimal exponent kept aside so that nothing overflows.
 */
export function kolmogorovSf(d: number, n: number): number {
  if (!(d > 0)) return 1
  if (d >= 1) return 0
  const s = d * d * n
  if (s > 7.24 || (s > 3.76 && n > 99)) return 2 * Math.exp(-(2.000071 + 0.331 / Math.sqrt(n) + 1.409 / n) * s)
  const k = Math.floor(n * d) + 1
  // Past k = 100 the (2k − 1)-square power costs ~log₂n · 8k³ operations; Kolmogorov's limit at Stephens' corrected
  // λ = (√n + 0.12 + 0.11/√n) d (Stephens, 1970, JRSS B 32(1)) is within ~0.3% relative there (n ≥ 1000).
  if (k > 100) return kolmogorovLimitSf((Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * d)
  const m = 2 * k - 1
  const h = k - n * d
  const H = new Float64Array(m * m)
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) H[i * m + j] = i - j + 1 < 0 ? 0 : 1
  for (let i = 0; i < m; i++) {
    H[i * m] -= h ** (i + 1)
    H[(m - 1) * m + i] -= h ** (m - i)
  }
  H[(m - 1) * m] += 2 * h - 1 > 0 ? (2 * h - 1) ** m : 0
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) if (i - j + 1 > 0) for (let g = 1; g <= i - j + 1; g++) H[i * m + j] /= g
  const { Q, exponent } = matrixPower(H, m, n)
  let v = Q[(k - 1) * m + k - 1]
  let e = exponent
  for (let i = 1; i <= n; i++) {
    v = (v * i) / n
    if (v < 1e-140) {
      v *= 1e140
      e -= 140
    }
  }
  return 1 - v * 10 ** e
}

/** A·B for m-square row-major matrices. */
function square(a: Float64Array, b: Float64Array, m: number): Float64Array {
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let l = 0; l < m; l++) {
      const ail = a[i * m + l]
      for (let j = 0; j < m; j++) out[i * m + j] += ail * b[l * m + j]
    }
  return out
}

/** Aᵖ as Q · 10^exponent, by repeated squaring, rescaling by 10⁻¹⁴⁰ whenever the centre entry exceeds 10¹⁴⁰. */
function matrixPower(a: Float64Array, m: number, p: number): { Q: Float64Array; exponent: number } {
  if (p === 1) return { Q: Float64Array.from(a), exponent: 0 }
  const half = matrixPower(a, m, Math.floor(p / 2))
  let Q = square(half.Q, half.Q, m)
  let exponent = 2 * half.exponent
  if (p % 2 === 1) Q = square(a, Q, m)
  const centre = Math.floor(m / 2)
  if (Q[centre * m + centre] > 1e140) {
    for (let i = 0; i < Q.length; i++) Q[i] *= 1e-140
    exponent += 140
  }
  return { Q, exponent }
}

/**
 * P(D⁺ₙ ≥ d), the one-sided statistic's exact law (scipy's `smirnov`), by the Birnbaum–Tingey formula
 * d Σⱼ C(n, j) (1 − d − j/n)ⁿ⁻ʲ (d + j/n)ʲ⁻¹ over 0 ≤ j ≤ ⌊n(1 − d)⌋, summed in logs.
 */
export function smirnovSf(d: number, n: number): number {
  if (!(d > 0)) return 1
  if (d >= 1) return 0
  let s = 0
  const last = Math.floor(n * (1 - d) + 1e-12)
  for (let j = 0; j <= last; j++) {
    const a = 1 - d - j / n
    const lead = n - j === 0 ? 0 : a <= 0 ? -Infinity : (n - j) * Math.log(a)
    s += Math.exp((logChoose(n, j) as number) + lead + (j - 1) * Math.log(d + j / n))
  }
  return Math.min(1, Math.max(0, d * s))
}

/**
 * P(D ≥ d) for two samples of sizes m and n under the null, exactly: one minus the probability that a uniformly random
 * monotone lattice path from (0, 0) to (m, n) stays strictly inside the band (|i/m − j/n| < d two-sided, i/m − j/n < d
 * one-sided). Compared in integers, i·n − j·m < d·m·n, since every attainable D is a multiple of 1/(mn). Each point's
 * probability is accumulated with the path's step probabilities, (m − i)/(m − i + n − j) for a step in i, so nothing
 * overflows.
 */
export function twoSampleKsSf(d: number, m: number, n: number, oneSided = false): number {
  if (!(d > 0)) return 1
  const bound = Math.round(d * m * n)
  const inside = (i: number, j: number) => (oneSided ? i * n - j * m : Math.abs(i * n - j * m)) < bound
  // p[j]: probability of reaching (i, j) without leaving the band, for the current i.
  let p = new Float64Array(n + 1)
  p[0] = 1
  for (let j = 1; j <= n; j++) p[j] = inside(0, j) ? (p[j - 1] * (n - j + 1)) / (m + n - j + 1) : 0
  for (let i = 1; i <= m; i++) {
    const next = new Float64Array(n + 1)
    for (let j = 0; j <= n; j++) {
      if (!inside(i, j)) continue
      // Arrive from (i − 1, j) by a step in i, or from (i, j − 1) by a step in j.
      const fromI = (p[j] * (m - i + 1)) / (m - i + 1 + n - j)
      const fromJ = j > 0 ? (next[j - 1] * (n - j + 1)) / (m - i + n - j + 1) : 0
      next[j] = fromI + fromJ
    }
    p = next
  }
  return Math.min(1, Math.max(0, 1 - p[n]))
}

/**
 * Hodges' (1958, eq. 5.3) corrected limit for the one-sided two-sample D⁺: exp(−2z² − 2z(M + 2N)/(3√(MN(M + N))))
 * with z = d√(MN/(M + N)) and M ≥ N the larger and smaller sizes (scipy's `ks_2samp`, `method='asymp'`). The
 * correction term makes it far closer to the exact law than Smirnov's plain exp(−2z²) at small sizes.
 */
function hodgesOneSidedSf(d: number, m: number, n: number): number {
  const [big, small] = m >= n ? [m, n] : [n, m]
  const z = d * Math.sqrt((big * small) / (big + small))
  const e = -2 * z * z - (2 * z * (big + 2 * small)) / Math.sqrt(big * small * (big + small)) / 3
  return Math.min(1, Math.max(0, Math.exp(e)))
}

/** The exact law of the one-sample D (two-sided) or D⁺ (one-sided) for n draws, on [0, 1]. */
export function kolmogorovNull(n: number, { oneSided = false, limit = false } = {}): Univariate {
  const sf = oneSided
    ? (d: number) => smirnovSf(d, n)
    : limit
      ? (d: number) => kolmogorovLimitSf(Math.sqrt(n) * d)
      : (d: number) => kolmogorovSf(d, n)
  return continuousLaw({
    name: oneSided ? 'Smirnov' : limit ? 'KolmogorovLimit' : 'Kolmogorov',
    params: { n },
    lower: 0,
    upper: 1,
    cdf: (d) => 1 - sf(d),
    survival: sf,
  })
}

/**
 * The law of the two-sample D (or D⁺) for sizes m and n: exact up to m·n = 10⁶ (or when `exact` is false), then the
 * approximations above.
 */
export function twoSampleKsNull(
  m: number,
  n: number,
  options: { oneSided?: boolean; exact?: boolean } = {},
): Univariate {
  const oneSided = options.oneSided ?? false
  const exact = (options.exact ?? true) && m * n <= 1e6
  const en = (m * n) / (m + n)
  const sf = exact
    ? (d: number) => twoSampleKsSf(d, m, n, oneSided)
    : oneSided
      ? (d: number) => hodgesOneSidedSf(d, m, n)
      : (d: number) => kolmogorovSf(d, Math.round(en))
  return continuousLaw({
    name: 'KolmogorovSmirnovTwoSample',
    params: { m, n },
    lower: 0,
    upper: 1,
    cdf: (d) => 1 - sf(d),
    survival: sf,
  })
}

// ── The test ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The result of `ksTest`: the protocol's fields with where the statistic is attained. */
export type KsTest = TestResult & { location: number; sign: 1 | -1 }

/**
 * The Kolmogorov–Smirnov test of a sample against a continuous cdf (`scipy.stats.ks_1samp`) or of two samples
 * (`scipy.stats.ks_2samp`). `two-sided` (default) uses D; `greater` uses D⁺ (the first sample's cdf above the
 * reference: its values tend to be smaller) and `less` uses D⁻. `method` 'exact' (default: the exact null laws; see
 * the module notes for the tail and large two-sample cases) or 'asymp' (Kolmogorov's limit for one sample; the
 * one-sample exact law at the effective size for two). Ties in a two-sample test make the p-value conservative.
 */
export function ksTest(
  x: VectorLike,
  reference: KsReference,
  options: { alternative?: Alternative; method?: 'exact' | 'asymp' } = {},
): KsTest {
  const alternative = options.alternative ?? 'two-sided'
  const method = options.method ?? 'exact'
  const xs = sortedValues(x, 'ksTest')
  const oneSided = alternative !== 'two-sided'
  if (typeof reference === 'function') {
    const s = oneSample(xs, reference, alternative)
    const n = xs.length
    const law = kolmogorovNull(n, { oneSided, limit: !oneSided && method === 'asymp' })
    return {
      ...result({
        test: 'ksTest',
        method: `One-sample Kolmogorov–Smirnov test (${oneSided ? 'exact' : method === 'asymp' ? 'asymptotic' : 'exact'})`,
        statistic: s.statistic,
        symbol: oneSided ? (alternative === 'greater' ? 'D^+' : 'D^-') : 'D',
        pValue: pValueOf(law, s.statistic, 'upper'),
        alternative,
        tail: 'upper',
        null: law,
        n,
      }),
      location: s.location,
      sign: s.sign,
    }
  }
  const ys = sortedValues(reference, 'ksTest')
  const s = twoSample(xs, ys, alternative)
  const [m, n] = [xs.length, ys.length]
  const exact = method === 'exact' && m * n <= 1e6
  // Beyond the exact range the two-sided law is the one-sample law at the effective size, the one-sided law Hodges'
  // corrected limit.
  const law =
    exact || oneSided ? twoSampleKsNull(m, n, { oneSided, exact }) : kolmogorovNull(Math.round((m * n) / (m + n)))
  return {
    ...result({
      test: 'ksTest',
      method: `Two-sample Kolmogorov–Smirnov test (${exact ? 'exact' : 'asymptotic'})`,
      statistic: s.statistic,
      symbol: oneSided ? (alternative === 'greater' ? 'D^+' : 'D^-') : 'D',
      pValue: pValueOf(law, s.statistic, 'upper'),
      alternative,
      tail: 'upper',
      null: law,
      n: m + n,
    }),
    location: s.location,
    sign: s.sign,
  }
}

// ── Shapiro–Wilk ─────────────────────────────────────────────────────────────────────────────────────────────────────

const poly = (c: readonly number[], x: number) => c.reduceRight((acc, ci) => acc * x + ci, 0)

/**
 * The Shapiro–Wilk coefficients aᵢ for a sample of size n ≥ 3 (Royston, 1995, AS R94): the normal scores
 * mᵢ = Φ⁻¹((i − 3/8)/(n + 1/4)) normalised, with the two most extreme corrected by polynomials in 1/√n. Returned
 * for the upper half (a₁ ≥ a₂ ≥ …), the lower half being their negatives.
 */
export function shapiroWilkCoefficients(n: number): Float64Array {
  if (!(Number.isInteger(n) && n >= 3)) throw new DomainError('shapiroWilkCoefficients', 'needs n ≥ 3')
  const half = Math.floor(n / 2)
  const a = new Float64Array(half)
  if (n === 3) {
    a[0] = Math.SQRT1_2
    return a
  }
  // m[i] for the lower half (negative), i = 1 … half.
  const m = Float64Array.from({ length: half }, (_, i) => normalQuantile((i + 1 - 0.375) / (n + 0.25)) as number)
  const summ2 = 2 * m.reduce((s, v) => s + v * v, 0)
  const ssumm2 = Math.sqrt(summ2)
  const rsn = 1 / Math.sqrt(n)
  const a1 = poly([0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056], rsn) - m[0] / ssumm2
  let fac: number
  let first: number
  if (n > 5) {
    const a2 = -m[1] / ssumm2 + poly([0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633], rsn)
    fac = Math.sqrt((summ2 - 2 * m[0] ** 2 - 2 * m[1] ** 2) / (1 - 2 * a1 ** 2 - 2 * a2 ** 2))
    a[1] = a2
    first = 2
  } else {
    fac = Math.sqrt((summ2 - 2 * m[0] ** 2) / (1 - 2 * a1 ** 2))
    first = 1
  }
  a[0] = a1
  for (let i = first; i < half; i++) a[i] = -m[i] / fac
  return a
}

/**
 * The null law of W used for its p-value (Royston, 1995): for n = 3 the exact P(W ≤ w) = (6/π)(asin √w − asin √¾) on
 * [¾, 1]; for 4 ≤ n ≤ 11, −log(γ − log(1 − W)) is normal with mean and log-sd polynomial in n, and γ = −2.273 +
 * 0.459n; for n ≥ 12, log(1 − W) is normal with mean and log-sd polynomial in log n. Small W is evidence against
 * normality, so the p-value is the lower tail.
 */
export function shapiroWilkNull(n: number): Univariate {
  if (n === 3) {
    const c = Math.asin(Math.sqrt(0.75))
    return continuousLaw({
      name: 'ShapiroWilk',
      params: { n },
      lower: 0.75,
      upper: 1,
      cdf: (w) => Math.min(1, Math.max(0, (6 / Math.PI) * (Math.asin(Math.sqrt(w)) - c))),
    })
  }
  // W = 1 − exp(log(1 − W)): from the normal Y through decreasing maps.
  const toW = [expBijector, affineBijector(1, -1)]
  if (n <= 11) {
    const gamma = poly([-2.273, 0.459], n)
    const mean = poly([0.544, -0.39978, 0.025054, -6.714e-4], n)
    const sd = Math.exp(poly([1.3822, -0.77857, 0.062767, -0.0020322], n))
    // Y = −log(γ − log(1 − W)), so log(1 − W) = γ − e^{−Y}.
    return Transformed(
      Normal(mean, sd),
      chainBijectors(affineBijector(0, -1), expBijector, affineBijector(gamma, -1), ...toW),
    )
  }
  const l = Math.log(n)
  const mean = poly([-1.5861, -0.31082, -0.083751, 0.0038915], l)
  const sd = Math.exp(poly([-0.4803, -0.082676, 0.0030302], l))
  return Transformed(Normal(mean, sd), chainBijectors(...toW))
}

/**
 * The Shapiro–Wilk test of normality (Shapiro and Wilk, 1965) for 3 ≤ n ≤ 5000 values: W = (Σ aᵢ x₍ᵢ₎)²/Σ (xᵢ − x̄)²,
 * the squared correlation between the order statistics and the normal scores; its p-value is the lower tail of
 * Royston's (1995) approximation to the null law (scipy's `shapiro`, which computes in single precision, agrees to
 * about 10⁻⁵).
 */
export function shapiroWilk(x: VectorLike): TestResult {
  const v = sample(x, 'shapiroWilk', 3).sort()
  const n = v.length
  if (n > 5000) throw new DomainError('shapiroWilk', 'shapiroWilk: Royston’s approximation holds for n ≤ 5000')
  const range = v[n - 1] - v[0]
  if (!(range > 0)) throw new DomainError('shapiroWilk', 'shapiroWilk: every value is equal')
  const a = shapiroWilkCoefficients(n)
  let num = 0
  for (let i = 0; i < a.length; i++) num += a[i] * (v[n - 1 - i] - v[i])
  let mean = 0
  for (const t of v) mean += t
  mean /= n
  let ss = 0
  for (const t of v) ss += (t - mean) ** 2
  const W = Math.min(1, (num * num) / ss)
  const law = shapiroWilkNull(n)
  return result({
    test: 'shapiroWilk',
    method: 'Shapiro–Wilk test of normality',
    statistic: W,
    symbol: 'W',
    pValue: pValueOf(law, W, 'lower'),
    alternative: 'two-sided',
    tail: 'lower',
    null: law,
    n,
  })
}
