/**
 * Goodness-of-fit tests of a sample's distribution: the Kolmogorov–Smirnov test (one sample against a continuous cdf,
 * or two samples; two- and one-sided, as scipy's `ks_1samp` and `ks_2samp`) with its exact null laws, and the
 * Shapiro–Wilk test of normality (Royston's 1995 algorithm AS R94, as scipy's `shapiro`).
 *
 * - One sample: $D = \sup_x \lvert F_n(x) - F(x) \rvert$, attained at a sorted draw $x_{(i)}$ as
 *   $\max(i/n - F(x_{(i)}), F(x_{(i)}) - (i - 1)/n)$; the one-sided $D^+$ and $D^-$ are the two halves. The exact law
 *   of $D$ for continuous $F$ is computed by Marsaglia, Tsang and Wang's matrix-power algorithm (2003, J. Stat. Softw.
 *   8(18)); far in the tail ($n d^2 > 7.24$, or $> 3.76$ with $n > 99$) their asymptotic expansion is used, as in their
 *   paper, and for $n d \ge 100$ Kolmogorov's limit with Stephens' correction (about 0.3% relative). The one-sided
 *   $D^+$ has the exact Birnbaum–Tingey law (Smirnov's).
 * - Two samples of sizes $m$ and $n$: $D = \sup_x \lvert F_m(x) - G_n(x) \rvert$ ($D^+ = \sup_x (F_m - G_n)$
 *   one-sided). Under the null every interleaving of the pooled sorted samples is equally likely, so $\pr(D \ge d)$ is
 *   one minus the probability that a uniformly random monotone lattice path from $(0, 0)$ to $(m, n)$ keeps
 *   $i/m - j/n$ inside the band, computed exactly by dynamic programming (Hodges, 1958, Ark. Mat. 3). Above
 *   $mn = 10^6$ the one-sample law at $n = \operatorname{round}(mn/(m + n))$ is used for two-sided tests (scipy's
 *   'asymp'), and Hodges' corrected limit (scipy's) for one-sided ones.
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
  /** $D$ (two-sided), $D^+$ (greater) or $D^-$ (less). */
  statistic: number
  /** The x at which it is attained. */
  location: number
  /** $+1$ when the (first) empirical cdf is above the reference there, $-1$ when below. */
  sign: 1 | -1
}

/** A reference distribution: a continuous cdf (one-sample test) or a second sample (two-sample test). */
export type KsReference = ((x: number) => number) | VectorLike

/**
 * A sample's values, checked (at least one, all finite) and sorted ascending.
 *
 * @param x The sample.
 * @param what The caller's name, for error messages.
 * @returns A new sorted Float64Array.
 */
function sortedValues(x: VectorLike, what: string): Float64Array {
  return sample(x, what).sort()
}

/**
 * $D$ (or $D^+$, $D^-$) for one sorted sample against a cdf: the largest of $i/n - F(x_{(i)})$ (the empirical cdf
 * above) and $F(x_{(i)}) - (i - 1)/n$ (below) over the sorted values.
 *
 * @param x The sample, sorted ascending.
 * @param cdf The reference cdf $F$.
 * @param alternative `two-sided` for $D$, `greater` for $D^+$ (above only), `less` for $D^-$ (below only).
 * @returns The statistic, its location and its sign.
 */
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

/**
 * $D$ (or $D^+$, $D^-$) for two sorted samples: the gaps between the empirical cdfs, read just after each distinct
 * value. A one-sided statistic is at least 0.
 *
 * @param x The first sample, sorted ascending.
 * @param y The second sample, sorted ascending.
 * @param alternative `two-sided` for $D$, `greater` for $D^+ = \sup (F_m - G_n)$, `less` for $D^- = \sup (G_n - F_m)$.
 * @returns The statistic, its location and its sign.
 */
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
 * The Kolmogorov–Smirnov statistic of a sample against a continuous cdf, or of two samples, with the $x$ at which it
 * is attained (scipy's `statistic_location` and `statistic_sign`): $D$ for `two-sided` (default),
 * $D^+ = \sup_x (F_n - F)$ for `greater`, $D^- = \sup_x (F - F_n)$ for `less`.
 *
 * @param x The sample (finite values; sorted here).
 * @param reference A continuous cdf, for the one-sample statistic, or a second sample.
 * @param alternative Which statistic.
 * @returns The statistic with its location and sign.
 *
 * @example Ten values against the uniform cdf on [0, 1]
 * const x = [0.05, 0.12, 0.31, 0.44, 0.52, 0.63, 0.71, 0.85, 0.93, 0.98]
 * const uniformCdf = (t) => Math.min(1, Math.max(0, t))
 * print(ksStatistic(x, uniformCdf))
 * print('D+ =', ksStatistic(x, uniformCdf, 'greater').statistic)
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

/**
 * Kolmogorov's limit $\pr(K > \lambda) = 2\sum_{k \ge 1} (-1)^{k-1} e^{-2k^2\lambda^2}$ (scipy's `kstwobign.sf`),
 * the law of $\sqrt n D_n$ as $n \to \infty$. Below $\lambda = 1$ the Jacobi-transformed series is summed instead.
 *
 * @param lambda The scaled distance $\lambda = \sqrt n\, d$.
 * @returns The survival probability, 1 for $\lambda \le 0$.
 *
 * @example The asymptotic 5% critical value is 1.36
 * for (const l of [0.5, 1, 1.36, 2]) print('P(K >', l, ') =', kolmogorovLimitSf(l))
 */
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
 * $\pr(D_n \ge d)$ for the one-sample statistic of $n$ draws from a continuous distribution (scipy's `kstwo.sf`), by
 * Marsaglia, Tsang and Wang (2003): $\pr(D_n < d) = \frac{n!}{n^n} (\Hmat^n)_{kk}$ with $k = \lfloor nd \rfloor + 1$
 * and $\Hmat$ the $(2k - 1) \times (2k - 1)$ matrix of their paper, its power taken by squaring with a decimal
 * exponent kept aside so that nothing overflows. Far in the tail their asymptotic expansion is used, and past
 * $k = 100$ Kolmogorov's limit with Stephens' correction (see the file's notes).
 *
 * @param d The observed distance, in $[0, 1]$.
 * @param n The sample size.
 * @returns The survival probability: 1 for $d \le 0$ and 0 for $d \ge 1$.
 *
 * @example The exact tail beside Kolmogorov's limit at n = 20
 * print('exact P(D >= 0.3) =', kolmogorovSf(0.3, 20))
 * print('limit at sqrt(20) * 0.3:', kolmogorovLimitSf(Math.sqrt(20) * 0.3))
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

/**
 * The product $\Amat\Bmat$ of two $m \times m$ matrices.
 *
 * @param a $\Amat$, row-major, $m^2$ values; not modified.
 * @param b $\Bmat$, row-major, $m^2$ values; not modified.
 * @param m The order of the matrices.
 * @returns A new row-major array of the product.
 */
function square(a: Float64Array, b: Float64Array, m: number): Float64Array {
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let l = 0; l < m; l++) {
      const ail = a[i * m + l]
      for (let j = 0; j < m; j++) out[i * m + j] += ail * b[l * m + j]
    }
  return out
}

/**
 * $\Amat^p$ as $\Qmat \cdot 10^{e}$, by repeated squaring, rescaling by $10^{-140}$ whenever the centre entry
 * exceeds $10^{140}$.
 *
 * @param a $\Amat$, row-major, $m^2$ values; not modified.
 * @param m The order of $\Amat$.
 * @param p The power, at least 1.
 * @returns `Q`, the scaled power (row-major), and `exponent`, the decimal exponent $e$ set aside.
 */
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
 * $\pr(D^+_n \ge d)$, the one-sided statistic's exact law (scipy's `smirnov`), by the Birnbaum–Tingey formula
 * $d \sum_j \binom{n}{j} (1 - d - j/n)^{n-j} (d + j/n)^{j-1}$ over $0 \le j \le \lfloor n(1 - d) \rfloor$, each term
 * computed in logs. $D^-_n$ has the same law.
 *
 * @param d The observed distance, in $[0, 1]$.
 * @param n The sample size.
 * @returns The survival probability: 1 for $d \le 0$ and 0 for $d \ge 1$.
 *
 * @example A one-sided tail is about half the two-sided one
 * print('P(D+ >= 0.3) =', smirnovSf(0.3, 20))
 * print('P(D >= 0.3) =', kolmogorovSf(0.3, 20))
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
 * $\pr(D \ge d)$ for two samples of sizes $m$ and $n$ under the null, exactly: one minus the probability that a
 * uniformly random monotone lattice path from $(0, 0)$ to $(m, n)$ stays strictly inside the band
 * ($\lvert i/m - j/n \rvert < d$ two-sided, $i/m - j/n < d$ one-sided). Compared in integers, $in - jm < dmn$ with
 * $dmn$ rounded, since every attainable $D$ is a multiple of $1/(mn)$. Each point's probability is accumulated with
 * the path's step probabilities, $(m - i)/(m - i + n - j)$ for a step in $i$, so nothing overflows. $O(mn)$
 * operations.
 *
 * @param d The observed distance.
 * @param m The size of the first sample.
 * @param n The size of the second sample.
 * @param oneSided Whether $d$ is the one-sided $D^+$ (the band is open on one side only).
 * @returns The survival probability, 1 for $d \le 0$.
 *
 * @example Two samples of five that overlap in one value
 * print('two-sided:', twoSampleKsSf(0.8, 5, 5))
 * print('one-sided:', twoSampleKsSf(0.8, 5, 5, true))
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
 * Hodges' (1958, eq. 5.3) corrected limit for the one-sided two-sample $D^+$:
 * $\exp(-2z^2 - 2z(M + 2N)/(3\sqrt{MN(M + N)}))$ with $z = d\sqrt{MN/(M + N)}$ and $M \ge N$ the larger and smaller
 * sizes (scipy's `ks_2samp`, `method='asymp'`). The correction term makes it far closer to the exact law than
 * Smirnov's plain $\exp(-2z^2)$ at small sizes.
 *
 * @param d The observed distance.
 * @param m The size of the first sample.
 * @param n The size of the second sample.
 * @returns The approximate survival probability, in $[0, 1]$.
 */
function hodgesOneSidedSf(d: number, m: number, n: number): number {
  const [big, small] = m >= n ? [m, n] : [n, m]
  const z = d * Math.sqrt((big * small) / (big + small))
  const e = -2 * z * z - (2 * z * (big + 2 * small)) / Math.sqrt(big * small * (big + small)) / 3
  return Math.min(1, Math.max(0, Math.exp(e)))
}

/**
 * The exact law of the one-sample $D$ (two-sided) or $D^+$ (one-sided) for $n$ draws, on $[0, 1]$; with `limit`, the
 * two-sided law is instead Kolmogorov's limit at $\sqrt n\, d$.
 *
 * @param n The sample size.
 * @param options Which law.
 * @param options.oneSided The law of $D^+$ (Smirnov's), rather than of $D$.
 * @param options.limit Use Kolmogorov's limit for $D$ (ignored when `oneSided`).
 * @returns The law, whose survival function is `kolmogorovSf`, `smirnovSf` or `kolmogorovLimitSf`.
 *
 * @example The 5% critical value of D for ten draws
 * const law = kolmogorovNull(10)
 * print('exact:', law.isf(0.05))
 * print('limit:', kolmogorovNull(10, { limit: true }).isf(0.05))
 */
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
 * The law of the two-sample $D$ (or $D^+$) for sizes $m$ and $n$: exact (`twoSampleKsSf`) up to $mn = 10^6$, and
 * beyond that or when `exact` is false the approximations of the file's notes: the one-sample law at the effective
 * size $\operatorname{round}(mn/(m + n))$ for $D$, Hodges' limit for $D^+$.
 *
 * @param m The size of the first sample.
 * @param n The size of the second sample.
 * @param options `oneSided`, the law of $D^+$ rather than $D$ (default false); `exact`, whether to use the exact law
 *   where it is affordable (default true).
 * @returns The law on $[0, 1]$.
 *
 * @example The tail of D for two samples of ten
 * // D is a multiple of 1/10 here: D >= 0.7 is the exact 5% region, D >= 0.6 is just outside it.
 * const exact = twoSampleKsNull(10, 10)
 * print('exact: P(D >= 0.6) =', exact.survival(0.6), ' P(D >= 0.7) =', exact.survival(0.7))
 * const approx = twoSampleKsNull(10, 10, { exact: false })
 * print('approximate: P(D >= 0.6) =', approx.survival(0.6), ' P(D >= 0.7) =', approx.survival(0.7))
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

/**
 * The result of `ksTest`: the protocol's fields with where the statistic is attained, `location`, and the side of the
 * reference the empirical cdf is on there, `sign` ($+1$ above, $-1$ below).
 */
export type KsTest = TestResult & { location: number; sign: 1 | -1 }

/**
 * The Kolmogorov–Smirnov test of a sample against a continuous cdf (`scipy.stats.ks_1samp`) or of two samples
 * (`scipy.stats.ks_2samp`). `two-sided` (default) uses $D$; `greater` uses $D^+$ (the first sample's cdf above the
 * reference: its values tend to be smaller) and `less` uses $D^-$. `method` is `exact` (default: the exact null
 * laws; see the file's notes for the tail and large two-sample cases) or `asymp`: Kolmogorov's limit for a two-sided
 * one-sample test, the one-sample exact law at the effective size for a two-sided two-sample test, and Hodges' limit
 * for a one-sided two-sample test (a one-sided one-sample test always uses Smirnov's exact law). Ties in a two-sample
 * test make the p-value conservative.
 *
 * @param x The sample (finite values).
 * @param reference A continuous cdf for the one-sample test, or a second sample for the two-sample test.
 * @param options `alternative` and `method`, as above.
 * @returns The test result, with the statistic's location and sign.
 *
 * @example Ten values against the uniform cdf: a fit, and a misfit that one side detects better
 * const x = [0.05, 0.12, 0.31, 0.44, 0.52, 0.63, 0.71, 0.85, 0.93, 0.98]
 * const uniformCdf = (t) => Math.min(1, Math.max(0, t))
 * const fit = ksTest(x, uniformCdf)
 * print('D =', fit.statistic, ' p =', fit.pValue)
 * const cubes = x.map((v) => v ** 3) // piled up near 0
 * print('cubes, two-sided: p =', ksTest(cubes, uniformCdf).pValue)
 * print('cubes, greater: p =', ksTest(cubes, uniformCdf, { alternative: 'greater' }).pValue)
 *
 * @example Two samples with shifted centres
 * const a = [0.61, 0.29, 0.06, 0.59, -1.73, -0.74, 0.51, -0.56, 0.39, 1.64]
 * const b = [2.13, 1.65, 2.62, 0.85, 1.49, 2.32, 1.97, 0.11, 1.68, 2.86]
 * const r = ksTest(a, b)
 * print(r.method)
 * print('D =', r.statistic, ' at', r.location, ' p =', r.pValue)
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

/**
 * The polynomial $\sum_i c_i x^i$, by Horner's rule.
 *
 * @param c The coefficients, constant term first.
 * @param x The point.
 * @returns The polynomial's value at $x$.
 */
const poly = (c: readonly number[], x: number) => c.reduceRight((acc, ci) => acc * x + ci, 0)

/**
 * The Shapiro–Wilk coefficients $a_i$ for a sample of size $n \ge 3$ (Royston, 1995, AS R94): the normal scores
 * $m_i = \Phi^{-1}((i - 3/8)/(n + 1/4))$ normalised, with the two most extreme (one when $n \le 5$) corrected by
 * polynomials in $1/\sqrt n$; for $n = 3$ the exact $\sqrt{1/2}$. Returned for the upper half, positive and
 * decreasing: the $i$-th weighs $x_{(n+1-i)} - x_{(i)}$, the lower half being their negatives (and the middle
 * coefficient of an odd $n$ zero). Throws `DomainError` unless $n$ is an integer of at least 3.
 *
 * @param n The sample size.
 * @returns The $\lfloor n/2 \rfloor$ coefficients, largest first; their squares sum to $\tfrac12$.
 *
 * @example The weights of the sample's extremes for n = 5 and 10
 * print('n = 5:', shapiroWilkCoefficients(5))
 * print('n = 10:', shapiroWilkCoefficients(10))
 */
export function shapiroWilkCoefficients(n: number): Float64Array {
  if (!(Number.isInteger(n) && n >= 3))
    throw new DomainError('shapiroWilkCoefficients', 'shapiroWilkCoefficients: needs an integer n ≥ 3')
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
 * The null law of $W$ used for its p-value (Royston, 1995): for $n = 3$ the exact
 * $\pr(W \le w) = \frac{6}{\pi}(\arcsin\sqrt w - \arcsin\sqrt{3/4})$ on $[3/4, 1]$; for $4 \le n \le 11$,
 * $-\log(\gamma - \log(1 - W))$ is normal with mean and log-sd polynomial in $n$, and $\gamma = -2.273 + 0.459n$;
 * for $n \ge 12$, $\log(1 - W)$ is normal with mean and log-sd polynomial in $\log n$. Small $W$ is evidence against
 * normality, so the p-value is the lower tail.
 *
 * @param n The sample size, at least 3 (Royston's fit is for $n \le 5000$).
 * @returns The law of $W$, a transformed normal (or, for $n = 3$, a `continuousLaw`).
 *
 * @example The 5% critical value of W grows towards 1 with n
 * for (const n of [3, 10, 50]) print('n =', n, ' W at 5% =', shapiroWilkNull(n).quantile(0.05))
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
 * The Shapiro–Wilk test of normality (Shapiro and Wilk, 1965) for $3 \le n \le 5000$ values:
 * $W = (\sum_i a_i x_{(i)})^2/\sum_i (x_i - \bar x)^2$, the squared correlation between the order statistics and
 * the normal scores; its p-value is the lower tail of Royston's (1995) approximation to the null law (scipy's
 * `shapiro`, which computes in single precision, agrees to about $10^{-5}$). Throws `DomainError` for more than 5000
 * values or when every value is equal.
 *
 * @param x The sample, 3 to 5000 finite values.
 * @returns The test result, with $W \in (0, 1]$ as its statistic.
 *
 * @example Evenly spread values pass, a skewed sample does not
 * const even = shapiroWilk([2.1, 2.9, 3.2, 3.5, 3.7, 4.0, 4.2, 4.6, 5.1, 5.8])
 * print('even: W =', even.statistic, ' p =', even.pValue)
 * const skewed = shapiroWilk([0.1, 0.2, 0.2, 0.3, 0.4, 0.6, 0.9, 1.5, 2.8, 6.0])
 * print('skewed: W =', skewed.statistic, ' p =', skewed.pValue)
 *
 * @example Heights with one tall outlier
 * const r = shapiroWilk([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])
 * print('W =', r.statistic, ' p =', r.pValue)
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
