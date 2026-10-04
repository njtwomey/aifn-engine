/**
 * The regularised incomplete beta function and its inverse, and the distribution functions built on it and on the
 * incomplete gamma function: Student t (cdf, quantile) and chi-square (cdf, survival).
 *
 * Method. I_x(a, b) by the continued fraction of Press et al., Numerical Recipes, 3rd ed., §6.4 (eq. 6.4.5), with the
 * symmetry I_x(a, b) = 1 − I_{1−x}(b, a) chosen so the fraction converges quickly. The inverse uses Newton's method on
 * log I in log x (and on the mirrored upper tail), inside a bracket.
 */

import { logBeta, regularisedGammaP, regularisedGammaQ, logGamma } from './gamma'
import { normalCdf, normalLogCdf, normalQuantile } from './normal'

const TINY = 1e-300
const MAX_ITER = 100_000

/** Continued fraction for I_x(a, b) · a B(a, b) / (xᵃ (1 − x)ᵇ), modified Lentz (NR3 eq. 6.4.5). */
function betaFraction(a: number, b: number, x: number): number {
  const qab = a + b
  const qap = a + 1
  const qam = a - 1
  let c = 1
  let d = 1 - (qab * x) / qap
  if (Math.abs(d) < TINY) d = TINY
  d = 1 / d
  let h = d
  for (let m = 1; m < MAX_ITER; m++) {
    const m2 = 2 * m
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2))
    d = 1 + aa * d
    if (Math.abs(d) < TINY) d = TINY
    c = 1 + aa / c
    if (Math.abs(c) < TINY) c = TINY
    d = 1 / d
    h *= d * c
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
    d = 1 + aa * d
    if (Math.abs(d) < TINY) d = TINY
    c = 1 + aa / c
    if (Math.abs(c) < TINY) c = TINY
    d = 1 / d
    const delta = d * c
    h *= delta
    if (Math.abs(delta - 1) < 1e-16) return h
  }
  return NaN
}

/** log of xᵃ (1 − x)ᵇ / B(a, b). */
function logBetaPrefactor(a: number, b: number, x: number): number {
  return a * Math.log(x) + b * Math.log1p(-x) - logBeta(a, b)
}

/**
 * The regularised incomplete beta function I_x(a, b) = ∫₀ˣ t^{a−1}(1 − t)^{b−1} dt / B(a, b), for a, b > 0 and
 * x in [0, 1]: the cdf of a Beta(a, b) variable. Argument order follows scipy.special.betainc. NaN if the continued
 * fraction fails to converge.
 */
export function regularisedBeta(a: number, b: number, x: number): number {
  if (!(a > 0 && b > 0 && x >= 0 && x <= 1)) return NaN
  if (x === 0) return 0
  if (x === 1) return 1
  // The fraction converges rapidly for x < (a + 1)/(a + b + 2); otherwise use the symmetry.
  if (x < (a + 1) / (a + b + 2)) return (Math.exp(logBetaPrefactor(a, b, x)) * betaFraction(a, b, x)) / a
  return 1 - (Math.exp(logBetaPrefactor(b, a, 1 - x)) * betaFraction(b, a, 1 - x)) / b
}

/** The Beta(a, b) density x^{a−1}(1 − x)^{b−1} / B(a, b), the x-derivative of I_x(a, b). */
export function betaDensity(a: number, b: number, x: number): number {
  if (x < 0 || x > 1) return 0
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x) - logBeta(a, b))
}

/**
 * log I_x(a, b): in log space in the directly summed branch, so that it does not underflow for tiny x, and log1p of the
 * complement in the mirrored branch, so that it keeps its relative accuracy where I_x ≈ 1. NaN for invalid arguments.
 */
export function logRegularisedBeta(a: number, b: number, x: number): number {
  if (!(a > 0 && b > 0 && x >= 0 && x <= 1)) return NaN
  if (x === 0) return -Infinity
  if (x === 1) return 0
  if (x < (a + 1) / (a + b + 2)) return logBetaPrefactor(a, b, x) + Math.log(betaFraction(a, b, x) / a)
  return Math.log1p(-(Math.exp(logBetaPrefactor(b, a, 1 - x)) * betaFraction(b, a, 1 - x)) / b)
}

/**
 * The inverse of the regularised incomplete beta function in x: I_x(a, b) = p for p in [0, 1], as (a, b, p) like
 * scipy.special.betaincinv. Relative accuracy about 1e-14 in x (and in 1 − x above the mean), including roots many
 * orders of magnitude below 1. NaN for invalid arguments.
 *
 * Method: below the mean m = a/(a + b), Newton's method on log I_x in s = log x, where the lower tail is nearly linear
 * (I_x ≈ xᵃ / (a B(a, b)) as x → 0, which also gives the first guess); above it, the same on the mirrored problem
 * I_y(b, a) = 1 − p with y = 1 − x. Every step is kept inside a bracket of the root.
 */
export function regularisedBetaInverse(a: number, b: number, p: number): number {
  if (!(a > 0 && b > 0 && p >= 0 && p <= 1)) return NaN
  if (p === 0) return 0
  if (p === 1) return 1
  const m = a / (a + b)
  if (p <= regularisedBeta(a, b, m)) return lowerTailInverse(a, b, Math.log(p), m)
  return 1 - lowerTailInverse(b, a, Math.log1p(-p), 1 - m)
}

/** Solve log I_x(a, b) = logP for x in (0, xMax], by safeguarded Newton steps in s = log x. */
function lowerTailInverse(a: number, b: number, logP: number, xMax: number): number {
  const lbeta = logBeta(a, b)
  let lo = -Infinity
  let hi = Math.log(xMax)
  let s = Math.min((logP + Math.log(a) + lbeta) / a, hi)
  for (let i = 0; i < 200; i++) {
    const x = Math.exp(s)
    const logI = logRegularisedBeta(a, b, x)
    const g = logI - logP
    if (g === 0) return x
    if (g < 0) lo = s
    else hi = s
    // d log I / d log x = x · density / I.
    const slope = Math.exp(a * s + (b - 1) * Math.log1p(-x) - lbeta - logI)
    let next = s - g / slope
    if (!(next > lo && next < hi)) next = Number.isFinite(lo) ? 0.5 * (lo + hi) : Math.min(s, hi) - 10
    // Newton converges quadratically, so once a step is below 1e-9 the next is at rounding level.
    if (Math.abs(next - s) < 1e-9 * Math.max(1, Math.abs(s)) && Math.abs(next - s) < 1e-7) {
      const x2 = Math.exp(next)
      const g2 = logRegularisedBeta(a, b, x2) - logP
      const slope2 = Math.exp(a * next + (b - 1) * Math.log1p(-x2) - lbeta - (g2 + logP))
      const last = next - g2 / slope2
      // Where the root underflows (x2 = 0), the polishing step is −∞/∞; keep the last iterate.
      return Math.exp(Number.isFinite(last) ? last : next)
    }
    s = next
  }
  return Math.exp(s)
}

/** The Student t density with ν degrees of freedom, the t-derivative of {@link studentTCdf}. */
export function studentTDensity(t: number, df: number): number {
  if (df === Infinity) return Math.exp(-0.5 * t * t) / Math.sqrt(2 * Math.PI)
  return Math.exp(
    logGamma((df + 1) / 2) -
      logGamma(df / 2) -
      0.5 * Math.log(df * Math.PI) -
      ((df + 1) / 2) * Math.log1p((t * t) / df),
  )
}

/**
 * The Student t cdf with ν > 0 degrees of freedom (ν = ∞ gives Φ), accurate in both tails.
 */
export function studentTCdf(t: number, df: number): number {
  if (Number.isNaN(t) || !(df > 0)) return NaN
  if (df === Infinity) return normalCdf(t)
  // The tail probability ½ I_x(ν/2, ½), x = ν/(ν + t²), keeps its relative accuracy for any t. Where the incomplete
  // beta would switch to its symmetric form, use I_x(ν/2, ½) = 1 − I_y(½, ν/2) with y = t²/(ν + t²) formed directly,
  // not as 1 − x.
  // x = ν/(ν + t²) and y = t²/(ν + t²) = 1 − x, formed from q = |t|/√ν (or its inverse) so that t² cannot overflow.
  const q = Math.abs(t) / Math.sqrt(df)
  const r = q > 1 ? 1 / q : q
  const small = (r * r) / (1 + r * r)
  const large = 1 / (1 + r * r)
  const x = q > 1 ? small : large
  const y = q > 1 ? large : small
  const a = df / 2
  if (q > 1e150) {
    // x = r² underflows; the leading term ½ xᵃ / (a B(a, ½)) of the tail has relative error O(x) there.
    const tail = 0.5 * Math.exp(2 * a * Math.log(r) - Math.log(a) - logBeta(a, 0.5))
    return t < 0 ? tail : 1 - tail
  }
  const tail = x < (a + 1) / (a + 2.5) ? 0.5 * regularisedBeta(a, 0.5, x) : 0.5 - 0.5 * regularisedBeta(0.5, a, y)
  return t < 0 ? tail : 1 - tail
}

/**
 * log of the Student t cdf with ν > 0 degrees of freedom. The lower tail is log ½ + log I_x(ν/2, ½) with
 * x = ν/(ν + t²), in log space so that it does not underflow; the upper tail is log1p of minus the lower tail at −t.
 */
export function studentTLogCdf(t: number, df: number): number {
  if (Number.isNaN(t) || !(df > 0)) return NaN
  if (df === Infinity) return normalLogCdf(t)
  if (t > 0) return Math.log1p(-studentTCdf(-t, df))
  const q = Math.abs(t) / Math.sqrt(df)
  const a = df / 2
  // The leading term of the tail where x = r² underflows (as in studentTCdf).
  if (q > 1e150) return -Math.LN2 - 2 * a * Math.log(q) - Math.log(a) - logBeta(a, 0.5)
  const r = q > 1 ? 1 / q : q
  const x = q > 1 ? (r * r) / (1 + r * r) : 1 / (1 + r * r)
  if (x < (a + 1) / (a + 2.5)) return -Math.LN2 + logRegularisedBeta(a, 0.5, x)
  return Math.log(studentTCdf(t, df))
}

/**
 * The Student t quantile with ν > 0 degrees of freedom: studentTCdf(q, ν) = p. For p < ¼ (and symmetrically p > ¾)
 * it inverts the tail form, so small tail probabilities keep their relative accuracy.
 */
export function studentTQuantile(p: number, df: number): number {
  if (!(p >= 0 && p <= 1 && df > 0)) return NaN
  if (df === Infinity) return normalQuantile(p)
  if (p === 0) return -Infinity
  if (p === 1) return Infinity
  if (p === 0.5) return 0
  const lower = p < 0.5
  const tail = lower ? p : 1 - p
  let t: number
  if (tail < 0.25) {
    // tail = ½ I_x(ν/2, ½) with x = ν/(ν + t²). When x would underflow (tiny ν, tiny tail), the leading term
    // I_x ≈ x^{ν/2} / ((ν/2) B(ν/2, ½)) gives log x directly and t ≈ √(ν/x) to relative error O(x).
    const a = df / 2
    const logX = (Math.log(2 * tail) + Math.log(a) + logBeta(a, 0.5)) / a
    if (logX < -600) t = Math.sqrt(df) * Math.exp(-0.5 * logX)
    else {
      const x = regularisedBetaInverse(a, 0.5, 2 * tail)
      t = Math.sqrt((df * (1 - x)) / x)
    }
  } else {
    // ½ − tail = ½ I_y(½, ν/2) with y = t²/(ν + t²); 1 − 2·tail is exact here.
    const y = regularisedBetaInverse(0.5, df / 2, 1 - 2 * tail)
    t = Math.sqrt((df * y) / (1 - y))
  }
  return lower ? -t : t
}

/** The chi-square cdf with k > 0 degrees of freedom: P(k/2, x/2). */
export function chiSquareCdf(x: number, k: number): number {
  if (x <= 0) return k > 0 ? 0 : NaN
  return regularisedGammaP(k / 2, x / 2)
}

/** The chi-square survival function 1 − F(x) = Q(k/2, x/2), accurate for small p-values. */
export function chiSquareSf(x: number, k: number): number {
  if (x <= 0) return k > 0 ? 1 : NaN
  return regularisedGammaQ(k / 2, x / 2)
}

/** The chi-square density with k degrees of freedom, the x-derivative of {@link chiSquareCdf}. */
export function chiSquareDensity(x: number, k: number): number {
  if (x < 0) return 0
  if (x === 0) return k === 2 ? 0.5 : k < 2 ? Infinity : 0
  const h = k / 2
  return Math.exp((h - 1) * Math.log(x / 2) - x / 2 - logGamma(h)) / 2
}
