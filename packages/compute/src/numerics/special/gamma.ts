/**
 * The gamma function family: log Γ, Γ, digamma ψ, trigamma ψ₁ and polygamma ψ⁽ⁿ⁾, log B, log n!, log (n choose k),
 * and the regularised incomplete gamma functions P and Q.
 *
 * Method. For x ≥ 10, log Γ comes from Stirling's series (A&S 6.1.40) with eight Bernoulli terms, written as
 * log Γ(x) = (x − ½) log x − x + ½ log 2π + δ(x) so that the small correction δ can be reused where large terms would
 * otherwise cancel (log B, log (n choose k), the incomplete gamma prefactor). Smaller x are shifted up by the recurrence
 * Γ(x + 1) = xΓ(x); negative x use the reflection formula Γ(x)Γ(1 − x) = π / sin πx. ψ and ψ⁽ⁿ⁾ use the same shift
 * and their asymptotic series (A&S 6.3.18, 6.4.11). P and Q use the power series and continued fraction of Press et
 * al., Numerical Recipes, 3rd ed., §6.2; log P and log Q sum the same series in log space, so neither underflows in its
 * own tail. The inverses solve log P = log p (or log Q = log q above the median) by safeguarded Newton steps in log x.
 */

import { normalQuantile } from './normal'
import { log1pmx } from './stable'

const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI)
/** Stirling-series coefficients B₂ₖ / (2k(2k − 1)), k = 1…8. */
const STIRLING = [1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188, -691 / 360360, 1 / 156, -3617 / 122400]
/** Bernoulli numbers B₂ₖ, k = 1…9. */
const BERNOULLI = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510, 43867 / 798]

/** δ(x) = log Γ(x) − [(x − ½) log x − x + ½ log 2π], for x ≥ 10 (Stirling's series; error below 1e-17). */
export function stirlingCorrection(x: number): number {
  const r = 1 / (x * x)
  let s = STIRLING[STIRLING.length - 1]
  for (let i = STIRLING.length - 2; i >= 0; i--) s = s * r + STIRLING[i]
  return s / x
}

/** sin(πx), exact at integers and accurate for large |x| (the argument is reduced modulo 2 exactly first). */
export function sinPi(x: number): number {
  let r = x % 2 // exact
  if (r < -1) r += 2
  else if (r > 1) r -= 2
  if (r === 0 || r === 1 || r === -1) return 0
  if (r === 0.5) return 1
  if (r === -0.5) return -1
  return Math.sin(Math.PI * r)
}

/** π / tan(πx), for the digamma reflection; ±∞ at integers. */
function piCotPi(x: number): number {
  let r = x % 1 // exact, in (−1, 1)
  if (r === 0) return Infinity
  if (r > 0.5) r -= 1
  else if (r < -0.5) r += 1
  return Math.PI / Math.tan(Math.PI * r)
}

const LOG_FACTORIAL: number[] = [0, 0]
for (let n = 2; n <= 170; n++) LOG_FACTORIAL[n] = LOG_FACTORIAL[n - 1] + Math.log(n)
const FACTORIAL: number[] = [1]
for (let n = 1; n <= 170; n++) FACTORIAL[n] = FACTORIAL[n - 1] * n

/**
 * log |Γ(x)|. +∞ at the poles x = 0, −1, −2, …. Relative error about 1e-15 away from the zeros at x = 1 and x = 2,
 * where the error is about 2e-15 absolute; exact (to rounding of log n!) at integers.
 */
export function logGamma(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x === Infinity) return Infinity
  if (x <= 0 && Number.isInteger(x)) return Infinity
  if (x < 0) return Math.log(Math.PI / Math.abs(sinPi(x))) - logGamma(1 - x)
  if (Number.isInteger(x) && x <= 171) return LOG_FACTORIAL[x - 1]
  if (x >= 10) return (x - 0.5) * Math.log(x) - x + HALF_LOG_2PI + stirlingCorrection(x)
  // Shift x up to 10 or more: log Γ(x) = log Γ(x + n) − log[x(x + 1)…(x + n − 1)].
  let prod = 1
  let y = x
  while (y < 10) prod *= y++
  return logGamma(y) - Math.log(prod)
}

/** The gamma function Γ(x). +∞ above x ≈ 171.62; NaN at the poles x = 0, −1, −2, …. Relative error about 1e-14. */
export function gamma(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x <= 0 && Number.isInteger(x)) return NaN
  if (x > 171.7) return Infinity
  if (Number.isInteger(x)) return FACTORIAL[x - 1]
  if (x < 0.5) return Math.PI / (sinPi(x) * gamma(1 - x))
  if (x < 10) {
    // Γ(x) = Γ(x + n) / [x(x + 1)…(x + n − 1)], with Γ(x + n) from Stirling.
    let prod = 1
    let y = x
    while (y < 10) prod *= y++
    return gamma(y) / prod
  }
  // Γ(x) = √(2π/x) (x/e)^x e^{δ(x)}, with the power halved so it does not overflow before the result does.
  const p = Math.pow(x, 0.5 * x - 0.25)
  return p * Math.exp(-x) * p * Math.sqrt(2 * Math.PI) * Math.exp(stirlingCorrection(x))
}

/**
 * The digamma function ψ(x) = d/dx log Γ(x). NaN at the poles x = 0, −1, −2, …. Absolute error about 1e-15 near its
 * zero x₀ ≈ 1.4616, relative error about 1e-15 elsewhere.
 */
export function digamma(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x === Infinity) return Infinity
  if (x <= 0 && Number.isInteger(x)) return NaN
  // Reflection: ψ(1 − x) − ψ(x) = π cot πx.
  if (x < 0) return digamma(1 - x) - piCotPi(x)
  let shift = 0
  while (x < 10) shift -= 1 / x++
  // ψ(x) ~ log x − 1/(2x) − Σ B₂ₖ / (2k x^{2k})  (A&S 6.3.18).
  const r = 1 / (x * x)
  let s = 0
  for (let k = 7; k >= 1; k--) s = s * r + BERNOULLI[k - 1] / (2 * k)
  return shift + Math.log(x) - 0.5 / x - s * r
}

/**
 * The polygamma function ψ⁽ⁿ⁾(x) = dⁿ⁺¹/dxⁿ⁺¹ log Γ(x) for integer n ≥ 1 and x > 0 (NaN otherwise; use
 * {@link trigamma} for negative x with n = 1). Recurrence to x ≥ 20 + n, then A&S 6.4.11.
 */
export function polygamma(n: number, x: number): number {
  if (!(Number.isInteger(n) && n >= 1 && x > 0)) return NaN
  if (x === Infinity) return 0
  // ψ⁽ⁿ⁾(x) = ψ⁽ⁿ⁾(x + 1) − (−1)ⁿ n! / x^{n+1}.
  const sign = n % 2 === 0 ? -1 : 1 // (−1)^{n+1}
  const nFact = FACTORIAL[n]
  let shift = 0
  while (x < 20 + n) shift += nFact / Math.pow(x++, n + 1)
  // ψ⁽ⁿ⁾(x) ~ (−1)^{n+1} [(n − 1)!/xⁿ + n!/(2x^{n+1}) + Σₖ B₂ₖ (2k + n − 1)!/((2k)! x^{2k+n})].
  let series = FACTORIAL[n - 1] / Math.pow(x, n) + nFact / (2 * Math.pow(x, n + 1))
  let ratio = FACTORIAL[n + 1] / (2 * Math.pow(x, n + 2)) // (2k + n − 1)!/((2k)! x^{2k+n}) at k = 1
  for (let k = 1; k <= BERNOULLI.length; k++) {
    const term = BERNOULLI[k - 1] * ratio
    series += term
    if (Math.abs(term) < 1e-17 * Math.abs(series)) break
    // Advance k → k + 1: multiply by (2k + n)(2k + n + 1) / ((2k + 1)(2k + 2) x²).
    ratio *= ((2 * k + n) * (2 * k + n + 1)) / ((2 * k + 1) * (2 * k + 2) * x * x)
  }
  return sign * (series + shift)
}

/** The trigamma function ψ₁(x) = d²/dx² log Γ(x), for all real x except the poles (reflection for x < 0). */
export function trigamma(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x <= 0 && Number.isInteger(x)) return NaN
  // ψ₁(1 − x) + ψ₁(x) = π² / sin² πx.
  if (x < 0) {
    const s = sinPi(x)
    return (Math.PI * Math.PI) / (s * s) - trigamma(1 - x)
  }
  return polygamma(1, x)
}

/**
 * log B(a, b) = log Γ(a) + log Γ(b) − log Γ(a + b) for a, b > 0. When a or b is large the three log Γ terms are
 * combined through their Stirling corrections, so the result stays accurate where they would cancel.
 */
export function logBeta(a: number, b: number): number {
  if (!(a > 0 && b > 0)) return NaN
  if (a > b) [a, b] = [b, a]
  if (b === Infinity) return -Infinity
  if (b < 10) return logGamma(a) + logGamma(b) - logGamma(a + b)
  const s = a + b
  const corr = stirlingCorrection(b) - stirlingCorrection(s)
  if (a < 10) {
    // log Γ(b) − log Γ(a + b) = δ(b) − δ(a + b) + (b − ½) log(b/(a + b)) − a log(a + b) + a.
    return logGamma(a) + corr + (b - 0.5) * Math.log1p(-a / s) - a * Math.log(s) + a
  }
  // Both large: ½ log 2π − ½ log b + (a − ½) log(a/(a + b)) + b log(b/(a + b)) + δ(a) + δ(b) − δ(a + b).
  return (
    HALF_LOG_2PI -
    0.5 * Math.log(b) +
    (a - 0.5) * Math.log(a / s) +
    b * Math.log1p(-a / s) +
    stirlingCorrection(a) +
    corr
  )
}

/** log n! = log Γ(n + 1) for n ≥ 0 (non-integer n uses the gamma function). */
export function logFactorial(n: number): number {
  if (Number.isInteger(n) && n >= 0 && n <= 170) return LOG_FACTORIAL[n]
  return logGamma(n + 1)
}

/**
 * log of the binomial coefficient (n choose k) for 0 ≤ k ≤ n (−∞ outside; real arguments allowed), computed as
 * −log(n + 1) − log B(n − k + 1, k + 1) so it is accurate even for n = 10⁹ and small k.
 */
export function logChoose(n: number, k: number): number {
  if (k < 0 || k > n) return -Infinity
  if (k === 0 || k === n) return 0
  return -Math.log1p(n) - logBeta(n - k + 1, k + 1)
}

/**
 * log of the incomplete gamma prefactor xᵃ e^{−x} / Γ(a). For a ≥ 10 it is written as
 * a·log1pmx((x − a)/a) + ½ log(a/2π) − δ(a) when x is near a, which avoids the cancellation of a log x − x against log Γ(a).
 */
function logGammaPrefactor(a: number, x: number): number {
  // Far from x = a there is no cancellation, and (x − a)/a near −1 would lose accuracy inside log1p.
  if (a < 10 || Math.abs(x - a) > 0.5 * a) return a * Math.log(x) - x - logGamma(a)
  return a * log1pmx((x - a) / a) + 0.5 * Math.log(a / (2 * Math.PI)) - stirlingCorrection(a)
}

const MAX_ITER = 100_000
const TINY = 1e-300

/** Series Σₙ xⁿ / (a(a+1)…(a+n)) for P(a, x) / prefactor (NR3 §6.2). NaN if it has not converged. */
function gammaSeries(a: number, x: number): number {
  let term = 1 / a
  let sum = term
  for (let n = 1; n < MAX_ITER; n++) {
    term *= x / (a + n)
    sum += term
    if (term < sum * 1e-17) return sum
  }
  return NaN
}

/** Continued fraction for Q(a, x) / prefactor, modified Lentz (NR3 §6.2). NaN if it has not converged. */
function gammaFraction(a: number, x: number): number {
  let b = x + 1 - a
  let c = 1 / TINY
  let d = 1 / b
  let h = d
  for (let i = 1; i < MAX_ITER; i++) {
    const an = -i * (i - a)
    b += 2
    d = an * d + b
    if (Math.abs(d) < TINY) d = TINY
    c = b + an / c
    if (Math.abs(c) < TINY) c = TINY
    d = 1 / d
    const delta = d * c
    h *= delta
    if (Math.abs(delta - 1) < 1e-16) return h
  }
  return NaN
}

/**
 * The regularised lower incomplete gamma function P(a, x) = γ(a, x)/Γ(a) = ∫₀ˣ t^{a−1} e^{−t} dt / Γ(a), for a > 0 and
 * x ≥ 0: the cdf of a Gamma(a, 1) variable. NaN if the series or fraction fails to converge (it converges for all
 * a below about 10¹⁰).
 */
export function regularisedGammaP(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return 0
  if (x === Infinity) return 1
  if (x < a + 1) return Math.exp(logGammaPrefactor(a, x)) * gammaSeries(a, x)
  return 1 - Math.exp(logGammaPrefactor(a, x)) * gammaFraction(a, x)
}

/** The regularised upper incomplete gamma function Q(a, x) = 1 − P(a, x), accurate in the upper tail. */
export function regularisedGammaQ(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return 1
  if (x === Infinity) return 0
  if (x < a + 1) return 1 - Math.exp(logGammaPrefactor(a, x)) * gammaSeries(a, x)
  return Math.exp(logGammaPrefactor(a, x)) * gammaFraction(a, x)
}

/** The Gamma(a, 1) density xᵃ⁻¹ e^{−x} / Γ(a), the x-derivative of P(a, x). */
export function gammaDensity(a: number, x: number): number {
  if (x < 0) return 0
  if (x === 0) return a === 1 ? 1 : a < 1 ? Infinity : 0
  return Math.exp(logGammaPrefactor(a, x)) / x
}

/**
 * log P(a, x). The lower tail is summed in log space (the prefactor's logarithm plus the series' logarithm), so it
 * keeps its relative accuracy where P underflows; above the series' range it is log1p(−Q), accurate where P ≈ 1.
 */
export function logRegularisedGammaP(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return -Infinity
  if (x === Infinity) return 0
  if (x < a + 1) return logGammaPrefactor(a, x) + Math.log(gammaSeries(a, x))
  return Math.log1p(-Math.exp(logGammaPrefactor(a, x)) * gammaFraction(a, x))
}

/** log Q(a, x): the upper tail in log space (the continued fraction's logarithm), and log1p(−P) below it. */
export function logRegularisedGammaQ(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return 0
  if (x === Infinity) return -Infinity
  if (x < a + 1) return Math.log1p(-Math.exp(logGammaPrefactor(a, x)) * gammaSeries(a, x))
  return logGammaPrefactor(a, x) + Math.log(gammaFraction(a, x))
}

/**
 * The x with P(a, x) = p, for a > 0 and p in [0, 1] (scipy.special.gammaincinv): the p-quantile of Gamma(a, 1).
 * Above p = ½ it solves Q(a, x) = 1 − p instead (1 − p is exact there), so upper quantiles keep their relative
 * accuracy. NaN for invalid arguments.
 */
export function regularisedGammaPInverse(a: number, p: number): number {
  if (!(a > 0 && p >= 0 && p <= 1)) return NaN
  if (p === 0) return 0
  if (p === 1) return Infinity
  return p <= 0.5 ? gammaInverse(a, Math.log(p), false) : gammaInverse(a, Math.log1p(-p), true)
}

/**
 * The x with Q(a, x) = q, for a > 0 and q in [0, 1] (scipy.special.gammainccinv): the inverse survival function of
 * Gamma(a, 1). Relative accuracy about 1e-14 for q down to the smallest normal double.
 */
export function regularisedGammaQInverse(a: number, q: number): number {
  if (!(a > 0 && q >= 0 && q <= 1)) return NaN
  if (q === 0) return Infinity
  if (q === 1) return 0
  return q <= 0.5 ? gammaInverse(a, Math.log(q), true) : gammaInverse(a, Math.log1p(-q), false)
}

/**
 * Solve log P(a, x) = logTarget (or log Q when `upper`) by Newton's method in s = log x, where both tails are close
 * to linear: log P ≈ a s − log Γ(a + 1) as x → 0, and log Q ≈ −eˢ for large x. d log P/ds = x·density/P and
 * d log Q/ds = −x·density/Q, with x·density = the prefactor xᵃ e^{−x}/Γ(a). Every step stays inside a bracket of the
 * root; a step that would leave it bisects (or moves by one unit of s while the bracket is open on that side). The
 * start is Wilson and Hilferty's cube-root normal approximation, or the lower tail's leading term for small a.
 */
function gammaInverse(a: number, logTarget: number, upper: boolean): number {
  const logF = upper ? logRegularisedGammaQ : logRegularisedGammaP
  // The standard normal deviate at the same probability: z for P = p, −z for Q = q.
  const zTail = normalQuantile(Math.exp(logTarget))
  const z = upper ? -zTail : zTail
  const cube = 1 - 1 / (9 * a) + z / (3 * Math.sqrt(a))
  let s: number
  if (a >= 1 && cube > 0) s = Math.log(a) + 3 * Math.log(cube)
  else if (!upper) s = (logTarget + logGamma(a + 1)) / a
  else s = Math.log(Math.max(1, -logTarget - logGamma(a)))
  if (!Number.isFinite(s)) s = Math.log(a)
  let lo = -Infinity
  let hi = Infinity
  for (let i = 0; i < 300; i++) {
    const x = Math.exp(s)
    const value = logF(a, x)
    const g = value - logTarget
    if (g === 0) return x
    // P increases and Q decreases in x: the root lies above s when log P is short, or log Q is long.
    if (upper ? g > 0 : g < 0) lo = s
    else hi = s
    const slope = (upper ? -1 : 1) * Math.exp(logGammaPrefactor(a, x) - value)
    let next = s - g / slope
    if (!(next > lo && next < hi) || !Number.isFinite(next)) {
      if (Number.isFinite(lo) && Number.isFinite(hi)) next = 0.5 * (lo + hi)
      else next = Number.isFinite(lo) ? lo + 1 : hi - 1
    }
    // Newton converges quadratically: once a step is below 1e-10 in log x, one more step reaches rounding level.
    if (Math.abs(next - s) < 1e-10) {
      const x2 = Math.exp(next)
      const v2 = logF(a, x2)
      const slope2 = (upper ? -1 : 1) * Math.exp(logGammaPrefactor(a, x2) - v2)
      const last = next - (v2 - logTarget) / slope2
      return Math.exp(Number.isFinite(last) ? last : next)
    }
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi - lo < 1e-15 * Math.max(1, Math.abs(s))) return Math.exp(next)
    s = next
  }
  return Math.exp(s)
}
