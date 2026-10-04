/**
 * The gamma function family: $\log \Gamma$, $\Gamma$, digamma $\psi$, trigamma $\psi_1$ and polygamma $\psi^{(n)}$, $\log B$, $\log n!$, $\log \binom{n}{k}$,
 * and the regularised incomplete gamma functions $P$ and $Q$.
 *
 * Method. For $x \ge 10$, $\log \Gamma$ comes from Stirling's series (A&S 6.1.40) with eight Bernoulli terms, written as
 * $\log \Gamma(x) = (x - 1/2) \log x - x + \frac{1}{2} \log 2\pi + \delta(x)$ so that the small correction $\delta$ can be reused where large terms would
 * otherwise cancel ($\log B$, $\log \binom{n}{k}$, the incomplete gamma prefactor). Smaller $x$ are shifted up by the recurrence
 * $\Gamma(x + 1) = x\Gamma(x)$; negative $x$ use the reflection formula $\Gamma(x)\Gamma(1 - x) = \pi / \sin \pi x$. $\psi$ and $\psi^{(n)}$ use the same shift
 * and their asymptotic series (A&S 6.3.18, 6.4.11). $P$ and $Q$ use the power series and continued fraction of Press et
 * al., Numerical Recipes, 3rd ed., §6.2; $\log P$ and $\log Q$ sum the same series in log space, so neither underflows in its
 * own tail. The inverses solve $\log P = \log p$ (or $\log Q = \log q$ above the median) by safeguarded Newton steps in $\log x$.
 */

import { normalQuantile } from './normal'
import { log1pmx } from './stable'

const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI)
/** Stirling-series coefficients B₂ₖ / (2k(2k − 1)), k = 1…8. */
const STIRLING = [1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188, -691 / 360360, 1 / 156, -3617 / 122400]
/** Bernoulli numbers B₂ₖ, k = 1…9. */
const BERNOULLI = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510, 43867 / 798]

/**
 * $\delta(x) = \log \Gamma(x) - [(x - 1/2) \log x - x + \frac{1}{2} \log 2\pi]$, for $x \ge 10$ (Stirling's series; error below $10^{-17}$).
 *
 * @param x - Real argument $x \ge 10$.
 * @returns Stirling correction value $\delta(x)$.
 */
export function stirlingCorrection(x: number): number {
  const r = 1 / (x * x)
  let s = STIRLING[STIRLING.length - 1]
  for (let i = STIRLING.length - 2; i >= 0; i--) s = s * r + STIRLING[i]
  return s / x
}

/**
 * $\sin(\pi x)$, exact at integers and accurate for large $|x|$ (the argument is reduced modulo 2 exactly first).
 *
 * @param x - Real argument.
 * @returns Value of $\sin(\pi x)$.
 */
export function sinPi(x: number): number {
  let r = x % 2 // exact
  if (r < -1) r += 2
  else if (r > 1) r -= 2
  if (r === 0 || r === 1 || r === -1) return 0
  if (r === 0.5) return 1
  if (r === -0.5) return -1
  return Math.sin(Math.PI * r)
}

/**
 * $\pi / \tan(\pi x)$, for the digamma reflection; $\pm\infty$ at integers.
 *
 * @param x - Real argument.
 * @returns Value of $\pi \cot(\pi x)$.
 */
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
 * $\log |\Gamma(x)|$. $+\infty$ at the poles $x = 0, -1, -2, \dots$. Relative error about $10^{-15}$ away from the zeros at $x = 1$ and $x = 2$,
 * where the error is about $2 \cdot 10^{-15}$ absolute; exact (to rounding of $\log n!$) at integers.
 *
 * @param x - Real argument.
 * @returns Natural logarithm of $|\Gamma(x)|$.
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

/**
 * The gamma function $\Gamma(x)$. $+\infty$ above $x \approx 171.62$; `NaN` at the poles $x = 0, -1, -2, \dots$. Relative error about $10^{-14}$.
 *
 * @param x - Real argument.
 * @returns Value of $\Gamma(x)$.
 */
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
 * The digamma function $\psi(x) = \frac{\mathrm{d}}{\mathrm{d}x} \log \Gamma(x)$. `NaN` at the poles $x = 0, -1, -2, \dots$. Absolute error about $10^{-15}$ near its
 * zero $x_0 \approx 1.4616$, relative error about $10^{-15}$ elsewhere.
 *
 * @param x - Real argument.
 * @returns Value of $\psi(x)$.
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
 * The polygamma function $\psi^{(n)}(x) = \frac{\mathrm{d}^{n+1}}{\mathrm{d}x^{n+1}} \log \Gamma(x)$ for integer $n \ge 1$ and $x > 0$ (`NaN` otherwise; use
 * {@link trigamma} for negative $x$ with $n = 1$). Recurrence to $x \ge 20 + n$, then A&S 6.4.11.
 *
 * @param n - Derivative order (integer $n \ge 1$).
 * @param x - Evaluation point $x > 0$.
 * @returns Value of $\psi^{(n)}(x)$.
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

/**
 * The trigamma function $\psi_1(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2} \log \Gamma(x)$, for all real $x$ except the poles (reflection for $x < 0$).
 *
 * @param x - Real argument.
 * @returns Value of $\psi_1(x)$.
 */
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
 * $\log B(a, b) = \log \Gamma(a) + \log \Gamma(b) - \log \Gamma(a + b)$ for $a, b > 0$. When $a$ or $b$ is large the three $\log \Gamma$ terms are
 * combined through their Stirling corrections, so the result stays accurate where they would cancel.
 *
 * @param a - First parameter $a > 0$.
 * @param b - Second parameter $b > 0$.
 * @returns Natural logarithm of the beta function $B(a, b)$.
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

/**
 * $\log n! = \log \Gamma(n + 1)$ for $n \ge 0$ (non-integer $n$ uses the gamma function).
 *
 * @param n - Non-negative number $n \ge 0$.
 * @returns Natural logarithm of $n!$.
 */
export function logFactorial(n: number): number {
  if (Number.isInteger(n) && n >= 0 && n <= 170) return LOG_FACTORIAL[n]
  return logGamma(n + 1)
}

/**
 * $\log$ of the binomial coefficient $\binom{n}{k}$ for $0 \le k \le n$ ($-\infty$ outside; real arguments allowed), computed as
 * $-\log(n + 1) - \log B(n - k + 1, k + 1)$ so it is accurate even for $n = 10^9$ and small $k$.
 *
 * @param n - Total items $n$.
 * @param k - Chosen items $k$.
 * @returns Natural logarithm of $\binom{n}{k}$.
 */
export function logChoose(n: number, k: number): number {
  if (k < 0 || k > n) return -Infinity
  if (k === 0 || k === n) return 0
  return -Math.log1p(n) - logBeta(n - k + 1, k + 1)
}

/**
 * $\log$ of the incomplete gamma prefactor $x^a e^{-x} / \Gamma(a)$. For $a \ge 10$ it is written as
 * $a \cdot \operatorname{log1pmx}((x - a)/a) + \frac{1}{2} \log(a/2\pi) - \delta(a)$ when $x$ is near $a$, which avoids the cancellation of $a \log x - x$ against $\log \Gamma(a)$.
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Evaluation point $x \ge 0$.
 * @returns Logarithmic prefactor.
 */
function logGammaPrefactor(a: number, x: number): number {
  // Far from x = a there is no cancellation, and (x − a)/a near −1 would lose accuracy inside log1p.
  if (a < 10 || Math.abs(x - a) > 0.5 * a) return a * Math.log(x) - x - logGamma(a)
  return a * log1pmx((x - a) / a) + 0.5 * Math.log(a / (2 * Math.PI)) - stirlingCorrection(a)
}

const MAX_ITER = 100_000
const TINY = 1e-300

/**
 * Series $\sum_n x^n / (a(a+1)\cdots(a+n))$ for $P(a, x) / \text{prefactor}$ (NR3 §6.2). `NaN` if it has not converged.
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Evaluation point $x \ge 0$.
 * @returns Series sum.
 */
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

/**
 * Continued fraction for $Q(a, x) / \text{prefactor}$, modified Lentz (NR3 §6.2). `NaN` if it has not converged.
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Evaluation point $x \ge 0$.
 * @returns Fraction value.
 */
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
 * The regularised lower incomplete gamma function $P(a, x) = \gamma(a, x)/\Gamma(a) = \int_0^x t^{a-1} e^{-t}\,\mathrm{d}t / \Gamma(a)$, for $a > 0$ and
 * $x \ge 0$: the cdf of a $\operatorname{Gamma}(a, 1)$ variable. `NaN` if the series or fraction fails to converge (it converges for all
 * $a$ below about $10^{10}$).
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Upper limit $x \ge 0$.
 * @returns Regularised lower incomplete gamma $P(a, x) \in [0, 1]$.
 */
export function regularisedGammaP(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return 0
  if (x === Infinity) return 1
  if (x < a + 1) return Math.exp(logGammaPrefactor(a, x)) * gammaSeries(a, x)
  return 1 - Math.exp(logGammaPrefactor(a, x)) * gammaFraction(a, x)
}

/**
 * The regularised upper incomplete gamma function $Q(a, x) = 1 - P(a, x)$, accurate in the upper tail.
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Lower limit $x \ge 0$.
 * @returns Regularised upper incomplete gamma $Q(a, x) \in [0, 1]$.
 */
export function regularisedGammaQ(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return 1
  if (x === Infinity) return 0
  if (x < a + 1) return 1 - Math.exp(logGammaPrefactor(a, x)) * gammaSeries(a, x)
  return Math.exp(logGammaPrefactor(a, x)) * gammaFraction(a, x)
}

/**
 * The $\operatorname{Gamma}(a, 1)$ density $x^{a-1} e^{-x} / \Gamma(a)$, the $x$-derivative of $P(a, x)$.
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Evaluation point $x \ge 0$.
 * @returns Probability density at $x$.
 */
export function gammaDensity(a: number, x: number): number {
  if (x < 0) return 0
  if (x === 0) return a === 1 ? 1 : a < 1 ? Infinity : 0
  return Math.exp(logGammaPrefactor(a, x)) / x
}

/**
 * $\log P(a, x)$. The lower tail is summed in log space (the prefactor's logarithm plus the series' logarithm), so it
 * keeps its relative accuracy where $P$ underflows; above the series' range it is $\operatorname{log1p}(-Q)$, accurate where $P \approx 1$.
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Evaluation point $x \ge 0$.
 * @returns Natural logarithm $\log P(a, x)$.
 */
export function logRegularisedGammaP(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return -Infinity
  if (x === Infinity) return 0
  if (x < a + 1) return logGammaPrefactor(a, x) + Math.log(gammaSeries(a, x))
  return Math.log1p(-Math.exp(logGammaPrefactor(a, x)) * gammaFraction(a, x))
}

/**
 * $\log Q(a, x)$: the upper tail in log space (the continued fraction's logarithm), and $\operatorname{log1p}(-P)$ below it.
 *
 * @param a - Shape parameter $a > 0$.
 * @param x - Evaluation point $x \ge 0$.
 * @returns Natural logarithm $\log Q(a, x)$.
 */
export function logRegularisedGammaQ(a: number, x: number): number {
  if (!(a > 0 && x >= 0)) return NaN
  if (x === 0) return 0
  if (x === Infinity) return -Infinity
  if (x < a + 1) return Math.log1p(-Math.exp(logGammaPrefactor(a, x)) * gammaSeries(a, x))
  return logGammaPrefactor(a, x) + Math.log(gammaFraction(a, x))
}

/**
 * The $x$ with $P(a, x) = p$, for $a > 0$ and $p \in [0, 1]$ (`scipy.special.gammaincinv`): the $p$-quantile of $\operatorname{Gamma}(a, 1)$.
 * Above $p = 1/2$ it solves $Q(a, x) = 1 - p$ instead ($1 - p$ is exact there), so upper quantiles keep their relative
 * accuracy. `NaN` for invalid arguments.
 *
 * @param a - Shape parameter $a > 0$.
 * @param p - Cumulative probability $p \in [0, 1]$.
 * @returns Quantile value $x$ such that $P(a, x) = p$.
 */
export function regularisedGammaPInverse(a: number, p: number): number {
  if (!(a > 0 && p >= 0 && p <= 1)) return NaN
  if (p === 0) return 0
  if (p === 1) return Infinity
  return p <= 0.5 ? gammaInverse(a, Math.log(p), false) : gammaInverse(a, Math.log1p(-p), true)
}

/**
 * The $x$ with $Q(a, x) = q$, for $a > 0$ and $q \in [0, 1]$ (`scipy.special.gammainccinv`): the inverse survival function of
 * $\operatorname{Gamma}(a, 1)$. Relative accuracy about $10^{-14}$ for $q$ down to the smallest normal double.
 *
 * @param a - Shape parameter $a > 0$.
 * @param q - Survival probability $q \in [0, 1]$.
 * @returns Value $x$ such that $Q(a, x) = q$.
 */
export function regularisedGammaQInverse(a: number, q: number): number {
  if (!(a > 0 && q >= 0 && q <= 1)) return NaN
  if (q === 0) return Infinity
  if (q === 1) return 0
  return q <= 0.5 ? gammaInverse(a, Math.log(q), true) : gammaInverse(a, Math.log1p(-q), false)
}

/**
 * Solve $\log P(a, x) = \text{logTarget}$ (or $\log Q$ when `upper`) by Newton's method in $s = \log x$, where both tails are close
 * to linear: $\log P \approx a s - \log \Gamma(a + 1)$ as $x \to 0$, and $\log Q \approx -e^s$ for large $x$. $\mathrm{d}\log P/\mathrm{d}s = x \cdot \text{density}/P$ and
 * $\mathrm{d}\log Q/\mathrm{d}s = -x \cdot \text{density}/Q$, with $x \cdot \text{density} = x^a e^{-x}/\Gamma(a)$. Every step stays inside a bracket of the
 * root; a step that would leave it bisects (or moves by one unit of $s$ while the bracket is open on that side). The
 * start is Wilson and Hilferty's cube-root normal approximation, or the lower tail's leading term for small $a$.
 *
 * @param a - Shape parameter $a > 0$.
 * @param logTarget - Target log-probability.
 * @param upper - Whether solving the upper tail ($\log Q$).
 * @returns Quantile $x$.
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
