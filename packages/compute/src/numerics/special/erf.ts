/**
 * The error function family: $\operatorname{erf}$, $\operatorname{erfc}$, the scaled $\operatorname{erfcx}(x) = e^{x^2} \operatorname{erfc}(x)$, and $\log \operatorname{erfc}$.
 *
 * Method. For $|x| < 1.5$, $\operatorname{erf}$ comes from its everywhere-positive power series
 *   $\operatorname{erf}(x) = (2/\sqrt{\pi}) x e^{-x^2} \sum_n (2x^2)^n / (1 \cdot 3 \cdot 5 \cdots (2n+1))$ (Abramowitz and Stegun 1964, 7.1.6),
 * which has no cancellation. For $x \ge 1.5$, $\operatorname{erfcx}$ comes from the continued fraction
 *   $\operatorname{erfcx}(x) = (1/\sqrt{\pi}) \cdot 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + \dots))))$ (A&S 7.1.14),
 * evaluated by the modified Lentz method (Press et al., Numerical Recipes, 3rd ed., §5.2). $\operatorname{erfc}(x) = e^{-x^2} \operatorname{erfcx}(x)$
 * keeps full relative accuracy in the upper tail, down to the underflow of $\operatorname{erfc}$ near $x = 26.5$, and $\log \operatorname{erfc}(x) =
 * \log \operatorname{erfcx}(x) - x^2$ stays accurate far beyond it. $e^{\pm x^2}$ is computed with $x^2$ split into an exactly representable
 * part (as in Cody 1969, "Rational Chebyshev approximations for the error function", Math. Comp. 23), so the rounding
 * of $x^2$ is not magnified by the exponential.
 */

const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI)
const INV_SQRT_PI = 1 / Math.sqrt(Math.PI)
/** Below this, the series for erf; above it, the continued fraction for erfcx. */
const SPLIT = 1.5

/**
 * $e^{s x^2}$ for $s = \pm 1$, with $x^2$ split as $hi^2 + (x - hi)(x + hi)$, $hi = x$ rounded to $1/16$ ($hi^2$ is exact).
 *
 * @param x - Real evaluation point.
 * @param sign - Exponent sign factor ($+1$ or $-1$).
 * @returns $e^{\operatorname{sign} \cdot x^2}$.
 */
export function expSquare(x: number, sign: 1 | -1): number {
  const hi = Math.trunc(x * 16) / 16
  const lo = x - hi
  return Math.exp(sign * hi * hi) * Math.exp(sign * lo * (x + hi))
}

/**
 * The positive series $(2/\sqrt{\pi}) x e^{-x^2} \sum (2x^2)^n / (2n+1)!!$, for $|x| < \text{SPLIT}$.
 *
 * @param x - Real evaluation point with $|x| < 1.5$.
 * @returns Error function value.
 */
function erfSeries(x: number): number {
  const x2 = 2 * x * x
  let term = 1
  let sum = 1
  for (let n = 1; n < 200; n++) {
    term *= x2 / (2 * n + 1)
    sum += term
    if (term < 1e-17 * sum) break
  }
  return TWO_OVER_SQRT_PI * x * expSquare(x, -1) * sum
}

/**
 * $\operatorname{erfcx}(x)$ for $x \ge \text{SPLIT}$ by the continued fraction (A&S 7.1.14), modified Lentz.
 *
 * @param x - Evaluation point $x \ge 1.5$.
 * @returns Scaled complementary error function value.
 */
function erfcxFraction(x: number): number {
  let f = x
  let c = f
  let d = 0
  for (let j = 1; j < 1000; j++) {
    const a = j / 2
    d = x + a * d
    c = x + a / c
    d = 1 / d
    const delta = c * d
    f *= delta
    if (Math.abs(delta - 1) < 1e-16) break
  }
  return INV_SQRT_PI / f
}

/**
 * The error function $\operatorname{erf}(x) = (2/\sqrt{\pi}) \int_0^x e^{-t^2}\,\mathrm{d}t$. Relative error about $10^{-15}$ for all $x$.
 *
 * @param x - Real evaluation point.
 * @returns Value of $\operatorname{erf}(x) \in [-1, 1]$.
 */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN
  const a = Math.abs(x)
  if (a < SPLIT) return erfSeries(x)
  if (a > 6) return Math.sign(x) // erfc(6) < 2.2e-17
  return Math.sign(x) * (1 - expSquare(a, -1) * erfcxFraction(a))
}

/**
 * The complementary error function $\operatorname{erfc}(x) = 1 - \operatorname{erf}(x)$, with full relative accuracy in the upper tail.
 *
 * @param x - Real evaluation point.
 * @returns Value of $\operatorname{erfc}(x)$.
 */
export function erfc(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x < 0) return 2 - erfc(-x)
  if (x < SPLIT) return 1 - erfSeries(x)
  if (x > 27.3) return 0 // below the smallest subnormal
  return expSquare(x, -1) * erfcxFraction(x)
}

/**
 * The scaled complementary error function $\operatorname{erfcx}(x) = e^{x^2} \operatorname{erfc}(x)$. It decays like $1/(x\sqrt{\pi})$ for large $x$ and never
 * underflows there; for $x$ below about $-26.6$ it overflows to $+\infty$.
 *
 * @param x - Real evaluation point.
 * @returns Value of $\operatorname{erfcx}(x)$.
 */
export function erfcx(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x < 0) return x < -26.7 ? Infinity : 2 * expSquare(x, 1) - erfcx(-x)
  if (x < SPLIT) return expSquare(x, 1) * (1 - erfSeries(x))
  return erfcxFraction(x)
}

/**
 * $\log \operatorname{erfc}(x)$, accurate far into the upper tail ($\log \operatorname{erfc}(x) \approx -x^2 - \log(x\sqrt{\pi})$ for large $x$).
 *
 * @param x - Real evaluation point.
 * @returns Natural logarithm of the complementary error function.
 */
export function logErfc(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x === Infinity) return -Infinity
  // log1p(−erf x) keeps the relative accuracy of log erfc near x = 0, where log(1 − erf x) and log 2 + log1p(…) cancel.
  if (x < SPLIT) return Math.log1p(-erf(x))
  return Math.log(erfcxFraction(x)) - x * x
}
