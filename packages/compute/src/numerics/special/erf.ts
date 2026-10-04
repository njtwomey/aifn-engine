/**
 * The error function family: erf, erfc, the scaled erfcx(x) = e^{x²} erfc(x), and log erfc.
 *
 * Method. For |x| < 1.5, erf comes from its everywhere-positive power series
 *   erf(x) = (2/√π) x e^{−x²} Σₙ (2x²)ⁿ / (1·3·5···(2n+1))            (Abramowitz and Stegun 1964, 7.1.6),
 * which has no cancellation. For x ≥ 1.5, erfcx comes from the continued fraction
 *   erfcx(x) = (1/√π) · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + …))))         (A&S 7.1.14),
 * evaluated by the modified Lentz method (Press et al., Numerical Recipes, 3rd ed., §5.2). erfc(x) = e^{−x²} erfcx(x)
 * keeps full relative accuracy in the upper tail, down to the underflow of erfc near x = 26.5, and log erfc(x) =
 * log erfcx(x) − x² stays accurate far beyond it. e^{±x²} is computed with x² split into an exactly representable
 * part (as in Cody 1969, "Rational Chebyshev approximations for the error function", Math. Comp. 23), so the rounding
 * of x² is not magnified by the exponential.
 */

const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI)
const INV_SQRT_PI = 1 / Math.sqrt(Math.PI)
/** Below this, the series for erf; above it, the continued fraction for erfcx. */
const SPLIT = 1.5

/** e^{s·x²} for s = ±1, with x² split as hi² + (x − hi)(x + hi), hi = x rounded to 1/16 (hi² is exact). */
export function expSquare(x: number, sign: 1 | -1): number {
  const hi = Math.trunc(x * 16) / 16
  const lo = x - hi
  return Math.exp(sign * hi * hi) * Math.exp(sign * lo * (x + hi))
}

/** The positive series (2/√π) x e^{−x²} Σ (2x²)ⁿ / (2n+1)!!, for |x| < SPLIT. */
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

/** erfcx(x) for x ≥ SPLIT by the continued fraction (A&S 7.1.14), modified Lentz. */
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

/** The error function erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt. Relative error about 1e-15 for all x. */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN
  const a = Math.abs(x)
  if (a < SPLIT) return erfSeries(x)
  if (a > 6) return Math.sign(x) // erfc(6) < 2.2e-17
  return Math.sign(x) * (1 - expSquare(a, -1) * erfcxFraction(a))
}

/** The complementary error function erfc(x) = 1 − erf(x), with full relative accuracy in the upper tail. */
export function erfc(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x < 0) return 2 - erfc(-x)
  if (x < SPLIT) return 1 - erfSeries(x)
  if (x > 27.3) return 0 // below the smallest subnormal
  return expSquare(x, -1) * erfcxFraction(x)
}

/**
 * The scaled complementary error function erfcx(x) = e^{x²} erfc(x). It decays like 1/(x√π) for large x and never
 * underflows there; for x below about −26.6 it overflows to +∞.
 */
export function erfcx(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x < 0) return x < -26.7 ? Infinity : 2 * expSquare(x, 1) - erfcx(-x)
  if (x < SPLIT) return expSquare(x, 1) * (1 - erfSeries(x))
  return erfcxFraction(x)
}

/** log erfc(x), accurate far into the upper tail (log erfc(x) ≈ −x² − log(x√π) for large x). */
export function logErfc(x: number): number {
  if (Number.isNaN(x)) return NaN
  if (x === Infinity) return -Infinity
  // log1p(−erf x) keeps the relative accuracy of log erfc near x = 0, where log(1 − erf x) and log 2 + log1p(…) cancel.
  if (x < SPLIT) return Math.log1p(-erf(x))
  return Math.log(erfcxFraction(x)) - x * x
}
