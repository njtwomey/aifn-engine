/**
 * Modified Bessel functions of the first kind of orders 0 and 1, I₀ and I₁, and the stable forms log I₀(x) and the
 * ratio A(x) = I₁(x)/I₀(x) that the von Mises distribution and the Kaiser window need.
 *
 * Method (Abramowitz and Stegun, 1964, 9.6.10 and 9.7.1): the power series Σₖ (x/2)^{2k+ν} / (k! (k+ν)!) for |x| ≤ 30,
 * whose terms are all positive (no cancellation), and the asymptotic expansion
 * I_ν(x) ≈ eˣ / √(2πx) · Σₖ (−1)ᵏ Πⱼ₌₁ᵏ (4ν² − (2j − 1)²) / (k! (8x)ᵏ) beyond, whose terms shrink until k ≈ 2x and
 * are below 1e-17 of the sum long before. Both are computed scaled by e^{−|x|} so nothing overflows until the final
 * product. I₀ is even and I₁ odd.
 */

/** e^{−x} I_ν(x) for ν ∈ {0, 1} and x ≥ 0. */
function besselIScaled(nu: 0 | 1, x: number): number {
  if (x === 0) return nu === 0 ? 1 : 0
  if (x <= 30) {
    const q = (x * x) / 4
    let term = nu === 0 ? 1 : x / 2
    let total = term
    for (let k = 1; k < 500; k++) {
      term *= q / (k * (k + nu))
      total += term
      if (term < 1e-17 * total) break
    }
    return total * Math.exp(-x)
  }
  const mu = 4 * nu * nu
  let term = 1
  let total = 1
  for (let k = 1; k < 100; k++) {
    const next = (term * -(mu - (2 * k - 1) ** 2)) / (k * 8 * x)
    if (Math.abs(next) >= Math.abs(term)) break
    term = next
    total += term
    if (Math.abs(term) < 1e-17 * Math.abs(total)) break
  }
  return total / Math.sqrt(2 * Math.PI * x)
}

/** I₀(x) for real x (even); +∞ beyond about |x| = 713. */
export function besselI0(x: number): number {
  if (Number.isNaN(x)) return NaN
  const a = Math.abs(x)
  return a === Infinity ? Infinity : besselIScaled(0, a) * Math.exp(a)
}

/** I₁(x) for real x (odd); ±∞ beyond about |x| = 713. */
export function besselI1(x: number): number {
  if (Number.isNaN(x)) return NaN
  const a = Math.abs(x)
  const v = a === Infinity ? Infinity : besselIScaled(1, a) * Math.exp(a)
  return x < 0 ? -v : v
}

/** dI₁/dx = I₀(x) − I₁(x)/x, with the limit 1/2 at x = 0. */
export function besselI1Derivative(x: number): number {
  return x === 0 ? 0.5 : besselI0(x) - besselI1(x) / x
}

/** log I₀(x) for x ≥ 0 (NaN below), without overflow. */
export function logBesselI0(x: number): number {
  if (!(x >= 0)) return NaN
  if (x < 1) {
    // log1p of the series' tail Σₖ≥₁ (x²/4)ᵏ/(k!)², so log I₀ ≈ x²/4 keeps its relative accuracy near 0 (x + log of
    // the scaled value would cancel).
    const q = (x * x) / 4
    let term = 1
    let tail = 0
    for (let k = 1; k < 50; k++) {
      term *= q / (k * k)
      tail += term
      if (term < 1e-17 * tail) break
    }
    return Math.log1p(tail)
  }
  return x + Math.log(besselIScaled(0, x))
}

/** A(x) = I₁(x)/I₀(x) for x ≥ 0 (NaN below): the mean resultant length of a von Mises distribution. */
export function besselRatio(x: number): number {
  if (!(x >= 0)) return NaN
  return besselIScaled(1, x) / besselIScaled(0, x)
}

/** A′(x) = 1 − A(x)/x − A(x)², with the limit 1/2 at x = 0. */
export function besselRatioDerivative(x: number, a: number): number {
  return x === 0 ? 0.5 : 1 - a / x - a * a
}
