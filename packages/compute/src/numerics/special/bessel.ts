/**
 * Modified Bessel functions of the first kind of orders 0 and 1, $I_0$ and $I_1$, and the stable forms $\log I_0(x)$ and the
 * ratio $A(x) = I_1(x)/I_0(x)$ that the von Mises distribution and the Kaiser window need.
 *
 * Method (Abramowitz and Stegun, 1964, 9.6.10 and 9.7.1): the power series $\sum_k (x/2)^{2k+\nu} / (k! (k+\nu)!)$ for $|x| \le 30$,
 * whose terms are all positive (no cancellation), and the asymptotic expansion
 * $I_\nu(x) \approx e^x / \sqrt{2\pi x} \cdot \sum_k (-1)^k \prod_{j=1}^k (4\nu^2 - (2j - 1)^2) / (k! (8x)^k)$ beyond, whose terms shrink until $k \approx 2x$ and
 * are below $10^{-17}$ of the sum long before. Both are computed scaled by $e^{-|x|}$ so nothing overflows until the final
 * product. $I_0$ is even and $I_1$ odd.
 */

/**
 * $e^{-x} I_\nu(x)$ for $\nu \in \{0, 1\}$ and $x \ge 0$.
 *
 * @param nu - Bessel order ($\nu \in \{0, 1\}$).
 * @param x - Non-negative evaluation point.
 * @returns Exponentially scaled modified Bessel function value.
 */
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

/**
 * $I_0(x)$ for real $x$ (even); $+\infty$ beyond about $|x| = 713$.
 *
 * @param x - Real argument.
 * @returns Value of the zero-order modified Bessel function of the first kind.
 */
export function besselI0(x: number): number {
  if (Number.isNaN(x)) return NaN
  const a = Math.abs(x)
  return a === Infinity ? Infinity : besselIScaled(0, a) * Math.exp(a)
}

/**
 * $I_1(x)$ for real $x$ (odd); $\pm\infty$ beyond about $|x| = 713$.
 *
 * @param x - Real argument.
 * @returns Value of the first-order modified Bessel function of the first kind.
 */
export function besselI1(x: number): number {
  if (Number.isNaN(x)) return NaN
  const a = Math.abs(x)
  const v = a === Infinity ? Infinity : besselIScaled(1, a) * Math.exp(a)
  return x < 0 ? -v : v
}

/**
 * $\mathrm{d}I_1/\mathrm{d}x = I_0(x) - I_1(x)/x$, with the limit $1/2$ at $x = 0$.
 *
 * @param x - Real evaluation point.
 * @returns First derivative of $I_1(x)$.
 */
export function besselI1Derivative(x: number): number {
  return x === 0 ? 0.5 : besselI0(x) - besselI1(x) / x
}

/**
 * $\log I_0(x)$ for $x \ge 0$ (`NaN` below), without overflow.
 *
 * @param x - Non-negative argument.
 * @returns Natural logarithm of $I_0(x)$.
 */
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

/**
 * $A(x) = I_1(x)/I_0(x)$ for $x \ge 0$ (`NaN` below): the mean resultant length of a von Mises distribution.
 *
 * @param x - Non-negative argument.
 * @returns Ratio $I_1(x) / I_0(x)$.
 */
export function besselRatio(x: number): number {
  if (!(x >= 0)) return NaN
  return besselIScaled(1, x) / besselIScaled(0, x)
}

/**
 * $A'(x) = 1 - A(x)/x - A(x)^2$, with the limit $1/2$ at $x = 0$.
 *
 * @param x - Argument $x \ge 0$.
 * @param a - Precomputed ratio $A(x) = I_1(x)/I_0(x)$.
 * @returns Derivative $A'(x)$.
 */
export function besselRatioDerivative(x: number, a: number): number {
  return x === 0 ? 0.5 : 1 - a / x - a * a
}
