/**
 * The expected maximum of $J$ independent standard normals, the size of the winner's curse when the best of $J$
 * equally good candidates is picked by a noisy score.
 *
 * With $\Phi$ the standard normal distribution function, $\max_j Z_j$ has distribution function $\Phi^J$, so
 * $$\expect\max_{1 \le j \le J} Z_j = \int_0^\infty \bigl(1 - \Phi(x)^J\bigr)\,dx - \int_0^\infty \Phi(-x)^J\,dx,$$
 * integrated numerically with $\Phi^J = \exp(J \log\Phi)$, so that large $J$ loses no precision. It grows like
 * $\sqrt{2\ln J}$, the maximal inequality's bound, which it never exceeds.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { integrate } from 'aifn-compute/numerics/quadrature'
import { normalLogCdf } from 'aifn-compute/numerics/special'

/**
 * $\expect\max_{1 \le j \le J} Z_j$ for $J$ independent standard normals, by adaptive Gauss–Kronrod quadrature of the
 * integral in the file's introduction (accurate to about $10^{-12}$). $J$ need not be an integer: the integral is the
 * mean of the distribution function $\Phi^J$. Throws `DomainError` for $J < 1$.
 *
 * @param J The number of normals $J \ge 1$.
 * @returns The expected maximum: 0 for $J = 1$, $1/\sqrt\pi$ for $J = 2$.
 *
 * @example Against the bound $\sqrt{2 \ln J}$
 * for (const J of [2, 10, 100, 1000]) {
 *   print('J =', J, ' E max =', expectedGaussianMaximum(J), ' bound =', Math.sqrt(2 * Math.log(J)))
 * }
 * print('1 / sqrt(pi) =', 1 / Math.sqrt(Math.PI))
 */
export function expectedGaussianMaximum(J: number): number {
  if (!(J >= 1 && Number.isFinite(J)))
    throw new DomainError('expectedGaussianMaximum', `expectedGaussianMaximum: J must be finite and >= 1, got ${J}`)
  const mode = Math.sqrt(2 * Math.log(Math.max(J, 2)))
  const f = (x: number) => -Math.expm1(J * normalLogCdf(x)) - Math.exp(J * normalLogCdf(-x))
  const r = integrate(f, 0, mode + 10, { atol: 1e-14, rtol: 1e-13, points: [mode], maxIntervals: 500 })
  return r.value
}
