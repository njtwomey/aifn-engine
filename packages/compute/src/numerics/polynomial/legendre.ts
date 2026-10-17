/**
 * The Legendre polynomials $P_0, \dots, P_{N-1}$ evaluated at points, by Bonnet's three-term recurrence
 * $(n + 1) P_{n+1}(z) = (2n + 1) z P_n(z) - n P_{n-1}(z)$ from $P_0 = 1$ and $P_1 = z$ (Abramowitz and Stegun,
 * 1964, 22.7.10). They are orthogonal on $[-1, 1]$ with $\int_{-1}^{1} P_m P_n \,dz = 2\delta_{mn}/(2n + 1)$, so
 * $\sqrt{2n + 1}\,P_n$ is orthonormal under the uniform measure $dz/2$; HiPPO's memories are coefficients in that
 * basis.
 * The nodes and weights of Gauss–Legendre quadrature are `gaussLegendre` of `aifn-compute/numerics/quadrature`.
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { add, div, mul, shapeOfValue, stack, sub, zeros, type Value } from 'aifn-compute/foundation/tensor'

/**
 * The Legendre polynomials $P_0(z), \dots, P_{N-1}(z)$ at every point of `z`, by Bonnet's recurrence. Written with
 * elementwise primitives, so it broadcasts over `z` and differentiates in it.
 *
 * @param order The number of polynomials $N$ (degrees 0 to $N - 1$).
 * @param z The points: a number or a tensor of any shape, normally in $[-1, 1]$.
 * @returns A tensor of shape `[...shape(z), N]`: the last axis holds $P_0(z), \dots, P_{N-1}(z)$ (a vector of $N$ for
 *   a number `z`).
 *
 * @example The first four polynomials at z = 0.5
 * print('P(0.5) =', legendre(4, 0.5))
 *
 * @example At the ends, P_n(1) = 1 and P_n(-1) = (-1)^n
 * print('P(-1), P(1) =', legendre(5, tensor([-1, 1])))
 */
export function legendre(order: Size, z: Value): Value {
  if (order <= 0) return zeros([...shapeOfValue(z), 0])
  const p: Value[] = [add(mul(0, z), 1)]
  if (order > 1) p.push(z)
  for (let n = 1; n + 1 < order; n++) p.push(div(sub(mul(mul(2 * n + 1, z), p[n]), mul(n, p[n - 1])), n + 1))
  return stack(p, -1)
}
