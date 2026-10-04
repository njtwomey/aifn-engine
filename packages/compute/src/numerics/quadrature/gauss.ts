/**
 * Gaussian quadrature rules: nodes and weights of Gauss–Legendre (weight 1 on $[-1, 1]$), Gauss–Hermite (weight $e^{-x^2}$,
 * or $e^{-x^2/2}$ in the probabilists' form) and generalised Gauss–Laguerre (weight $x^\alpha e^{-x}$ on $[0, \infty)$). An $n$-point rule
 * integrates polynomials of degree up to $2n - 1$ exactly against its weight.
 *
 * Nodes are the roots of the orthogonal polynomial, found by Newton's method on its three-term recurrence from
 * asymptotic initial guesses (Press et al., 2007, "Numerical Recipes", 3rd ed., §4.6, `gauleg`, `gauher`, `gaulag`);
 * the weights follow from the derivative at each node. Nodes are returned in ascending order, as numpy's `leggauss`,
 * `hermgauss`, `hermegauss` and `laggauss` return them.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { logGamma } from 'aifn-compute/numerics/special'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'

/** A quadrature rule: $\int w(x) f(x)\,dx \approx \sum \text{weights}[i] \cdot f(\text{nodes}[i])$. */
export type QuadratureRule = {
  /** The quadrature evaluation nodes as a rank-1 tensor in ascending order. */
  nodes: Tensor
  /** The corresponding quadrature weights as a rank-1 tensor. */
  weights: Tensor
}

const MAX_NEWTON = 100

/**
 * Sort quadrature nodes in ascending order and construct a `QuadratureRule`.
 *
 * @param nodes Array of quadrature node locations.
 * @param weights Array of corresponding quadrature weights.
 * @returns A `QuadratureRule` holding nodes and weights as sorted 1D tensors.
 */
const rule = (nodes: number[], weights: number[]): QuadratureRule => {
  const order = nodes.map((_, i) => i).sort((a, b) => nodes[a] - nodes[b])
  return {
    nodes: fromData(
      Float64Array.from(order, (i) => nodes[i]),
      [nodes.length],
    ),
    weights: fromData(
      Float64Array.from(order, (i) => weights[i]),
      [nodes.length],
    ),
  }
}

/**
 * Check that the quadrature rule order $n$ is a positive integer, throwing `DomainError` if not.
 *
 * @param n Order or number of points.
 * @param where Caller name for error messages.
 */
function checkOrder(n: number, where: string) {
  if (!Number.isInteger(n) || n < 1) throw new DomainError(where, `${where}: n must be a positive integer, got ${n}`)
}

/**
 * The $n$-point Gauss–Legendre rule on $[a, b]$ (default $[-1, 1]$):
 * $\int_a^b f(x)\,dx \approx \sum w_i f(x_i)$, exact for polynomials of degree $\le 2n - 1$.
 * On $[a, b]$ the nodes are mapped linearly and the weights scaled by $(b - a)/2$.
 *
 * @param n Number of quadrature nodes ($n \ge 1$).
 * @param interval The integration interval $[a, b]$ (default $[-1, 1]$).
 * @returns A `QuadratureRule` with sorted nodes and weights.
 *
 * @example Compute nodes and weights of 3-point Gauss-Legendre rule
 * const { nodes, weights } = gaussLegendre(3)
 * print('nodes =', nodes)
 */
export function gaussLegendre(n: Size, interval: readonly [Scalar, Scalar] = [-1, 1]): QuadratureRule {
  checkOrder(n, 'gaussLegendre')
  const x = new Array<number>(n)
  const w = new Array<number>(n)
  const m = Math.floor((n + 1) / 2)
  for (let i = 1; i <= m; i++) {
    // Tricomi's approximation to the i-th largest root.
    let z = Math.cos((Math.PI * (i - 0.25)) / (n + 0.5))
    let pp = 0
    for (let its = 0; its < MAX_NEWTON; its++) {
      let p1 = 1
      let p2 = 0
      for (let j = 1; j <= n; j++) {
        const p3 = p2
        p2 = p1
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
      }
      // P_n'(z) from P_n and P_{n−1}.
      pp = (n * (z * p1 - p2)) / (z * z - 1)
      const z1 = z
      z = z1 - p1 / pp
      if (Math.abs(z - z1) <= 1e-15) break
    }
    x[i - 1] = -z
    x[n - i] = z
    w[i - 1] = w[n - i] = 2 / ((1 - z * z) * pp * pp)
  }
  if (n % 2 === 1) x[(n - 1) / 2] = 0
  const [a, b] = interval
  const half = (b - a) / 2
  const mid = (a + b) / 2
  return rule(
    x.map((xi) => mid + half * xi),
    w.map((wi) => half * wi),
  )
}

/**
 * The $n$-point Gauss–Hermite rule. Physicists' form (default):
 * $\int_{-\infty}^\infty e^{-x^2} f(x)\,dx \approx \sum w_i f(x_i)$, as numpy's `hermgauss`.
 * With `probabilists: true`: $\int_{-\infty}^\infty e^{-x^2/2} f(x)\,dx$, as `hermegauss` (weights sum to $\sqrt{2\pi}$).
 *
 * @param n Number of quadrature nodes ($n \ge 1$).
 * @param options Options controlling rule formulation.
 * @param options.probabilists Use probabilists' weight $e^{-x^2/2}$ instead of physicists' $e^{-x^2}$ (default false).
 * @returns A `QuadratureRule` with sorted nodes and weights.
 *
 * @example Compute 3-point Gauss-Hermite rule
 * const { nodes, weights } = gaussHermite(3)
 * print('nodes =', nodes)
 */
export function gaussHermite(n: Size, { probabilists = false }: { probabilists?: boolean } = {}): QuadratureRule {
  checkOrder(n, 'gaussHermite')
  const PIM4 = 0.7511255444649425 // π^{−1/4}
  const x = new Array<number>(n)
  const w = new Array<number>(n)
  const m = Math.floor((n + 1) / 2)
  let z = 0
  for (let i = 0; i < m; i++) {
    // Initial guesses for the largest roots, then extrapolation from the previous ones.
    if (i === 0) z = Math.sqrt(2 * n + 1) - 1.85575 * (2 * n + 1) ** -0.16667
    else if (i === 1) z -= (1.14 * n ** 0.426) / z
    else if (i === 2) z = 1.86 * z - 0.86 * x[0]
    else if (i === 3) z = 1.91 * z - 0.91 * x[1]
    else z = 2 * z - x[i - 2]
    let pp = 0
    for (let its = 0; its < MAX_NEWTON; its++) {
      // Orthonormal Hermite recurrence, which avoids overflow.
      let p1 = PIM4
      let p2 = 0
      for (let j = 1; j <= n; j++) {
        const p3 = p2
        p2 = p1
        p1 = z * Math.sqrt(2 / j) * p2 - Math.sqrt((j - 1) / j) * p3
      }
      pp = Math.sqrt(2 * n) * p2
      const z1 = z
      z = z1 - p1 / pp
      if (Math.abs(z - z1) <= 1e-15 * Math.max(1, Math.abs(z))) break
    }
    x[i] = z
    x[n - 1 - i] = -z
    w[i] = w[n - 1 - i] = 2 / (pp * pp)
  }
  if (n % 2 === 1) x[(n - 1) / 2] = 0
  if (!probabilists) return rule(x, w)
  return rule(
    x.map((xi) => Math.SQRT2 * xi),
    w.map((wi) => Math.SQRT2 * wi),
  )
}

/**
 * The $n$-point generalised Gauss–Laguerre rule:
 * $\int_0^\infty x^\alpha e^{-x} f(x)\,dx \approx \sum w_i f(x_i)$, $\alpha > -1$ (default 0, as numpy's
 * `laggauss`; scipy's `roots_genlaguerre` for $\alpha \neq 0$).
 *
 * @param n Number of quadrature nodes ($n \ge 1$).
 * @param options Options controlling polynomial exponent.
 * @param options.alpha Exponent $\alpha > -1$ in weight $x^\alpha e^{-x}$ (default 0).
 * @returns A `QuadratureRule` with sorted nodes and weights.
 *
 * @example Compute 3-point Gauss-Laguerre rule
 * const { nodes, weights } = gaussLaguerre(3)
 * print('nodes =', nodes)
 */
export function gaussLaguerre(n: Size, { alpha = 0 }: { alpha?: Scalar } = {}): QuadratureRule {
  checkOrder(n, 'gaussLaguerre')
  if (!(alpha > -1)) throw new DomainError('gaussLaguerre', `gaussLaguerre: alpha must exceed −1, got ${alpha}`)
  const x = new Array<number>(n)
  const w = new Array<number>(n)
  let z = 0
  for (let i = 0; i < n; i++) {
    if (i === 0) z = ((1 + alpha) * (3 + 0.92 * alpha)) / (1 + 2.4 * n + 1.8 * alpha)
    else if (i === 1) z += (15 + 6.25 * alpha) / (1 + 0.9 * alpha + 2.5 * n)
    else {
      const ai = i - 1
      z += (((1 + 2.55 * ai) / (1.9 * ai) + (1.26 * ai * alpha) / (1 + 3.5 * ai)) * (z - x[i - 2])) / (1 + 0.3 * alpha)
    }
    let pp = 0
    let p2 = 0
    for (let its = 0; its < MAX_NEWTON; its++) {
      let p1 = 1
      p2 = 0
      for (let j = 0; j < n; j++) {
        const p3 = p2
        p2 = p1
        p1 = ((2 * j + 1 + alpha - z) * p2 - (j + alpha) * p3) / (j + 1)
      }
      pp = (n * p1 - (n + alpha) * p2) / z
      const z1 = z
      z = z1 - p1 / pp
      if (Math.abs(z - z1) <= 1e-15 * Math.max(1, Math.abs(z))) break
    }
    x[i] = z
    w[i] = -Math.exp((logGamma(alpha + n) as number) - (logGamma(n) as number)) / (pp * n * p2)
  }
  return rule(x, w)
}

/**
 * $\expect[f(X)]$ for $X \sim \mathcal{N}(\mu, \sigma^2)$ by $n$-point Gauss–Hermite quadrature (default $n = 32$):
 * with $X = \mu + \sqrt{2}\sigma z$, $\expect f(X) = \pi^{-1/2} \int_{-\infty}^\infty e^{-z^2} f(\mu + \sqrt{2}\sigma z)\,dz$.
 * Exact for polynomials of degree $\le 2n - 1$.
 *
 * @param f Function to take expectation of.
 * @param mean Distribution mean $\mu$.
 * @param sd Standard deviation $\sigma$.
 * @param options Options specifying quadrature order.
 * @param options.n Number of Gauss-Hermite quadrature points (default 32).
 * @returns The expected value $\expect[f(X)]$.
 *
 * @example Expectation of X^2 for standard normal
 * print('E[X^2] =', normalExpectation((x) => x * x, 0, 1))
 */
export function normalExpectation(
  f: (x: number) => number,
  mean: number,
  sd: number,
  { n = 32 }: { n?: number } = {},
): number {
  const { nodes, weights } = gaussHermite(n)
  let total = 0
  for (let i = 0; i < n; i++) total += weights.data[i] * f(mean + Math.SQRT2 * sd * nodes.data[i])
  return total / Math.sqrt(Math.PI)
}

/**
 * $\int_a^b f(x)\,dx$ by the $n$-point Gauss–Legendre rule (default $n = 20$).
 *
 * @param f Integrand function.
 * @param a Lower integration bound.
 * @param b Upper integration bound.
 * @param options Options specifying node count.
 * @param options.n Number of quadrature points (default 20).
 * @returns The approximated integral value.
 *
 * @example Integrate x^3 over [0, 2]
 * print('integral =', integrateGauss((x) => x * x * x, 0, 2))
 */
export function integrateGauss(
  f: (x: number) => number,
  a: number,
  b: number,
  { n = 20 }: { n?: number } = {},
): number {
  const { nodes, weights } = gaussLegendre(n, [a, b])
  let total = 0
  for (let i = 0; i < n; i++) total += weights.data[i] * f(nodes.data[i])
  return total
}
