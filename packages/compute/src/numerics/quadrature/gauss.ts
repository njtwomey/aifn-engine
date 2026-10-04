/**
 * Gaussian quadrature rules: nodes and weights of Gauss–Legendre (weight 1 on [−1, 1]), Gauss–Hermite (weight e^{−x²},
 * or e^{−x²/2} in the probabilists' form) and generalised Gauss–Laguerre (weight x^α e^{−x} on [0, ∞)). An n-point rule
 * integrates polynomials of degree up to 2n − 1 exactly against its weight.
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

/** A quadrature rule: ∫ w(x) f(x) dx ≈ Σ weights[i]·f(nodes[i]). */
export type QuadratureRule = { nodes: Tensor; weights: Tensor }

const MAX_NEWTON = 100
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

function checkOrder(n: number, where: string) {
  if (!Number.isInteger(n) || n < 1) throw new DomainError(where, `${where}: n must be a positive integer, got ${n}`)
}

/**
 * The n-point Gauss–Legendre rule on [a, b] (default [−1, 1]): ∫ₐᵇ f(x) dx ≈ Σ wᵢ f(xᵢ), exact for polynomials of
 * degree ≤ 2n − 1. On [a, b] the nodes are mapped linearly and the weights scaled by (b − a)/2.
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
 * The n-point Gauss–Hermite rule. Physicists' form (default): ∫ e^{−x²} f(x) dx ≈ Σ wᵢ f(xᵢ), as numpy's `hermgauss`.
 * With `probabilists: true`: ∫ e^{−x²/2} f(x) dx, as `hermegauss` (the weights then sum to √(2π)).
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
 * The n-point generalised Gauss–Laguerre rule: ∫₀^∞ x^α e^{−x} f(x) dx ≈ Σ wᵢ f(xᵢ), α > −1 (default 0, as numpy's
 * `laggauss`; scipy's `roots_genlaguerre` for α ≠ 0).
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
 * E[f(X)] for X ~ N(mean, sd²) by n-point Gauss–Hermite quadrature (default n = 32): with X = mean + √2·sd·z,
 * E f(X) = π^{−1/2} ∫ e^{−z²} f(mean + √2·sd·z) dz. Exact for polynomials of degree ≤ 2n − 1.
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

/** ∫ₐᵇ f(x) dx by the n-point Gauss–Legendre rule (default n = 20). */
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
