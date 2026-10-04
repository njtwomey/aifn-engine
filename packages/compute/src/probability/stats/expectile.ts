/**
 * Expectiles (Newey and Powell, 1987, "Asymmetric least squares estimation and testing", Econometrica 55(4)). The
 * τ-expectile of a weighted sample minimises Σ wᵢ |τ − 1(xᵢ < m)| (xᵢ − m)². It solves τ U(m) = (1 − τ) L(m) with the
 * excess U(m) = Σ wᵢ (xᵢ − m)⁺ and the shortfall L(m) = Σ wᵢ (m − xᵢ)⁺; τ = ½ gives the weighted mean. A distribution's
 * expectile is that of its atoms with their masses, or (continuous) of its quantiles at many equally weighted levels.
 */

import { allValues, type Data } from './input'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `expectile`. */
export type ExpectileOptions = {
  /** Non-negative weights (masses), one per value; default equal. */
  weights?: Data
}

/** The values sorted with their weights, and the running sums Σw and Σwx. */
function prepare(x: Data, weights?: Data) {
  const v = allValues(x)
  const n = v.length
  if (n === 0) throw new DomainError('expectile', 'expectile: no values')
  const w = weights ? allValues(weights) : null
  if (w && w.length !== n) throw new ShapeError('expectile', `expectile: ${n} values but ${w.length} weights`)
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => v[a] - v[b])
  const xs = Float64Array.from(order, (i) => v[i])
  const ws = Float64Array.from(order, (i) => (w ? w[i] : 1))
  const W = new Float64Array(n + 1)
  const S = new Float64Array(n + 1)
  for (let i = 0; i < n; i++) {
    if (!(ws[i] >= 0)) throw new DomainError('expectile', 'expectile: weights must be non-negative')
    W[i + 1] = W[i] + ws[i]
    S[i + 1] = S[i] + ws[i] * xs[i]
  }
  if (!(W[n] > 0)) throw new DomainError('expectile', 'expectile: the weights sum to zero')
  return { xs, W, S, n }
}

/**
 * With the k smallest values below m: m = (τ(S − Sₖ) + (1 − τ)Sₖ) / (τ(W − Wₖ) + (1 − τ)Wₖ). The left side of
 * τU = (1 − τ)L decreases in m and the right increases, so the k whose m lies in [x₍ₖ₋₁₎, x₍ₖ₎] is found by bisection.
 */
function solve(p: ReturnType<typeof prepare>, tau: number): number {
  if (!(tau > 0 && tau < 1)) throw new DomainError('expectile', `expectile: τ = ${tau} is not in (0, 1)`)
  const { xs, W, S, n } = p
  const at = (k: number) => {
    const den = tau * (W[n] - W[k]) + (1 - tau) * W[k]
    return (tau * (S[n] - S[k]) + (1 - tau) * S[k]) / den
  }
  // g(k) = m(k) − x₍ₖ₎ (0-based x₍ₖ₎, the first value not below m): the answer is the smallest k with m(k) ≤ x₍ₖ₎.
  let lo = 1
  let hi = n
  while (lo < hi) {
    const k = (lo + hi) >> 1
    if (at(k) <= xs[k]) hi = k
    else lo = k + 1
  }
  return Math.min(Math.max(at(lo), xs[0]), xs[n - 1])
}

/** The τ-expectile of the (weighted) values, τ ∈ (0, 1) (see the module comment). */
export function expectile(x: Data, tau: number, options: ExpectileOptions = {}): number {
  return solve(prepare(x, options.weights), tau)
}

/** The expectiles at several levels, sorting the values once. */
export function expectiles(x: Data, taus: Data, options: ExpectileOptions = {}): Float64Array {
  const p = prepare(x, options.weights)
  return Float64Array.from(allValues(taus), (t) => solve(p, t))
}
