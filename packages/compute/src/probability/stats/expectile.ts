/**
 * Expectiles of a weighted sample (Newey and Powell, 1987, "Asymmetric least squares estimation and testing",
 * Econometrica 55(4)).
 *
 * The $\tau$-expectile of a weighted sample minimises
 * $\sum_i w_i \lvert \tau - \indicator(x_i < m) \rvert (x_i - m)^2$. It solves $\tau U(m) = (1 - \tau) L(m)$ with
 * the excess $U(m) = \sum_i w_i (x_i - m)^+$ and the shortfall $L(m) = \sum_i w_i (m - x_i)^+$; $\tau = \tfrac12$
 * gives the weighted mean. A distribution's expectile is that of its atoms with their masses, or (continuous) of its
 * quantiles at many equally weighted levels. The values are sorted once and the root found exactly, by bisection over
 * how many values lie below it.
 */

import { allValues, type Data } from './input'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Options of `expectile`. */
export type ExpectileOptions = {
  /**
   * Non-negative weights (masses), one per value, with a positive sum: an array or a tensor (every element). Default
   * equal.
   */
  weights?: Data
}

/**
 * The values sorted with their weights, and the running sums $W_k = \sum_{i < k} w_i$ and $S_k = \sum_{i < k} w_i x_i$
 * over the $k$ smallest. Throws `DomainError` for no values, a negative or NaN weight, or weights that sum to 0, and
 * `ShapeError` when the counts differ.
 *
 * @param x The values: an array, or a tensor of any rank (every element).
 * @param weights One non-negative weight per value; left out, every weight is 1.
 * @returns The sorted values `xs`, the running sums `W` and `S` ($n + 1$ entries each, from 0) and the count `n`.
 */
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
 * With the $k$ smallest values below $m$:
 * $m = (\tau(S_n - S_k) + (1 - \tau)S_k) / (\tau(W_n - W_k) + (1 - \tau)W_k)$. The left side of
 * $\tau U = (1 - \tau)L$ decreases in $m$ and the right increases, so the $k$ whose $m$ lies in
 * $[x_{(k-1)}, x_{(k)}]$ (0-based order statistics) is found by bisection. Throws `DomainError` unless
 * $0 < \tau < 1$.
 *
 * @param p The sorted values and running sums, as `prepare` returns them.
 * @param tau The level $\tau$, in $(0, 1)$.
 * @returns The $\tau$-expectile, clamped to the range of the values.
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

/**
 * The $\tau$-expectile of the (weighted) values, $\tau \in (0, 1)$: the $m$ with $\tau U(m) = (1 - \tau) L(m)$ (see
 * the file comment). As `scipy.stats.expectile`. Throws `DomainError` for no values, $\tau$ outside $(0, 1)$ or bad
 * weights, and `ShapeError` when the weights do not match the values.
 *
 * @param x The values: an array, or a tensor of any rank (every element).
 * @param tau The level $\tau$, in $(0, 1)$; $\tfrac12$ gives the (weighted) mean.
 * @param options The weights of the values (default equal).
 * @returns The expectile.
 *
 * @example Low, middle and high expectiles
 * const x = [1, 2, 3, 4, 10]
 * print('tau = 0.1:', expectile(x, 0.1))
 * print('tau = 0.5 (the mean):', expectile(x, 0.5))
 * print('tau = 0.9:', expectile(x, 0.9))
 *
 * @example Weights act as repeats
 * print('weighted =', expectile([1, 2, 3], 0.8, { weights: [1, 1, 2] }))
 * print('repeated =', expectile([1, 2, 3, 3], 0.8))
 */
export function expectile(x: Data, tau: number, options: ExpectileOptions = {}): number {
  return solve(prepare(x, options.weights), tau)
}

/**
 * The expectiles at several levels, sorting the values once. Throws as `expectile` does.
 *
 * @param x The values: an array, or a tensor of any rank (every element).
 * @param taus The levels $\tau$, each in $(0, 1)$: an array or a tensor (every element).
 * @param options The weights of the values (default equal).
 * @returns The expectile at each level, in the order of `taus`.
 *
 * @example Expectiles rise with the level
 * print('expectiles =', expectiles([1, 2, 3, 4, 10], [0.1, 0.25, 0.5, 0.75, 0.9]))
 */
export function expectiles(x: Data, taus: Data, options: ExpectileOptions = {}): Float64Array {
  const p = prepare(x, options.weights)
  return Float64Array.from(allValues(taus), (t) => solve(p, t))
}
