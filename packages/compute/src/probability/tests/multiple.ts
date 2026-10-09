/**
 * Multiple-testing procedures: adjusted p-values and rejections for a family of $m$ tests at level $\alpha$.
 * Bonferroni, Holm and Hochberg control the family-wise error rate (the probability of any false rejection);
 * Benjamini–Hochberg and Benjamini–Yekutieli the false discovery rate (the expected share of false rejections among
 * the rejections). Adjusted p-values follow statsmodels' `multipletests`: the hypothesis $i$ is rejected exactly when
 * its adjusted p-value is at most $\alpha$. P-values outside $[0, 1]$ and $\alpha$ outside $(0, 1)$ throw
 * `DomainError`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'

/** The result of a multiple-testing procedure, in the order of the input p-values. */
export type MultipleTesting = {
  /** Always `'multiple-testing'`. */
  readonly kind: 'multiple-testing'
  /** The procedure's registry key (`holm`). */
  readonly method: string
  /** Adjusted p-values, each in [0, 1]. */
  readonly adjusted: Tensor
  /** True where the hypothesis is rejected at `alpha` (a bool tensor). */
  readonly rejected: Tensor
  /** The family's level $\alpha$. */
  readonly alpha: number
  /** The number of hypotheses. */
  readonly m: number
}

/**
 * The p-values as a Float64Array, checked: at least one, each in $[0, 1]$ (a `DomainError` otherwise).
 *
 * @param p The p-values.
 * @param where The caller's name, for error messages.
 * @returns A new array of them.
 */
function pValues(p: VectorLike, where: string): Float64Array {
  const v = dense.toF64(p, where)
  if (v.length === 0) throw new DomainError(where, `${where}: needs at least one p-value`)
  for (const a of v) if (!(a >= 0 && a <= 1)) throw new DomainError(where, `${where}: p-values must be in [0, 1]`)
  return v
}

/**
 * Indices that sort `p` ascending (stable: equal p-values keep their order).
 *
 * @param p The p-values.
 * @returns The permutation, smallest p-value's index first.
 */
const ascending = (p: Float64Array) => Array.from(p.keys()).sort((i, j) => p[i] - p[j] || i - j)

/**
 * The result of a procedure from its adjusted p-values: rejected where they are at most $\alpha$. Throws
 * `DomainError` unless $\alpha$ is in $(0, 1)$.
 *
 * @param method The procedure's registry key, also used in the error message.
 * @param adjusted The adjusted p-values, in input order.
 * @param alpha The family's level $\alpha$.
 * @returns The result.
 */
function finish(method: string, adjusted: Float64Array, alpha: number): MultipleTesting {
  if (!(alpha > 0 && alpha < 1)) throw new DomainError(method, `${method}: α must be in (0, 1)`)
  return {
    kind: 'multiple-testing',
    method,
    adjusted: fromData(adjusted, [adjusted.length]),
    rejected: fromData(
      Uint8Array.from(adjusted, (q) => (q <= alpha ? 1 : 0)),
      [adjusted.length],
      'bool',
    ),
    alpha,
    m: adjusted.length,
  }
}

/**
 * Step-down adjustment: with the p-values sorted ascending, $q_{(i)} = \max_{j \le i} \min(1, c_j p_{(j)})$ (Holm's
 * form), or step-up, $q_{(i)} = \min_{j \ge i} \min(1, c_j p_{(j)})$ (Hochberg's and Benjamini's), for the rank
 * weights $c_j$.
 *
 * @param p The p-values, in input order; not modified.
 * @param weight The weight $c_j$ of the $j$-th smallest p-value ($j$ from 1), given $j$ and $m$.
 * @param down True for step-down (running maximum from the smallest), false for step-up (running minimum from the
 *   largest).
 * @returns The adjusted p-values, in input order.
 */
function stepwise(p: Float64Array, weight: (rank: number, m: number) => number, down: boolean): Float64Array {
  const m = p.length
  const order = ascending(p)
  const q = new Float64Array(m)
  const sorted = order.map((i, r) => Math.min(1, weight(r + 1, m) * p[i]))
  if (down) for (let r = 1; r < m; r++) sorted[r] = Math.max(sorted[r], sorted[r - 1])
  else for (let r = m - 2; r >= 0; r--) sorted[r] = Math.min(sorted[r], sorted[r + 1])
  order.forEach((i, r) => (q[i] = sorted[r]))
  return q
}

/**
 * Bonferroni's correction: $q_i = \min(1, m p_i)$; controls the family-wise error rate under any dependence.
 *
 * @param p The p-values of the $m$ tests.
 * @param options Options of the procedure.
 * @param options.alpha The family-wise level $\alpha$, in $(0, 1)$.
 * @returns The adjusted p-values and rejections.
 *
 * @example Seven p-values: two survive
 * const r = bonferroni([0.001, 0.004, 0.012, 0.03, 0.2, 0.5, 0.8])
 * print('adjusted:', r.adjusted)
 * print('rejected:', r.rejected)
 */
export function bonferroni(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  const v = pValues(p, 'bonferroni')
  return finish(
    'bonferroni',
    v.map((a) => Math.min(1, v.length * a)),
    alpha,
  )
}

/**
 * Holm's step-down procedure (Holm, 1979): the $i$-th smallest p-value is compared with $\alpha/(m - i + 1)$,
 * stopping at the first that fails; uniformly more powerful than Bonferroni with the same family-wise guarantee.
 *
 * @param p The p-values of the $m$ tests.
 * @param options Options of the procedure.
 * @param options.alpha The family-wise level $\alpha$, in $(0, 1)$.
 * @returns The adjusted p-values and rejections.
 *
 * @example Smaller adjusted p-values than Bonferroni's, for the same p-values
 * const p = [0.001, 0.004, 0.012, 0.03, 0.2, 0.5, 0.8]
 * print('Holm:', holm(p).adjusted)
 * print('Bonferroni:', bonferroni(p).adjusted)
 * print('rejected:', holm(p).rejected)
 */
export function holm(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  return finish(
    'holm',
    stepwise(pValues(p, 'holm'), (r, m) => m - r + 1, true),
    alpha,
  )
}

/**
 * Hochberg's step-up procedure (Hochberg, 1988): the same thresholds $\alpha/(m - i + 1)$ as Holm's, scanned from the
 * largest p-value down; controls the family-wise error rate for independent or positively dependent tests.
 *
 * @param p The p-values of the $m$ tests.
 * @param options Options of the procedure.
 * @param options.alpha The family-wise level $\alpha$, in $(0, 1)$.
 * @returns The adjusted p-values and rejections.
 *
 * @example Three p-values below 0.05: Hochberg rejects all three, Holm none
 * const p = [0.02, 0.03, 0.04]
 * print('Hochberg:', hochberg(p).adjusted, hochberg(p).rejected)
 * print('Holm:', holm(p).adjusted, holm(p).rejected)
 */
export function hochberg(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  return finish(
    'hochberg',
    stepwise(pValues(p, 'hochberg'), (r, m) => m - r + 1, false),
    alpha,
  )
}

/**
 * The Benjamini–Hochberg step-up procedure (Benjamini and Hochberg, 1995): reject the $k$ smallest p-values for the
 * largest $k$ with $p_{(k)} \le k\alpha/m$; controls the false discovery rate at $\alpha$ for independent or
 * positively dependent tests. The adjusted p-values are $\min_{j \ge i} \min(1, m p_{(j)}/j)$.
 *
 * @param p The p-values of the $m$ tests.
 * @param options Options of the procedure.
 * @param options.alpha The false discovery rate $\alpha$, in $(0, 1)$.
 * @returns The adjusted p-values and rejections.
 *
 * @example Controlling the false discovery rate rejects one more than Holm
 * const p = [0.001, 0.004, 0.012, 0.03, 0.2, 0.5, 0.8]
 * const r = benjaminiHochberg(p)
 * print('adjusted:', r.adjusted)
 * print('rejected:', r.rejected, ' Holm:', holm(p).rejected)
 */
export function benjaminiHochberg(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  return finish(
    'benjaminiHochberg',
    stepwise(pValues(p, 'benjaminiHochberg'), (r, m) => m / r, false),
    alpha,
  )
}

/**
 * The Benjamini–Yekutieli procedure (Benjamini and Yekutieli, 2001): Benjamini–Hochberg at $\alpha/c(m)$, with
 * $c(m) = \sum_{i=1}^{m} 1/i$; controls the false discovery rate under any dependence.
 *
 * @param p The p-values of the $m$ tests.
 * @param options Options of the procedure.
 * @param options.alpha The false discovery rate $\alpha$, in $(0, 1)$.
 * @returns The adjusted p-values and rejections.
 *
 * @example The price of arbitrary dependence: larger adjusted p-values than Benjamini–Hochberg's
 * const p = [0.001, 0.004, 0.012, 0.03, 0.2, 0.5, 0.8]
 * print('BY:', benjaminiYekutieli(p).adjusted)
 * print('BH:', benjaminiHochberg(p).adjusted)
 * print('rejected:', benjaminiYekutieli(p).rejected)
 */
export function benjaminiYekutieli(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  const v = pValues(p, 'benjaminiYekutieli')
  let c = 0
  for (let i = 1; i <= v.length; i++) c += 1 / i
  return finish(
    'benjaminiYekutieli',
    stepwise(v, (r, m) => (c * m) / r, false),
    alpha,
  )
}
