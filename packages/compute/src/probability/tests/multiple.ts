/**
 * Multiple-testing procedures: adjusted p-values and rejections for a family of m tests at level α. Bonferroni, Holm
 * and Hochberg control the family-wise error rate; Benjamini–Hochberg and Benjamini–Yekutieli the false discovery
 * rate. Adjusted p-values follow statsmodels' `multipletests`: the hypothesis i is rejected exactly when its adjusted
 * p-value is at most α.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor, type VectorLike } from 'aifn-compute/foundation/tensor'

/** The result of a multiple-testing procedure, in the order of the input p-values. */
export type MultipleTesting = {
  readonly kind: 'multiple-testing'
  /** The procedure's registry key (`holm`). */
  readonly method: string
  /** Adjusted p-values, each in [0, 1]. */
  readonly adjusted: Tensor
  /** True where the hypothesis is rejected at `alpha` (a bool tensor). */
  readonly rejected: Tensor
  readonly alpha: number
  /** The number of hypotheses. */
  readonly m: number
}

function pValues(p: VectorLike, where: string): Float64Array {
  const v = dense.toF64(p, where)
  if (v.length === 0) throw new DomainError(where, `${where}: needs at least one p-value`)
  for (const a of v) if (!(a >= 0 && a <= 1)) throw new DomainError(where, `${where}: p-values must be in [0, 1]`)
  return v
}

/** Indices that sort `p` ascending (stable). */
const ascending = (p: Float64Array) => Array.from(p.keys()).sort((i, j) => p[i] - p[j] || i - j)

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
 * Step-down adjustment: with the p-values sorted ascending, q₍ᵢ₎ = maxⱼ≤ᵢ min(1, cⱼ p₍ⱼ₎) (Holm's form), or step-up,
 * q₍ᵢ₎ = minⱼ≥ᵢ min(1, cⱼ p₍ⱼ₎) (Hochberg's and Benjamini's), for the rank weights cⱼ.
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

/** Bonferroni's correction: qᵢ = min(1, m pᵢ); controls the family-wise error rate under any dependence. */
export function bonferroni(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  const v = pValues(p, 'bonferroni')
  return finish(
    'bonferroni',
    v.map((a) => Math.min(1, v.length * a)),
    alpha,
  )
}

/**
 * Holm's step-down procedure (Holm, 1979): the i-th smallest p-value is compared with α/(m − i + 1), stopping at the
 * first that fails; uniformly more powerful than Bonferroni with the same family-wise guarantee.
 */
export function holm(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  return finish(
    'holm',
    stepwise(pValues(p, 'holm'), (r, m) => m - r + 1, true),
    alpha,
  )
}

/**
 * Hochberg's step-up procedure (Hochberg, 1988): the same thresholds α/(m − i + 1) as Holm's, scanned from the
 * largest p-value down; controls the family-wise error rate for independent or positively dependent tests.
 */
export function hochberg(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  return finish(
    'hochberg',
    stepwise(pValues(p, 'hochberg'), (r, m) => m - r + 1, false),
    alpha,
  )
}

/**
 * The Benjamini–Hochberg step-up procedure (Benjamini and Hochberg, 1995): reject the k smallest p-values for the
 * largest k with p₍ₖ₎ ≤ kα/m; controls the false discovery rate at α for independent or positively dependent tests.
 */
export function benjaminiHochberg(p: VectorLike, { alpha = 0.05 }: { alpha?: number } = {}): MultipleTesting {
  return finish(
    'benjaminiHochberg',
    stepwise(pValues(p, 'benjaminiHochberg'), (r, m) => m / r, false),
    alpha,
  )
}

/**
 * The Benjamini–Yekutieli procedure (Benjamini and Yekutieli, 2001): Benjamini–Hochberg at α/c(m), with
 * c(m) = Σᵢ₌₁ᵐ 1/i; controls the false discovery rate under any dependence.
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
