/**
 * Standardised effect sizes for means: Cohen's d and Hedges' g. The effect sizes of tables (the odds ratio and
 * Cramér's V) are in `contingency.ts`.
 */

import type { VectorLike } from 'aifn-compute/foundation/tensor'
import { logGamma } from 'aifn-compute/numerics/special'
import { meanAndVariance, sample } from './protocol'

/**
 * Cohen's d (Cohen, 1988): the difference of means in units of the pooled standard deviation,
 * (x̄ − ȳ)/s_p with s_p² = ((m − 1)s_x² + (n − 1)s_y²)/(m + n − 2). With one sample it is (x̄ − μ₀)/s_x, the
 * standardised distance from `mu` (default 0); for paired data pass the differences.
 */
export function cohensD(x: VectorLike, y?: VectorLike | null, { mu = 0 }: { mu?: number } = {}): number {
  const a = meanAndVariance(sample(x, 'cohensD', 2))
  if (!y) return (a.mean - mu) / Math.sqrt(a.variance)
  const yv = sample(y, 'cohensD', 2)
  const b = meanAndVariance(yv)
  const m = sample(x, 'cohensD').length
  const n = yv.length
  const pooled = ((m - 1) * a.variance + (n - 1) * b.variance) / (m + n - 2)
  return (a.mean - b.mean) / Math.sqrt(pooled)
}

/**
 * The small-sample correction J(ν) = Γ(ν/2)/(√(ν/2) Γ((ν − 1)/2)) that makes J · d unbiased for the population
 * effect (Hedges, 1981), exactly rather than by the approximation 1 − 3/(4ν − 1).
 */
export function hedgesCorrection(df: number): number {
  return Math.exp((logGamma(df / 2) as number) - (logGamma((df - 1) / 2) as number) - 0.5 * Math.log(df / 2))
}

/**
 * Hedges' g: Cohen's d times the exact correction J(ν), with ν = m + n − 2 for two samples and n − 1 for one.
 */
export function hedgesG(x: VectorLike, y?: VectorLike | null, options: { mu?: number } = {}): number {
  const m = sample(x, 'hedgesG', 2).length
  const df = y ? m + sample(y, 'hedgesG', 2).length - 2 : m - 1
  return hedgesCorrection(df) * cohensD(x, y, options)
}
