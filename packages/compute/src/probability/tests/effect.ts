/**
 * Standardised effect sizes for means: Cohen's $d$ and Hedges' $g$, a difference of means in units of the standard
 * deviation, which does not depend on the scale of the data. Hedges' $g$ removes the small-sample bias of $d$ with the
 * exact factor $J(\nu)$. The effect sizes of tables (the odds ratio and Cramér's $V$) are in `contingency.ts`.
 */

import type { VectorLike } from 'aifn-compute/foundation/tensor'
import { logGamma } from 'aifn-compute/numerics/special'
import { meanAndVariance, sample } from './protocol'

/**
 * Cohen's $d$ (Cohen, 1988): the difference of means in units of the pooled standard deviation,
 * $(\bar x - \bar y)/s_p$ with $s_p^2 = ((m - 1)s_x^2 + (n - 1)s_y^2)/(m + n - 2)$. With one sample it is
 * $(\bar x - \mu_0)/s_x$, the standardised distance from `mu` (default 0); for paired data pass the differences.
 * Throws `DomainError` for a sample with fewer than two values or a non-finite one.
 *
 * @param x The first sample ($m \ge 2$ values).
 * @param y The second sample ($n \ge 2$ values), or null or left out for the one-sample form.
 * @param options Options of the one-sample form.
 * @param options.mu The reference mean $\mu_0$ of the one-sample form; ignored when `y` is given.
 * @returns The effect size $d$.
 *
 * @example Two samples whose means differ by 4 standard deviations
 * print('two samples:', cohensD([5, 6, 7], [1, 2, 3]))
 * print('one sample against 0:', cohensD([1, 2, 3]))
 * print('one sample against 1:', cohensD([1, 2, 3], null, { mu: 1 }))
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
 * The small-sample correction $J(\nu) = \Gamma(\nu/2)/(\sqrt{\nu/2}\,\Gamma((\nu - 1)/2))$ that makes $J d$ unbiased
 * for the population effect (Hedges, 1981), exactly rather than by the approximation $1 - 3/(4\nu - 1)$.
 *
 * @param df The degrees of freedom $\nu$ of the standard deviation: $m + n - 2$ for two samples, $n - 1$ for one.
 *   Must be above 1.
 * @returns The factor $J(\nu)$, in $(0, 1)$ and rising to 1 as $\nu$ grows.
 *
 * @example The exact factor beside Hedges' approximation
 * for (const df of [2, 4, 10, 100]) print('df =', df, ' J =', hedgesCorrection(df), ' approx =', 1 - 3 / (4 * df - 1))
 */
export function hedgesCorrection(df: number): number {
  return Math.exp((logGamma(df / 2) as number) - (logGamma((df - 1) / 2) as number) - 0.5 * Math.log(df / 2))
}

/**
 * Hedges' $g$: Cohen's $d$ times the exact correction $J(\nu)$, with $\nu = m + n - 2$ for two samples and $n - 1$
 * for one.
 *
 * @param x The first sample (at least two values).
 * @param y The second sample (at least two values), or null or left out for the one-sample form.
 * @param options Passed to `cohensD`: `mu`, the reference mean of the one-sample form.
 * @returns The effect size $g$, smaller in magnitude than $d$.
 *
 * @example Three values per group shrink d by a fifth
 * print("Cohen's d =", cohensD([5, 6, 7], [1, 2, 3]))
 * print("Hedges' g =", hedgesG([5, 6, 7], [1, 2, 3]))
 */
export function hedgesG(x: VectorLike, y?: VectorLike | null, options: { mu?: number } = {}): number {
  const m = sample(x, 'hedgesG', 2).length
  const df = y ? m + sample(y, 'hedgesG', 2).length - 2 : m - 1
  return hedgesCorrection(df) * cohensD(x, y, options)
}
