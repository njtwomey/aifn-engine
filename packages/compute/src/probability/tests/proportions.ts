/**
 * Tests and intervals for proportions: the exact binomial test (scipy's `binomtest`), the two-proportion $z$-test,
 * the Wald, Wilson (with and without continuity correction) and Clopper–Pearson intervals for one proportion (scipy's
 * `proportion_ci`), and the Wald and Newcombe intervals for a difference of proportions. The data are counts: $k$
 * successes in $n$ trials, checked to be integers with $0 \le k \le n$ and $n \ge 1$ (a `DomainError` otherwise).
 * Effect sizes are Cohen's $h$, a difference on the arcsine scale.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { Beta, Binomial, Normal } from 'aifn-compute/probability/distributions'
import {
  checkLevel,
  pivotInterval,
  pValueOf,
  result,
  tailOf,
  type Alternative,
  type Interval,
  type TestOptions,
  type TestResult,
} from './protocol'

/**
 * How an interval for one proportion is built: `wald` (the normal approximation at $\hat p$), `wilson` (the score
 * interval), `wilson-cc` (with a continuity correction) or `clopper-pearson` (exact).
 */
export type ProportionIntervalMethod = 'wald' | 'wilson' | 'wilson-cc' | 'clopper-pearson'

/**
 * Check a binomial count: throws `DomainError` unless $n$ is a positive integer and $k$ an integer in $[0, n]$.
 *
 * @param k The number of successes.
 * @param n The number of trials.
 * @param where The caller's name, for error messages.
 */
function counts(k: number, n: number, where: string): void {
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError(where, `${where}: n must be a positive integer`)
  if (!(Number.isInteger(k) && k >= 0 && k <= n))
    throw new DomainError(where, `${where}: k must be an integer in [0, n]`)
}

const standardNormal = Normal(0, 1)
/**
 * The standard normal's upper quantile.
 *
 * @param p An upper-tail probability.
 * @returns The $z$ with $\pr(Z > z) = p$.
 */
const z = (p: number) => standardNormal.isf(p) as number

/**
 * A confidence interval for a binomial proportion from $k$ successes in $n$ trials. `wald`:
 * $\hat p \pm z \sqrt{\hat p(1 - \hat p)/n}$, cut to $[0, 1]$; `wilson` (default): the score interval (Wilson, 1927),
 * the $p$ whose score test does not reject; `wilson-cc`: with the continuity correction (Newcombe, 1998);
 * `clopper-pearson`: the exact interval from beta quantiles (Clopper and Pearson, 1934), which always covers at least
 * `level`. One-sided for `alternative` less (lower end 0) or greater (upper end 1). The Wilson and Clopper–Pearson
 * intervals end at 0 when $k = 0$ and at 1 when $k = n$. Wilson's formulas follow scipy's `proportion_ci`.
 *
 * @param k The number of successes, an integer in $[0, n]$.
 * @param n The number of trials, a positive integer.
 * @param options The interval.
 * @param options.method Which interval (see `ProportionIntervalMethod`).
 * @param options.level The confidence level, in $(0, 1)$.
 * @param options.alternative `two-sided`, `less` (an upper bound, from 0) or `greater` (a lower bound, to 1).
 * @returns The interval, within $[0, 1]$.
 *
 * @example Four intervals for 7 successes in 20 trials
 * for (const method of ['wald', 'wilson', 'wilson-cc', 'clopper-pearson'])
 *   print(method, proportionInterval(7, 20, { method }))
 *
 * @example No successes: Wald's interval collapses, Wilson's does not
 * print('wald:', proportionInterval(0, 20, { method: 'wald' }))
 * print('wilson:', proportionInterval(0, 20))
 */
export function proportionInterval(
  k: number,
  n: number,
  {
    method = 'wilson',
    level = 0.95,
    alternative = 'two-sided',
  }: { method?: ProportionIntervalMethod; level?: number; alternative?: Alternative } = {},
): Interval {
  counts(k, n, 'proportionInterval')
  checkLevel(level, 'proportionInterval')
  const p = k / n
  const q = 1 - p
  const lowerOpen = alternative !== 'less'
  const upperOpen = alternative !== 'greater'
  if (method === 'clopper-pearson') {
    const a = alternative === 'two-sided' ? (1 - level) / 2 : 1 - level
    const lower = lowerOpen && k > 0 ? (Beta(k, n - k + 1).quantile(a) as number) : 0
    const upper = upperOpen && k < n ? (Beta(k + 1, n - k).isf(a) as number) : 1
    return { lower, upper, level }
  }
  const zz = alternative === 'two-sided' ? z((1 - level) / 2) : z(1 - level)
  if (method === 'wald') {
    const r = pivotInterval(p, Math.sqrt((p * q) / n), standardNormal, level, alternative)
    return { lower: Math.max(0, r.lower), upper: Math.min(1, r.upper), level }
  }
  const denom = 2 * (n + zz * zz)
  const centre = (2 * n * p + zz * zz) / denom
  if (method === 'wilson') {
    const delta = (zz / denom) * Math.sqrt(4 * n * p * q + zz * zz)
    return {
      lower: lowerOpen && k > 0 ? centre - delta : 0,
      upper: upperOpen && k < n ? centre + delta : 1,
      level,
    }
  }
  const lower =
    lowerOpen && k > 0 ? centre - (1 + zz * Math.sqrt(zz * zz - 2 - 1 / n + 4 * p * (n * q + 1))) / denom : 0
  const upper =
    upperOpen && k < n ? centre + (1 + zz * Math.sqrt(zz * zz + 2 - 1 / n + 4 * p * (n * q - 1))) / denom : 1
  return { lower: Math.max(0, lower), upper: Math.min(1, upper), level }
}

/**
 * Cohen's $h = 2\arcsin\sqrt{p_1} - 2\arcsin\sqrt{p_2}$, the difference of proportions on the variance-stabilising
 * scale, where a difference of 0.2, 0.5 or 0.8 is called small, medium or large (Cohen, 1988).
 *
 * @param p1 The first proportion, in $[0, 1]$.
 * @param p2 The second proportion, in $[0, 1]$.
 * @returns The effect size $h$, in $[-\pi, \pi]$.
 *
 * @example The same difference of 0.15 is larger near the ends of the scale
 * print('0.45 vs 0.30:', cohensH(0.45, 0.3))
 * print('0.20 vs 0.05:', cohensH(0.2, 0.05))
 */
export function cohensH(p1: number, p2: number): number {
  return 2 * Math.asin(Math.sqrt(p1)) - 2 * Math.asin(Math.sqrt(p2))
}

/**
 * The exact binomial test of $H_0: \pr(\text{success}) = p$ (`p`, default $\tfrac12$) from $k$ successes in $n$
 * trials. The statistic is $k$, with null law $\Binom(n, p)$; a one-sided p-value is a binomial tail, and the
 * two-sided one sums the probabilities of every outcome no more likely than $k$ (scipy's `binomtest`). The interval
 * is for the proportion (Clopper–Pearson unless `interval` names another method); the effect size is Cohen's $h$ of
 * $k/n$ against $p$. Throws `DomainError` for counts that are not valid or $p$ outside $[0, 1]$.
 *
 * @param k The number of successes, an integer in $[0, n]$.
 * @param n The number of trials, a positive integer.
 * @param options The alternative (`greater`: the success probability is above $p$) and confidence level; `p`, the
 *   success probability under the null; and `interval`, the method of the interval reported.
 * @returns The test result, with the exact null law $\Binom(n, p)$.
 *
 * @example Is a coin fair? 9 heads in 10 tosses rejects, 7 does not
 * const r = binomialTest(9, 10)
 * print('9 of 10: p =', r.pValue, ' interval for P(heads):', r.ci)
 * print('7 of 10: p =', binomialTest(7, 10).pValue)
 *
 * @example A one-sided test against p = 0.4
 * print('3 of 20, less: p =', binomialTest(3, 20, { p: 0.4, alternative: 'less' }).pValue)
 */
export function binomialTest(
  k: number,
  n: number,
  options: TestOptions & { p?: number; interval?: ProportionIntervalMethod } = {},
): TestResult {
  counts(k, n, 'binomialTest')
  const p = options.p ?? 0.5
  if (!(p >= 0 && p <= 1)) throw new DomainError('binomialTest', 'binomialTest: p must be in [0, 1]')
  const alternative = options.alternative ?? 'two-sided'
  const tail = alternative === 'two-sided' ? 'likelihood' : tailOf(alternative)
  const law = Binomial(n, p)
  return result({
    test: 'binomialTest',
    method: 'Exact binomial test',
    statistic: k,
    symbol: 'k',
    pValue: pValueOf(law, k, tail),
    alternative,
    tail,
    null: law,
    estimand: 'proportion',
    estimate: k / n,
    nullValue: p,
    ci: proportionInterval(k, n, {
      method: options.interval ?? 'clopper-pearson',
      level: options.level ?? 0.95,
      alternative,
    }),
    effectSize: { name: "Cohen's h", value: cohensH(k / n, p) },
    n,
  })
}

/**
 * The two-proportion $z$-test of $H_0: p_1 = p_2$ from $k_1$ of $n_1$ and $k_2$ of $n_2$:
 * $z = (\hat p_1 - \hat p_2)/\sqrt{\bar p(1 - \bar p)(1/n_1 + 1/n_2)}$ with the pooled
 * $\bar p = (k_1 + k_2)/(n_1 + n_2)$, standard normal under the null ($z^2$ is the uncorrected $\chi^2$ of the
 * $2 \times 2$ table). The interval is the Wald interval for $p_1 - p_2$ (unpooled standard error); the effect size is
 * Cohen's $h$.
 *
 * @param k1 The successes in the first group.
 * @param n1 The trials in the first group.
 * @param k2 The successes in the second group.
 * @param n2 The trials in the second group.
 * @param options The alternative (`greater`: $p_1 > p_2$) and the confidence level of the interval.
 * @returns The test result for $p_1 - p_2$.
 *
 * @example 45% against 30% in groups of 100
 * const r = twoProportionZTest(45, 100, 30, 100)
 * print('z =', r.statistic, ' p =', r.pValue)
 * print('z squared =', r.statistic ** 2, ', the uncorrected chi-square of [[45, 55], [30, 70]]')
 * print('difference =', r.estimate, ' 95% interval:', r.ci)
 *
 * @example 60% against 50% in groups of 20 is not enough evidence
 * print('p =', twoProportionZTest(12, 20, 10, 20).pValue)
 */
export function twoProportionZTest(
  k1: number,
  n1: number,
  k2: number,
  n2: number,
  options: TestOptions = {},
): TestResult {
  counts(k1, n1, 'twoProportionZTest')
  counts(k2, n2, 'twoProportionZTest')
  const alternative = options.alternative ?? 'two-sided'
  const [p1, p2] = [k1 / n1, k2 / n2]
  const pooled = (k1 + k2) / (n1 + n2)
  const statistic = (p1 - p2) / Math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
  const tail = tailOf(alternative)
  return result({
    test: 'twoProportionZTest',
    method: 'Two-proportion z-test (pooled)',
    statistic,
    symbol: 'z',
    pValue: pValueOf(standardNormal, statistic, tail),
    alternative,
    tail,
    null: standardNormal,
    estimand: 'difference of proportions',
    estimate: p1 - p2,
    nullValue: 0,
    ci: differenceOfProportionsInterval(k1, n1, k2, n2, { level: options.level ?? 0.95, alternative }),
    effectSize: { name: "Cohen's h", value: cohensH(p1, p2) },
    n: n1 + n2,
  })
}

/**
 * A confidence interval for $p_1 - p_2$ from independent binomial samples: `wald` (default),
 * $(\hat p_1 - \hat p_2) \pm z \sqrt{\hat p_1 \hat q_1/n_1 + \hat p_2 \hat q_2/n_2}$ with $\hat q = 1 - \hat p$; or
 * `newcombe`, the hybrid score interval (Newcombe, 1998, method 10) that combines the two Wilson intervals
 * $(l_i, u_i)$: from $d - \sqrt{(\hat p_1 - l_1)^2 + (u_2 - \hat p_2)^2}$ to
 * $d + \sqrt{(u_1 - \hat p_1)^2 + (\hat p_2 - l_2)^2}$ with $d = \hat p_1 - \hat p_2$, which stays inside $[-1, 1]$.
 * Newcombe's interval is two-sided only: a one-sided `alternative` throws `DomainError`.
 *
 * @param k1 The successes in the first group.
 * @param n1 The trials in the first group.
 * @param k2 The successes in the second group.
 * @param n2 The trials in the second group.
 * @param options The interval.
 * @param options.method `wald` or `newcombe`.
 * @param options.level The confidence level, in $(0, 1)$.
 * @param options.alternative `two-sided`, `less` (an upper bound) or `greater` (a lower bound); Wald only.
 * @returns The interval for $p_1 - p_2$.
 *
 * @example Wald's and Newcombe's intervals for 45 of 100 against 30 of 100
 * print('wald:', differenceOfProportionsInterval(45, 100, 30, 100))
 * print('newcombe:', differenceOfProportionsInterval(45, 100, 30, 100, { method: 'newcombe' }))
 */
export function differenceOfProportionsInterval(
  k1: number,
  n1: number,
  k2: number,
  n2: number,
  {
    method = 'wald',
    level = 0.95,
    alternative = 'two-sided',
  }: { method?: 'wald' | 'newcombe'; level?: number; alternative?: Alternative } = {},
): Interval {
  counts(k1, n1, 'differenceOfProportionsInterval')
  counts(k2, n2, 'differenceOfProportionsInterval')
  const [p1, p2] = [k1 / n1, k2 / n2]
  const d = p1 - p2
  if (method === 'wald')
    return pivotInterval(d, Math.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2), standardNormal, level, alternative)
  const a = proportionInterval(k1, n1, { method: 'wilson', level, alternative: 'two-sided' })
  const b = proportionInterval(k2, n2, { method: 'wilson', level, alternative: 'two-sided' })
  if (alternative !== 'two-sided')
    throw new DomainError(
      'differenceOfProportionsInterval',
      "differenceOfProportionsInterval: Newcombe's interval is two-sided",
    )
  return {
    lower: d - Math.sqrt((p1 - a.lower) ** 2 + (b.upper - p2) ** 2),
    upper: d + Math.sqrt((a.upper - p1) ** 2 + (p2 - b.lower) ** 2),
    level,
  }
}
