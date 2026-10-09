/**
 * Tests and intervals for means: the one-sample, paired, pooled and Welch $t$-tests (scipy's `ttest_1samp`,
 * `ttest_rel`, `ttest_ind`), the $z$-test with a known standard deviation, and the confidence intervals for a mean and
 * a difference of means. Each statistic is a standardised difference $(\hat\theta - \theta_0)/\mathrm{se}$ whose null
 * law is Student's $t$ (or the standard normal), so the interval is the same pivot inverted. Samples with a
 * non-finite value or too few values throw `DomainError`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import type { VectorLike } from 'aifn-compute/foundation/tensor'
import { Normal, StudentT, type Univariate } from 'aifn-compute/probability/distributions'
import { cohensD } from './effect'
import {
  meanAndVariance,
  pivotInterval,
  pValueOf,
  result,
  sample,
  tailOf,
  type Alternative,
  type Interval,
  type TestOptions,
  type TestResult,
} from './protocol'

/**
 * A pivot-based test: the statistic $(\hat\theta - \theta_0)/\mathrm{se}$ against `law`, its interval inverted from
 * the same pivot. The p-value is NaN when the statistic is (a zero standard error with a zero difference).
 *
 * @param spec The test: its registry key `test`, display name `method` and TeX `symbol`; the `estimand` and its
 *   `estimate` $\hat\theta$, `nullValue` $\theta_0$ and standard error `se`; the null `law` of the statistic and its
 *   `df`; the number of observations `n`; the caller's `options` (alternative and level); and the `effect` size to
 *   report.
 * @returns The test result.
 */
function pivotTest(spec: {
  test: string
  method: string
  symbol: string
  estimand: string
  estimate: number
  nullValue: number
  se: number
  law: Univariate
  df?: number
  n: number
  options: TestOptions
  effect?: { name: string; value: number }
}): TestResult {
  const alternative: Alternative = spec.options.alternative ?? 'two-sided'
  const level = spec.options.level ?? 0.95
  const statistic = (spec.estimate - spec.nullValue) / spec.se
  const tail = tailOf(alternative)
  return result({
    test: spec.test,
    method: spec.method,
    statistic,
    symbol: spec.symbol,
    df: spec.df,
    pValue: Number.isNaN(statistic) ? NaN : pValueOf(spec.law, statistic, tail),
    alternative,
    tail,
    null: spec.law,
    estimand: spec.estimand,
    estimate: spec.estimate,
    nullValue: spec.nullValue,
    ci: pivotInterval(spec.estimate, spec.se, spec.law, level, alternative),
    effectSize: spec.effect,
    n: spec.n,
  })
}

/**
 * The one-sample $t$-test of $H_0: \mu = \mu_0$ (`mu`, default 0) for a normal sample $\xvec$ of size $n \ge 2$:
 * $t = (\bar x - \mu_0)/(s/\sqrt n)$ with $n - 1$ degrees of freedom; the interval is for the mean, the effect size
 * Cohen's $d = (\bar x - \mu_0)/s$. As scipy's `ttest_1samp`.
 *
 * @param x The sample, at least two finite values.
 * @param options The alternative (`greater`: the mean is above $\mu_0$), the confidence level of the interval, and
 *   `mu`, the mean $\mu_0$ under the null.
 * @returns The test result: $t$, its $n - 1$ degrees of freedom, the p-value, and the interval for the mean.
 *
 * @example Is the mean 5? Rejected at 5%, though not by much
 * const x = [5.1, 4.9, 5.6, 5.8, 6.0, 5.3]
 * const r = oneSampleTTest(x, { mu: 5 })
 * print('t =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 * print('95% interval for the mean:', r.ci)
 * print('one-sided (greater): p =', oneSampleTTest(x, { mu: 5, alternative: 'greater' }).pValue)
 *
 * @example A null mean inside the data is not rejected
 * const r = oneSampleTTest([5.1, 4.9, 5.6, 5.8, 6.0, 5.3], { mu: 5.5 })
 * print('t =', r.statistic, ' p =', r.pValue)
 */
export function oneSampleTTest(x: VectorLike, options: TestOptions & { mu?: number } = {}): TestResult {
  const v = sample(x, 'oneSampleTTest', 2)
  const n = v.length
  const { mean, variance } = meanAndVariance(v)
  const mu = options.mu ?? 0
  return pivotTest({
    test: 'oneSampleTTest',
    method: 'One-sample t-test',
    symbol: 't',
    estimand: 'mean',
    estimate: mean,
    nullValue: mu,
    se: Math.sqrt(variance / n),
    law: StudentT(n - 1),
    df: n - 1,
    n,
    options,
    effect: { name: "Cohen's d", value: (mean - mu) / Math.sqrt(variance) },
  })
}

/**
 * The differences $x_i - y_i$ of two paired samples. Throws `DomainError` when their lengths differ.
 *
 * @param x The first sample, at least two values.
 * @param y The second sample, as long as `x`.
 * @param where The caller's name, for error messages.
 * @returns The differences, in order.
 */
function differences(x: VectorLike, y: VectorLike, where: string): Float64Array {
  const a = sample(x, where, 2)
  const b = sample(y, where, 2)
  if (a.length !== b.length) throw new DomainError(where, `${where}: paired samples must have equal lengths`)
  return a.map((v, i) => v - b[i])
}

/**
 * The paired $t$-test: the one-sample $t$-test of the differences $x_i - y_i$ against `mu` (default 0), with $n - 1$
 * degrees of freedom (scipy's `ttest_rel`). The interval is for the mean difference; the effect size is $d_z$, the
 * mean difference over the standard deviation of the differences.
 *
 * @param x The first measurement of each pair.
 * @param y The second measurement of each pair, in the same order; as long as `x`.
 * @param options The alternative (`greater`: $x$ tends to exceed $y$), the confidence level, and `mu`, the mean
 *   difference under the null.
 * @returns The test result for the mean difference.
 *
 * @example Blood pressure before and after a treatment
 * const before = [72, 80, 65, 90, 77, 84]
 * const after = [70, 76, 66, 85, 72, 80]
 * const r = pairedTTest(before, after)
 * print('t =', r.statistic, ' p =', r.pValue)
 * print('mean drop =', r.estimate, ' 95% interval:', r.ci)
 */
export function pairedTTest(x: VectorLike, y: VectorLike, options: TestOptions & { mu?: number } = {}): TestResult {
  const r = oneSampleTTest(differences(x, y, 'pairedTTest'), options)
  return {
    ...r,
    test: 'pairedTTest',
    method: 'Paired t-test',
    estimand: 'mean difference',
    effectSize: { name: "Cohen's d_z", value: r.effectSize!.value },
  }
}

/**
 * The two-sample $t$-test with a pooled variance (Student's): for independent normal samples of sizes $m$ and $n$
 * with a common variance, $t = (\bar x - \bar y - \delta_0)/(s_p \sqrt{1/m + 1/n})$ with $m + n - 2$ degrees of
 * freedom, $\delta_0$ = `mu` (default 0). The interval is for the difference of means, the effect size Cohen's $d$. As
 * scipy's `ttest_ind`.
 *
 * @param x The first sample, at least two values.
 * @param y The second sample, at least two values.
 * @param options The alternative (`greater`: the mean of $x$ is the larger), the confidence level, and `mu`, the
 *   difference of means $\delta_0$ under the null.
 * @returns The test result for $\mu_x - \mu_y$.
 *
 * @example Two groups with similar spreads
 * const r = pooledTTest([20.1, 22.4, 19.8, 21.5, 23.0], [18.2, 19.0, 17.5, 18.8, 19.9, 18.4])
 * print('t =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 * print('difference =', r.estimate, ' 95% interval:', r.ci)
 * print(r.effectSize)
 */
export function pooledTTest(x: VectorLike, y: VectorLike, options: TestOptions & { mu?: number } = {}): TestResult {
  const a = sample(x, 'pooledTTest', 2)
  const b = sample(y, 'pooledTTest', 2)
  const [m, n] = [a.length, b.length]
  const p = meanAndVariance(a)
  const q = meanAndVariance(b)
  const pooled = ((m - 1) * p.variance + (n - 1) * q.variance) / (m + n - 2)
  return pivotTest({
    test: 'pooledTTest',
    method: 'Two-sample t-test (pooled variance)',
    symbol: 't',
    estimand: 'difference of means',
    estimate: p.mean - q.mean,
    nullValue: options.mu ?? 0,
    se: Math.sqrt(pooled * (1 / m + 1 / n)),
    law: StudentT(m + n - 2),
    df: m + n - 2,
    n: m + n,
    options,
    effect: { name: "Cohen's d", value: cohensD(a, b) },
  })
}

/**
 * The Welch–Satterthwaite degrees of freedom $(u + v)^2/(u^2/(m - 1) + v^2/(n - 1))$, with $u = s_x^2/m$ and
 * $v = s_y^2/n$ the squared standard errors of the two means. It lies between $\min(m, n) - 1$ and $m + n - 2$.
 *
 * @param u The squared standard error $s_x^2/m$ of the first mean.
 * @param v The squared standard error $s_y^2/n$ of the second mean.
 * @param m The size of the first sample.
 * @param n The size of the second sample.
 * @returns The degrees of freedom, not rounded.
 *
 * @example Equal standard errors give the pooled value; one dominant error gives its own sample's
 * print('equal:', welchDegreesOfFreedom(1, 1, 6, 6))
 * print('second dominates:', welchDegreesOfFreedom(0.005, 1.6165, 5, 6))
 */
export function welchDegreesOfFreedom(u: number, v: number, m: number, n: number): number {
  return (u + v) ** 2 / (u ** 2 / (m - 1) + v ** 2 / (n - 1))
}

/**
 * Welch's two-sample $t$-test (Welch, 1947): no common variance is assumed;
 * $t = (\bar x - \bar y - \delta_0)/\sqrt{s_x^2/m + s_y^2/n}$, and its null law is approximated by Student's $t$ with
 * the Welch–Satterthwaite degrees of freedom (`welchDegreesOfFreedom`). The interval is for the difference of means,
 * the effect size Cohen's $d$ (pooled standard deviation). As scipy's `ttest_ind(equal_var=False)`.
 *
 * @param x The first sample, at least two values.
 * @param y The second sample, at least two values.
 * @param options The alternative (`greater`: the mean of $x$ is the larger), the confidence level, and `mu`, the
 *   difference of means $\delta_0$ under the null.
 * @returns The test result for $\mu_x - \mu_y$, with fractional degrees of freedom.
 *
 * @example A clear difference of means is detected
 * const r = welchTTest([20.1, 22.4, 19.8, 21.5, 23.0], [18.2, 19.0, 17.5, 18.8, 19.9, 18.4])
 * print('t =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 *
 * @example Equal means and very unequal spreads: not rejected, with few degrees of freedom
 * const r = welchTTest([10.2, 9.8, 10.1, 10.0, 9.9], [12.0, 6.5, 14.1, 8.0, 11.9, 7.2])
 * print('t =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 * print('Welch interval:', r.ci)
 * print('pooled interval:', pooledTTest([10.2, 9.8, 10.1, 10.0, 9.9], [12.0, 6.5, 14.1, 8.0, 11.9, 7.2]).ci)
 */
export function welchTTest(x: VectorLike, y: VectorLike, options: TestOptions & { mu?: number } = {}): TestResult {
  const a = sample(x, 'welchTTest', 2)
  const b = sample(y, 'welchTTest', 2)
  const [m, n] = [a.length, b.length]
  const p = meanAndVariance(a)
  const q = meanAndVariance(b)
  const [u, v] = [p.variance / m, q.variance / n]
  const df = welchDegreesOfFreedom(u, v, m, n)
  return pivotTest({
    test: 'welchTTest',
    method: 'Welch two-sample t-test',
    symbol: 't',
    estimand: 'difference of means',
    estimate: p.mean - q.mean,
    nullValue: options.mu ?? 0,
    se: Math.sqrt(u + v),
    law: StudentT(df),
    df,
    n: m + n,
    options,
    effect: { name: "Cohen's d", value: cohensD(a, b) },
  })
}

/**
 * The $z$-test of $H_0: \mu = \mu_0$ (`mu`, default 0) for a sample with a known standard deviation $\sigma$
 * (`sigma`): $z = (\bar x - \mu_0)/(\sigma/\sqrt n)$, standard normal under the null. With `y`, the two-sample form
 * $z = (\bar x - \bar y - \delta_0)/\sqrt{\sigma^2/m + \sigma_y^2/n}$, with $\sigma_y$ = `sigmaY` (default
 * $\sigma$). The effect size is the difference over $\sigma$. Throws `DomainError` unless $\sigma > 0$.
 *
 * @param x The sample (one value is enough).
 * @param options The alternative and confidence level; `mu`, the mean (or difference of means) under the null;
 *   `sigma`, the known standard deviation (required); `y`, a second sample for the two-sample form; and `sigmaY`, the
 *   second sample's known standard deviation.
 * @returns The test result, with the standard normal as its null law.
 *
 * @example The six values above with a known standard deviation of 0.5
 * const r = zTest([5.1, 4.9, 5.6, 5.8, 6.0, 5.3], { mu: 5, sigma: 0.5 })
 * print('z =', r.statistic, ' p =', r.pValue)
 * print('95% interval:', r.ci)
 *
 * @example Two samples with a common known standard deviation
 * const r = zTest([10.4, 9.7, 10.9, 10.2], { y: [9.1, 9.6, 8.8, 9.5], sigma: 0.5 })
 * print('z =', r.statistic, ' p =', r.pValue, ' difference =', r.estimate)
 */
export function zTest(
  x: VectorLike,
  options: TestOptions & { mu?: number; sigma: number; y?: VectorLike; sigmaY?: number },
): TestResult {
  const a = sample(x, 'zTest')
  const { sigma } = options
  if (!(sigma > 0)) throw new DomainError('zTest', 'zTest: the known standard deviation must be positive')
  const p = meanAndVariance(a)
  const two = options.y !== undefined
  const b = two ? sample(options.y!, 'zTest') : null
  const sy = options.sigmaY ?? sigma
  const se = b ? Math.sqrt(sigma ** 2 / a.length + sy ** 2 / b.length) : sigma / Math.sqrt(a.length)
  const estimate = b ? p.mean - meanAndVariance(b).mean : p.mean
  const nullValue = options.mu ?? 0
  return pivotTest({
    test: 'zTest',
    method: b ? 'Two-sample z-test (known σ)' : 'One-sample z-test (known σ)',
    symbol: 'z',
    estimand: b ? 'difference of means' : 'mean',
    estimate,
    nullValue,
    se,
    law: Normal(0, 1),
    n: a.length + (b?.length ?? 0),
    options,
    effect: { name: b ? 'standardised difference' : "Cohen's d", value: (estimate - nullValue) / sigma },
  })
}

// ── Intervals ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The confidence interval for a mean: $\bar x \pm t_{n-1} s/\sqrt n$ (Student), or $\bar x \pm z \sigma/\sqrt n$
 * with a known $\sigma$ (`sigma`). One-sided for `alternative` less or greater.
 *
 * @param x The sample: at least two values, or one with `sigma`.
 * @param options The interval.
 * @param options.level The confidence level, in $(0, 1)$.
 * @param options.alternative `two-sided`, `less` (an upper bound) or `greater` (a lower bound).
 * @param options.sigma The known standard deviation; when left out, the sample's is used with Student's $t$.
 * @returns The interval.
 *
 * @example Student's interval, and the narrower one with a known standard deviation
 * const x = [5.1, 4.9, 5.6, 5.8, 6.0, 5.3]
 * print('t interval:', meanInterval(x))
 * print('z interval, sigma = 0.5:', meanInterval(x, { sigma: 0.5 }))
 * print('lower bound only:', meanInterval(x, { alternative: 'greater' }))
 */
export function meanInterval(
  x: VectorLike,
  { level = 0.95, alternative = 'two-sided', sigma }: TestOptions & { sigma?: number } = {},
): Interval {
  const v = sample(x, 'meanInterval', sigma ? 1 : 2)
  const n = v.length
  const { mean, variance } = meanAndVariance(v)
  return sigma
    ? pivotInterval(mean, sigma / Math.sqrt(n), Normal(0, 1), level, alternative)
    : pivotInterval(mean, Math.sqrt(variance / n), StudentT(n - 1), level, alternative)
}

/**
 * The confidence interval for a difference of means $\mu_x - \mu_y$ from independent samples: Welch's (default) or,
 * with `equalVariances`, the pooled-variance interval. It is the `ci` of `welchTTest` or `pooledTTest`.
 *
 * @param x The first sample, at least two values.
 * @param y The second sample, at least two values.
 * @param options The interval.
 * @param options.level The confidence level, in $(0, 1)$.
 * @param options.alternative `two-sided`, `less` (an upper bound) or `greater` (a lower bound).
 * @param options.equalVariances Assume a common variance and use the pooled interval.
 * @returns The interval for $\mu_x - \mu_y$.
 *
 * @example Welch's and the pooled interval for two groups
 * const x = [20.1, 22.4, 19.8, 21.5, 23.0]
 * const y = [18.2, 19.0, 17.5, 18.8, 19.9, 18.4]
 * print('Welch:', differenceOfMeansInterval(x, y))
 * print('pooled:', differenceOfMeansInterval(x, y, { equalVariances: true }))
 */
export function differenceOfMeansInterval(
  x: VectorLike,
  y: VectorLike,
  { level = 0.95, alternative = 'two-sided', equalVariances = false }: TestOptions & { equalVariances?: boolean } = {},
): Interval {
  const t = equalVariances ? pooledTTest(x, y, { level, alternative }) : welchTTest(x, y, { level, alternative })
  return t.ci!
}
