/**
 * Tests and intervals for means: the one-sample, paired, pooled and Welch t-tests (scipy's `ttest_1samp`,
 * `ttest_rel`, `ttest_ind`), the z-test with a known standard deviation, and the confidence intervals for a mean and
 * a difference of means. Each statistic is a standardised difference whose null law is Student's t (or the standard
 * normal), so the interval is the same pivot inverted.
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

/** A pivot-based test: statistic (estimate − null)/se against `law`, its interval inverted from the same pivot. */
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
 * The one-sample t-test of H₀: μ = `mu` (default 0) for a normal sample x of size n ≥ 2: t = (x̄ − μ₀)/(s/√n) with
 * n − 1 degrees of freedom; the interval is for the mean, the effect size Cohen's d = (x̄ − μ₀)/s.
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

/** Differences x − y of two equal-length samples. */
function differences(x: VectorLike, y: VectorLike, where: string): Float64Array {
  const a = sample(x, where, 2)
  const b = sample(y, where, 2)
  if (a.length !== b.length) throw new DomainError(where, `${where}: paired samples must have equal lengths`)
  return a.map((v, i) => v - b[i])
}

/**
 * The paired t-test: the one-sample t-test of the differences x − y against `mu` (default 0), n − 1 degrees of
 * freedom. The interval is for the mean difference; the effect size is d_z, the mean difference over the standard
 * deviation of the differences.
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
 * The two-sample t-test with a pooled variance (Student's): for independent normal samples of sizes m and n with a
 * common variance, t = (x̄ − ȳ − δ₀)/(s_p √(1/m + 1/n)) with m + n − 2 degrees of freedom, δ₀ = `mu` (default 0).
 * The interval is for the difference of means, the effect size Cohen's d.
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

/** The Welch–Satterthwaite degrees of freedom (u + v)²/(u²/(m − 1) + v²/(n − 1)), u = s_x²/m and v = s_y²/n. */
export function welchDegreesOfFreedom(u: number, v: number, m: number, n: number): number {
  return (u + v) ** 2 / (u ** 2 / (m - 1) + v ** 2 / (n - 1))
}

/**
 * Welch's two-sample t-test (Welch, 1947): no common variance is assumed; t = (x̄ − ȳ − δ₀)/√(s_x²/m + s_y²/n), and
 * its null law is approximated by Student's t with the Welch–Satterthwaite degrees of freedom. The interval is for the
 * difference of means, the effect size Cohen's d (pooled standard deviation).
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
 * The z-test of H₀: μ = `mu` for a sample with a known standard deviation σ (`sigma`): z = (x̄ − μ₀)/(σ/√n), standard
 * normal under the null. With `y`, the two-sample form z = (x̄ − ȳ − δ₀)/(σ √(1/m + 1/n)) for a common known σ (or
 * `sigmaY` for the second sample).
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
 * The confidence interval for a mean: x̄ ± t_{n−1} s/√n (Student), or x̄ ± z σ/√n with a known σ (`sigma`). One-sided
 * for `alternative` less or greater.
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
 * The confidence interval for a difference of means μ_x − μ_y from independent samples: Welch's (default) or, with
 * `equalVariances`, the pooled-variance interval.
 */
export function differenceOfMeansInterval(
  x: VectorLike,
  y: VectorLike,
  { level = 0.95, alternative = 'two-sided', equalVariances = false }: TestOptions & { equalVariances?: boolean } = {},
): Interval {
  const t = equalVariances ? pooledTTest(x, y, { level, alternative }) : welchTTest(x, y, { level, alternative })
  return t.ci!
}
