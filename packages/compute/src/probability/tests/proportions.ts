/**
 * Tests and intervals for proportions: the exact binomial test (scipy's `binomtest`), the two-proportion z-test,
 * the Wald, Wilson (with and without continuity correction) and Clopper–Pearson intervals for one proportion (scipy's
 * `proportion_ci`), and the Wald and Newcombe intervals for a difference of proportions.
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

/** How an interval for one proportion is built. */
export type ProportionIntervalMethod = 'wald' | 'wilson' | 'wilson-cc' | 'clopper-pearson'

function counts(k: number, n: number, where: string): void {
  if (!(Number.isInteger(n) && n >= 1)) throw new DomainError(where, `${where}: n must be a positive integer`)
  if (!(Number.isInteger(k) && k >= 0 && k <= n))
    throw new DomainError(where, `${where}: k must be an integer in [0, n]`)
}

const standardNormal = Normal(0, 1)
const z = (p: number) => standardNormal.isf(p) as number

/**
 * A confidence interval for a binomial proportion from k successes in n trials. `wald`: p̂ ± z √(p̂(1 − p̂)/n), cut to
 * [0, 1]; `wilson`: the score interval (Wilson, 1927), the p whose score test does not reject; `wilson-cc`: with
 * the continuity correction (Newcombe, 1998); `clopper-pearson`: the exact interval from beta quantiles (Clopper and
 * Pearson, 1934), which always covers at least `level`. One-sided for `alternative` less (lower end 0) or greater
 * (upper end 1). Wilson's formulas follow scipy's `proportion_ci`.
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

/** Cohen's h = 2 asin √p₁ − 2 asin √p₂, the difference of proportions on the variance-stabilising scale. */
export function cohensH(p1: number, p2: number): number {
  return 2 * Math.asin(Math.sqrt(p1)) - 2 * Math.asin(Math.sqrt(p2))
}

/**
 * The exact binomial test of H₀: P(success) = `p` (default ½) from k successes in n trials. The statistic is k, with
 * null law Binomial(n, p); a one-sided p-value is a binomial tail, and the two-sided one sums the probabilities of every
 * outcome no more likely than k (scipy's `binomtest`). The interval is for the proportion (Clopper–Pearson unless
 * `interval` names another method); the effect size is Cohen's h of k/n against p.
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
 * The two-proportion z-test of H₀: p₁ = p₂ from k₁ of n₁ and k₂ of n₂: z = (p̂₁ − p̂₂)/√(p̄(1 − p̄)(1/n₁ + 1/n₂)) with
 * the pooled p̄ = (k₁ + k₂)/(n₁ + n₂), standard normal under the null (z² is the uncorrected χ² of the 2 × 2 table).
 * The interval is the Wald interval for p₁ − p₂ (unpooled standard error); the effect size is Cohen's h.
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
 * A confidence interval for p₁ − p₂ from independent binomial samples: `wald`, (p̂₁ − p̂₂) ± z √(p̂₁q̂₁/n₁ + p̂₂q̂₂/n₂);
 * or `newcombe`, the hybrid score interval (Newcombe, 1998, method 10) that combines the two Wilson intervals
 * (lᵢ, uᵢ): [d − √((p̂₁ − l₁)² + (u₂ − p̂₂)²), d + √((u₁ − p̂₁)² + (p̂₂ − l₂)²)], which stays inside [−1, 1].
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
