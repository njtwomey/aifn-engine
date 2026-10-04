/**
 * Rank tests: the Mann–Whitney U test of two independent samples (scipy's `mannwhitneyu`) and the Wilcoxon
 * signed-rank test of paired differences (scipy's `wilcoxon`, zero method `wilcox`). Without ties each statistic has
 * an exact null law, built here as a `Categorical` over its integer values by a recursion on the sample size; with
 * ties (or large samples) the normal approximation with the tie-corrected variance is used, and `method: 'exact'`
 * with ties throws.
 *
 * Two conventions differ from scipy 1.18. `wilcoxonSignedRank` reports T⁺ for every alternative (as R's `wilcox.test`
 * reports V), where scipy's two-sided result reports min(T⁺, T⁻); the p-values agree. With ties or zero differences
 * and n ≤ 50, scipy's `auto` runs a permutation test over the actual midranks, and here the normal approximation
 * is used (the p-values differ by a few hundredths at small n).
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { tensor, toFlat, type VectorLike } from 'aifn-compute/foundation/tensor'
import { Categorical, Normal, type Univariate } from 'aifn-compute/probability/distributions'
import { ranks } from 'aifn-compute/probability/stats'
import { pValueOf, result, sample, tailOf, type Alternative, type TestOptions, type TestResult } from './protocol'

/** How a rank test computes its p-value: the exact null law, the normal approximation, or `auto` (as scipy). */
export type RankMethod = 'auto' | 'exact' | 'asymptotic'

/**
 * The exact null law of the Mann–Whitney U for samples of sizes m and n without ties, P(U = u) for u = 0 … mn: the
 * coefficients of the Gaussian binomial [m + n choose m]_q over C(m + n, m), built by the q-Pascal rule
 * [k, j] = [k − 1, j − 1] + qʲ [k − 1, j] in probability form, P_{k,j}(u) = (j/k) P_{k−1,j−1}(u) +
 * ((k − j)/k) P_{k−1,j}(u − j), which adds only positive terms. O((m + n) m · mn) operations.
 */
export function mannWhitneyNull(m: number, n: number): Univariate {
  // row[j]: P_{k,j} for the current k, a distribution on 0 … j(k − j).
  let row: Float64Array[] = [Float64Array.of(1)]
  for (let k = 1; k <= m + n; k++) {
    const next: Float64Array[] = []
    for (let j = 0; j <= Math.min(k, m); j++) {
      const out = new Float64Array(j * (k - j) + 1)
      if (j >= 1) {
        const a = row[j - 1]
        for (let u = 0; u < a.length; u++) out[u] += (j / k) * a[u]
      }
      if (j <= k - 1 && row[j]) {
        const b = row[j]
        for (let u = 0; u < b.length; u++) out[u + j] += ((k - j) / k) * b[u]
      }
      next.push(out)
    }
    row = next
  }
  return Categorical(tensor(Array.from(row[m])))
}

/**
 * The exact null law of the Wilcoxon signed-rank statistic T⁺ (the sum of the ranks of the positive differences) for
 * n differences without ties or zeros: each rank is positive with probability ½ independently, so
 * P_k(t) = ½ P_{k−1}(t) + ½ P_{k−1}(t − k), on 0 … n(n + 1)/2.
 */
export function signedRankNull(n: number): Univariate {
  let p = Float64Array.of(1)
  for (let k = 1; k <= n; k++) {
    const next = new Float64Array(p.length + k)
    for (let t = 0; t < p.length; t++) {
      next[t] += 0.5 * p[t]
      next[t + k] += 0.5 * p[t]
    }
    p = next
  }
  return Categorical(tensor(Array.from(p)))
}

/** Σ (t³ − t) over the groups of tied values, for the tie-corrected variances. */
function tieSum(values: Float64Array): number {
  const v = Float64Array.from(values).sort()
  let s = 0
  for (let i = 0; i < v.length;) {
    let j = i
    while (j + 1 < v.length && v[j + 1] === v[i]) j++
    const t = j - i + 1
    s += t ** 3 - t
    i = j + 1
  }
  return s
}

/**
 * The p-value of a normal approximation with a continuity correction: the statistic is moved half a unit towards the
 * mean before standardising, z = (s − μ ∓ ½)/σ.
 */
function correctedNormal(s: number, mu: number, sd: number, alternative: Alternative, correction: boolean): number {
  const cc = correction ? 0.5 : 0
  const std = Normal(0, 1)
  let p: number
  if (alternative === 'greater') p = std.survival((s - mu - cc) / sd) as number
  else if (alternative === 'less') p = std.cdf((s - mu + cc) / sd) as number
  else p = 2 * (std.survival((Math.abs(s - mu) - cc) / sd) as number)
  return Math.min(1, Math.max(0, p))
}

/**
 * The Mann–Whitney U test (Mann and Whitney, 1947; Wilcoxon's rank-sum test) of two independent samples: U = R₁ −
 * m(m + 1)/2, with R₁ the sum of the pooled ranks of x (ties averaged), counts the pairs with xᵢ > yⱼ (ties count ½).
 * `greater` tests x stochastically larger than y. `exact` uses the exact null law (no ties); `asymptotic` the normal
 * law N(mn/2, mn/12 · (N + 1 − Σ(t³ − t)/(N(N − 1)))) with a continuity correction unless `continuity` is false;
 * `auto` is exact when there are no ties and either sample has at most 8 values. The effect size is the rank-biserial
 * correlation 2U/(mn) − 1.
 */
export function mannWhitneyU(
  x: VectorLike,
  y: VectorLike,
  options: TestOptions & { method?: RankMethod; continuity?: boolean } = {},
): TestResult {
  const a = sample(x, 'mannWhitneyU')
  const b = sample(y, 'mannWhitneyU')
  const [m, n] = [a.length, b.length]
  const pooled = new Float64Array(m + n)
  pooled.set(a)
  pooled.set(b, m)
  const r = toFlat(ranks(pooled))
  let r1 = 0
  for (let i = 0; i < m; i++) r1 += r[i]
  const U = r1 - (m * (m + 1)) / 2
  const ties = tieSum(pooled)
  const alternative = options.alternative ?? 'two-sided'
  let method = options.method ?? 'auto'
  if (method === 'auto') method = (m > 8 && n > 8) || ties > 0 ? 'asymptotic' : 'exact'
  else if (method === 'exact' && ties > 0)
    throw new DomainError('mannWhitneyU', "mannWhitneyU: the exact law assumes no ties; use method 'asymptotic'")
  const N = m + n
  const mu = (m * n) / 2
  const sd = Math.sqrt(((m * n) / 12) * (N + 1 - ties / (N * (N - 1))))
  const exact = method === 'exact'
  const law = exact ? mannWhitneyNull(m, n) : Normal(mu, sd)
  const tail = tailOf(alternative)
  const continuity = options.continuity ?? true
  return result({
    test: 'mannWhitneyU',
    method: `Mann–Whitney U test (${exact ? 'exact' : continuity ? 'normal approximation, continuity-corrected' : 'normal approximation'})`,
    statistic: U,
    symbol: 'U',
    pValue: exact ? pValueOf(law, U, tail) : correctedNormal(U, mu, sd, alternative, continuity),
    alternative,
    tail,
    null: law,
    effectSize: { name: 'rank-biserial correlation', value: (2 * U) / (m * n) - 1 },
    n: N,
  })
}

/**
 * The Wilcoxon signed-rank test (Wilcoxon, 1945) of H₀: the differences d = x − y (or x alone) are symmetric about
 * `mu` (default 0). Zero differences are dropped (Wilcoxon's rule); the absolute differences are ranked (ties
 * averaged) and T⁺ is the sum of the ranks of the positive ones. `greater` tests a shift above `mu`. `exact` uses the
 * exact null law (no ties or zeros); `asymptotic` the normal law N(n(n + 1)/4, n(n + 1)(2n + 1)/24 − Σ(t³ − t)/48),
 * with a continuity correction when `correction` is true (default false, as scipy); `auto` is exact when there are no
 * ties or zeros and n ≤ 50. The effect size is the matched-pairs rank-biserial correlation (T⁺ − T⁻)/(T⁺ + T⁻).
 */
export function wilcoxonSignedRank(
  x: VectorLike,
  y?: VectorLike | null,
  options: TestOptions & { mu?: number; method?: RankMethod; correction?: boolean } = {},
): TestResult {
  const a = sample(x, 'wilcoxonSignedRank')
  const b = y ? sample(y, 'wilcoxonSignedRank') : null
  if (b && b.length !== a.length)
    throw new DomainError('wilcoxonSignedRank', 'wilcoxonSignedRank: paired samples must have equal lengths')
  const mu0 = options.mu ?? 0
  const all = a.map((v, i) => v - (b ? b[i] : 0) - mu0)
  const d = all.filter((v) => v !== 0)
  const zeros = all.length - d.length
  const n = d.length
  if (n === 0) throw new DomainError('wilcoxonSignedRank', 'wilcoxonSignedRank: every difference is zero')
  const abs = d.map(Math.abs)
  const r = toFlat(ranks(abs))
  let plus = 0
  for (let i = 0; i < n; i++) if (d[i] > 0) plus += r[i]
  const total = (n * (n + 1)) / 2
  const ties = tieSum(abs)
  const alternative = options.alternative ?? 'two-sided'
  let method = options.method ?? 'auto'
  if (method === 'auto') method = ties === 0 && zeros === 0 && n <= 50 ? 'exact' : 'asymptotic'
  else if (method === 'exact' && (ties > 0 || zeros > 0))
    throw new DomainError(
      'wilcoxonSignedRank',
      "wilcoxonSignedRank: the exact law assumes no ties or zero differences; use method 'asymptotic'",
    )
  const exact = method === 'exact'
  const mu = total / 2
  const sd = Math.sqrt((n * (n + 1) * (2 * n + 1)) / 24 - ties / 48)
  const law = exact ? signedRankNull(n) : Normal(mu, sd)
  const tail = tailOf(alternative)
  const correction = options.correction ?? false
  return result({
    test: 'wilcoxonSignedRank',
    method: `Wilcoxon signed-rank test (${exact ? 'exact' : correction ? 'normal approximation, continuity-corrected' : 'normal approximation'})`,
    statistic: plus,
    symbol: 'T^+',
    pValue: exact ? pValueOf(law, plus, tail) : correctedNormal(plus, mu, sd, alternative, correction),
    alternative,
    tail,
    null: law,
    effectSize: { name: 'matched-pairs rank-biserial correlation', value: (2 * plus - total) / total },
    n,
  })
}
