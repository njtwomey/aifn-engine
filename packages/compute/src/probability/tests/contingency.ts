/**
 * Tests on counts: Pearson's χ² and the G-test (likelihood ratio) for goodness of fit and for independence in a
 * contingency table, both members of Cressie and Read's power-divergence family (scipy's `power_divergence`,
 * `chisquare`, `chi2_contingency`); Fisher's exact test of a 2 × 2 table (scipy's `fisher_exact`); and the table's
 * effect sizes, the odds ratio (sample, or conditional maximum likelihood with its exact interval, as scipy's
 * `contingency.odds_ratio`) and Cramér's V.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type MatrixLike, type VectorLike } from 'aifn-compute/foundation/tensor'
import { logChoose } from 'aifn-compute/numerics/special'
import { ChiSquare, Hypergeometric, Normal } from 'aifn-compute/probability/distributions'
import {
  checkLevel,
  pValueOf,
  result,
  tailOf,
  type Alternative,
  type Interval,
  type TestOptions,
  type TestResult,
} from './protocol'

const xlogy = (x: number, y: number) => (x === 0 ? 0 : x * Math.log(y))

/**
 * The Cressie–Read power divergence 2/(λ(λ + 1)) Σ o[(o/e)^λ − 1] between observed and expected counts: λ = 1 is
 * Pearson's Σ (o − e)²/e, λ = 0 its limit 2 Σ o log(o/e), the G (log-likelihood ratio) statistic.
 */
export function powerDivergence(observed: ArrayLike<number>, expected: ArrayLike<number>, lambda: number): number {
  let s = 0
  for (let i = 0; i < observed.length; i++) {
    const o = observed[i]
    const e = expected[i]
    if (lambda === 1) s += (o - e) ** 2 / e
    else if (lambda === 0) s += 2 * xlogy(o, o / e)
    else s += (2 / (lambda * (lambda + 1))) * o * ((o / e) ** lambda - 1)
  }
  return s
}

/** Observed counts as a Float64Array of non-negative values. */
function countsOf(x: VectorLike, where: string): Float64Array {
  const v = dense.toF64(x, where)
  if (v.length < 2) throw new DomainError(where, `${where}: needs at least two categories`)
  for (const a of v) if (!(a >= 0)) throw new DomainError(where, `${where}: counts must be non-negative`)
  return v
}

type GoodnessOptions = {
  /** Expected counts (summing to the observed total), or omitted for equal expected counts. */
  expected?: VectorLike
  /** Category probabilities, an alternative to `expected` (normalised here). */
  probabilities?: VectorLike
  /** Degrees of freedom lost to parameters estimated from the data (df = k − 1 − ddof). */
  ddof?: number
}

function goodnessOfFit(observed: VectorLike, options: GoodnessOptions, lambda: 0 | 1, test: string): TestResult {
  const o = countsOf(observed, test)
  const total = o.reduce((a, b) => a + b, 0)
  let e: Float64Array
  if (options.expected) {
    e = dense.toF64(options.expected, test)
    const se = e.reduce((a, b) => a + b, 0)
    if (Math.abs(se - total) > 1e-8 * total)
      throw new DomainError(test, `${test}: expected counts must sum to the observed total ${total} (got ${se})`)
  } else if (options.probabilities) {
    const p = dense.toF64(options.probabilities, test)
    const sp = p.reduce((a, b) => a + b, 0)
    e = p.map((v) => (v / sp) * total)
  } else e = new Float64Array(o.length).fill(total / o.length)
  if (e.length !== o.length) throw new DomainError(test, `${test}: observed and expected lengths differ`)
  const df = o.length - 1 - (options.ddof ?? 0)
  const statistic = powerDivergence(o, e, lambda)
  const law = ChiSquare(df)
  return result({
    test,
    method: lambda === 1 ? 'Pearson χ² goodness-of-fit test' : 'G-test of goodness of fit',
    statistic,
    symbol: lambda === 1 ? '\\chi^2' : 'G',
    df,
    pValue: pValueOf(law, statistic, 'upper'),
    alternative: 'two-sided',
    tail: 'upper',
    null: law,
    effectSize: { name: "Cohen's w", value: Math.sqrt(statistic / total) },
    n: total,
  })
}

/**
 * Pearson's χ² goodness-of-fit test (Pearson, 1900): X² = Σ (oᵢ − eᵢ)²/eᵢ over k categories, χ²(k − 1 − ddof) under
 * the null (asymptotically; every eᵢ should be at least about 5). Effect size: Cohen's w = √(X²/N).
 */
export function chiSquareGoodnessOfFit(observed: VectorLike, options: GoodnessOptions = {}): TestResult {
  return goodnessOfFit(observed, options, 1, 'chiSquareGoodnessOfFit')
}

/** The G-test of goodness of fit: G = 2 Σ oᵢ log(oᵢ/eᵢ), with the same χ² null as Pearson's test. */
export function gTestGoodnessOfFit(observed: VectorLike, options: GoodnessOptions = {}): TestResult {
  return goodnessOfFit(observed, options, 0, 'gTestGoodnessOfFit')
}

/** A contingency table with its expected counts under independence, eᵢⱼ = rᵢ cⱼ / N, and degrees of freedom. */
export type Table = {
  rows: number
  cols: number
  observed: Float64Array
  expected: Float64Array
  total: number
  df: number
}

/** The expected counts rᵢcⱼ/N of an r × c table under independence, and its (r − 1)(c − 1) degrees of freedom. */
export function expectedCounts(table: MatrixLike): Table {
  const { data, m, n } = dense.toMatrixF64(table, 'expectedCounts')
  if (m < 2 || n < 2) throw new DomainError('expectedCounts', 'expectedCounts: the table needs at least 2 × 2 cells')
  for (const a of data)
    if (!(a >= 0)) throw new DomainError('expectedCounts', 'expectedCounts: counts must be non-negative')
  const r = new Float64Array(m)
  const c = new Float64Array(n)
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      r[i] += data[i * n + j]
      c[j] += data[i * n + j]
    }
  const total = r.reduce((a, b) => a + b, 0)
  const expected = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) expected[i * n + j] = (r[i] * c[j]) / total
  if (expected.some((e) => e === 0))
    throw new DomainError('expectedCounts', 'expectedCounts: a row or column of the table sums to zero')
  return { rows: m, cols: n, observed: data, expected, total, df: (m - 1) * (n - 1) }
}

function independence(table: MatrixLike, correction: boolean, lambda: 0 | 1, test: string): TestResult {
  const t = expectedCounts(table)
  let o = t.observed
  // Yates' correction (one degree of freedom only): move each count half a unit towards its expectation, never past it.
  const corrected = correction && t.df === 1
  if (corrected) o = o.map((v, i) => v + Math.sign(t.expected[i] - v) * Math.min(0.5, Math.abs(t.expected[i] - v)))
  const statistic = powerDivergence(o, t.expected, lambda)
  const law = ChiSquare(t.df)
  const name = lambda === 1 ? 'Pearson χ² test of independence' : 'G-test of independence'
  return result({
    test,
    method: corrected ? `${name} (Yates-corrected)` : name,
    statistic,
    symbol: lambda === 1 ? '\\chi^2' : 'G',
    df: t.df,
    pValue: pValueOf(law, statistic, 'upper'),
    alternative: 'two-sided',
    tail: 'upper',
    null: law,
    effectSize: { name: "Cramér's V", value: cramersV(table) },
    n: t.total,
  })
}

/**
 * Pearson's χ² test of independence of the rows and columns of an r × c table: X² = Σ (oᵢⱼ − eᵢⱼ)²/eᵢⱼ with
 * eᵢⱼ = rᵢcⱼ/N, χ²((r − 1)(c − 1)) under the null. For a 2 × 2 table Yates' continuity correction is applied unless
 * `correction` is false (as scipy's `chi2_contingency`). Effect size: Cramér's V (uncorrected).
 */
export function chiSquareIndependence(
  table: MatrixLike,
  { correction = true }: { correction?: boolean } = {},
): TestResult {
  return independence(table, correction, 1, 'chiSquareIndependence')
}

/** The G-test of independence: G = 2 Σ oᵢⱼ log(oᵢⱼ/eᵢⱼ) (Yates-corrected counts for 2 × 2 unless `correction` is false). */
export function gTestIndependence(table: MatrixLike, { correction = true }: { correction?: boolean } = {}): TestResult {
  return independence(table, correction, 0, 'gTestIndependence')
}

/**
 * Cramér's V = √(X²/(N (min(r, c) − 1))) for an r × c table, from the uncorrected Pearson X²: 0 for independence, 1
 * for a perfect association (scipy's `contingency.association(method='cramer')`). The
 * registered metric is `aifn-compute/learning/metrics`' `cramersV`; this is the effect size the independence tests report.
 */
function cramersV(table: MatrixLike): number {
  const t = expectedCounts(table)
  return Math.sqrt(powerDivergence(t.observed, t.expected, 1) / (t.total * (Math.min(t.rows, t.cols) - 1)))
}

// ── 2 × 2 tables ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** The cells a, b, c, d of a 2 × 2 table [[a, b], [c, d]] as non-negative integers. */
function twoByTwo(table: MatrixLike, where: string): [number, number, number, number] {
  const { data, m, n } = dense.toMatrixF64(table, where)
  if (m !== 2 || n !== 2) throw new DomainError(where, `${where}: needs a 2 × 2 table`)
  for (const v of data)
    if (!(Number.isInteger(v) && v >= 0)) throw new DomainError(where, `${where}: cells must be non-negative integers`)
  return [data[0], data[1], data[2], data[3]]
}

/**
 * Fisher's noncentral hypergeometric law of a, the top-left cell given the margins, at odds ratio ψ:
 * P(a = x) ∝ C(K, x) C(N − K, n − x) ψˣ over max(0, n + K − N) ≤ x ≤ min(n, K), with N the total, K the first column's
 * total and n the first row's. Returned as the support's lower end and the probabilities.
 */
function noncentralHypergeometric(N: number, K: number, n: number, logPsi: number): { lo: number; p: Float64Array } {
  const lo = Math.max(0, n + K - N)
  const hi = Math.min(n, K)
  const logw = new Float64Array(hi - lo + 1)
  let top = -Infinity
  for (let x = lo; x <= hi; x++) {
    const w = (logChoose(K, x) as number) + (logChoose(N - K, n - x) as number) + x * logPsi
    logw[x - lo] = w
    top = Math.max(top, w)
  }
  const p = logw.map((w) => Math.exp(w - top))
  const s = p.reduce((a, b) => a + b, 0)
  return { lo, p: p.map((v) => v / s) }
}

/** Solve f(log ψ) = target for an f increasing in log ψ, by bisection on [−60, 60] (ψ from e⁻⁶⁰ to e⁶⁰). */
function solveLogPsi(f: (logPsi: number) => number, target: number): number {
  let lo = -60
  let hi = 60
  for (let i = 0; i < 200 && hi - lo > 1e-13; i++) {
    const mid = (lo + hi) / 2
    if (f(mid) < target) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

/**
 * The odds ratio of a 2 × 2 table [[a, b], [c, d]] with a confidence interval. `sample`: ad/(bc), with Woolf's
 * interval exp(log OR ± z √(1/a + 1/b + 1/c + 1/d)); `conditional` (default): the conditional maximum-likelihood
 * estimate, the ψ at which the noncentral hypergeometric mean of a equals the observed a, with the exact interval that
 * inverts the two one-sided Fisher tests (Cornfield, 1956), as scipy's `odds_ratio`.
 */
export function oddsRatio(
  table: MatrixLike,
  {
    kind = 'conditional',
    level = 0.95,
    alternative = 'two-sided',
  }: { kind?: 'sample' | 'conditional'; level?: number; alternative?: Alternative } = {},
): { value: number; ci: Interval } {
  checkLevel(level, 'oddsRatio')
  const [a, b, c, d] = twoByTwo(table, 'oddsRatio')
  if (kind === 'sample') {
    const value = (a * d) / (b * c)
    if (a === 0 || b === 0 || c === 0 || d === 0) return { value, ci: { lower: 0, upper: Infinity, level } }
    const se = Math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    const q = (Normal(0, 1).isf(alternative === 'two-sided' ? (1 - level) / 2 : 1 - level) as number) * se
    const l = Math.log(value)
    return {
      value,
      ci: {
        lower: alternative === 'less' ? 0 : Math.exp(l - q),
        upper: alternative === 'greater' ? Infinity : Math.exp(l + q),
        level,
      },
    }
  }
  const N = a + b + c + d
  const K = a + c
  const n = a + b
  const mean = (logPsi: number) => {
    const { lo, p } = noncentralHypergeometric(N, K, n, logPsi)
    return p.reduce((s, v, i) => s + v * (lo + i), 0)
  }
  // P_ψ(A ≥ a) and P_ψ(A ≤ a), both monotone in ψ (increasing and decreasing).
  const upperTail = (logPsi: number) => {
    const { lo, p } = noncentralHypergeometric(N, K, n, logPsi)
    return p.slice(a - lo).reduce((s, v) => s + v, 0)
  }
  const lowerTail = (logPsi: number) => {
    const { lo, p } = noncentralHypergeometric(N, K, n, logPsi)
    return p.slice(0, a - lo + 1).reduce((s, v) => s + v, 0)
  }
  const lo = Math.max(0, n + K - N)
  const hi = Math.min(n, K)
  const value = a === lo ? 0 : a === hi ? Infinity : Math.exp(solveLogPsi(mean, a))
  const tailMass = alternative === 'two-sided' ? (1 - level) / 2 : 1 - level
  const lower = alternative === 'less' || a === lo ? 0 : Math.exp(solveLogPsi(upperTail, tailMass))
  const upper =
    alternative === 'greater' || a === hi ? Infinity : Math.exp(solveLogPsi((l) => -lowerTail(l), -tailMass))
  return { value, ci: { lower, upper, level } }
}

/**
 * Fisher's exact test of a 2 × 2 table [[a, b], [c, d]] (Fisher, 1922): given the margins, a follows the
 * hypergeometric law Hypergeometric(N, a + c, a + b) when rows and columns are independent (odds ratio 1). The
 * statistic is a; `greater` tests an odds ratio above 1 (large a), and the two-sided p-value sums every table no more
 * likely than the observed one. The effect size is the conditional maximum-likelihood odds ratio with its exact
 * interval.
 */
export function fisherExact(table: MatrixLike, options: TestOptions = {}): TestResult {
  const [a, b, c, d] = twoByTwo(table, 'fisherExact')
  const alternative = options.alternative ?? 'two-sided'
  const tail = alternative === 'two-sided' ? 'likelihood' : tailOf(alternative)
  const law = Hypergeometric(a + b + c + d, a + c, a + b)
  const or = oddsRatio(table, { kind: 'conditional', level: options.level ?? 0.95, alternative })
  return result({
    test: 'fisherExact',
    method: "Fisher's exact test",
    statistic: a,
    symbol: 'a',
    pValue: pValueOf(law, a, tail),
    alternative,
    tail,
    null: law,
    estimand: 'odds ratio',
    estimate: or.value,
    nullValue: 1,
    ci: or.ci,
    effectSize: { name: 'odds ratio (conditional MLE)', value: or.value, ci: or.ci },
    n: a + b + c + d,
  })
}
