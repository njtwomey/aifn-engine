/**
 * Tests on counts: Pearson's $\chi^2$ and the $G$-test (likelihood ratio) for goodness of fit and for independence in
 * a contingency table, both members of Cressie and Read's power-divergence family (scipy's `power_divergence`,
 * `chisquare`, `chi2_contingency`); Fisher's exact test of a $2 \times 2$ table (scipy's `fisher_exact`); and the
 * table's effect sizes, the odds ratio (sample, or conditional maximum likelihood with its exact interval, as scipy's
 * `contingency.odds_ratio`) and Cramér's $V$.
 *
 * The $\chi^2$ and $G$ tests compare observed counts $o$ with the counts $e$ expected under the null, and their
 * $\chi^2$ null laws are asymptotic (every $e$ should be at least about 5). Fisher's test conditions on the table's
 * margins and is exact. Tables are matrices of non-negative counts, rows by columns; bad counts throw `DomainError`.
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

/**
 * $x \log y$, taken as 0 when $x = 0$ (whatever $y$ is), so that empty cells add nothing to $G$.
 *
 * @param x The factor, a count.
 * @param y The argument of the logarithm.
 * @returns $x \log y$, or 0.
 */
const xlogy = (x: number, y: number) => (x === 0 ? 0 : x * Math.log(y))

/**
 * The Cressie–Read power divergence $\frac{2}{\lambda(\lambda + 1)} \sum_i o_i[(o_i/e_i)^\lambda - 1]$ between
 * observed and expected counts: $\lambda = 1$ is Pearson's $\sum_i (o_i - e_i)^2/e_i$, $\lambda = 0$ its limit
 * $2\sum_i o_i \log(o_i/e_i)$, the $G$ (log-likelihood ratio) statistic, and $\lambda = 2/3$ Cressie and Read's
 * recommended compromise. $\lambda = -1$ is not supported (it divides by zero).
 *
 * @param observed The observed counts $o_i$.
 * @param expected The expected counts $e_i$, as many as `observed`, all positive.
 * @param lambda The power $\lambda$.
 * @returns The divergence, 0 when every $o_i = e_i$.
 *
 * @example Pearson's, the G and Cressie–Read's statistic for the same counts
 * const o = [10, 20, 30]
 * const e = [20, 20, 20]
 * print('lambda = 1 (Pearson):', powerDivergence(o, e, 1))
 * print('lambda = 0 (G):', powerDivergence(o, e, 0))
 * print('lambda = 2/3:', powerDivergence(o, e, 2 / 3))
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

/**
 * Observed counts as a Float64Array of non-negative values: throws `DomainError` for fewer than two categories or a
 * negative (or NaN) count.
 *
 * @param x The counts, one per category (need not be integers).
 * @param where The caller's name, for error messages.
 * @returns A new array of the counts.
 */
function countsOf(x: VectorLike, where: string): Float64Array {
  const v = dense.toF64(x, where)
  if (v.length < 2) throw new DomainError(where, `${where}: needs at least two categories`)
  for (const a of v) if (!(a >= 0)) throw new DomainError(where, `${where}: counts must be non-negative`)
  return v
}

/**
 * Options of the goodness-of-fit tests: what the null hypothesis expects, given as counts or as probabilities (equal
 * counts when both are left out; `expected` wins when both are given), and the degrees of freedom to remove.
 */
type GoodnessOptions = {
  /** Expected counts (summing to the observed total), or omitted for equal expected counts. */
  expected?: VectorLike
  /** Category probabilities, an alternative to `expected` (normalised here). */
  probabilities?: VectorLike
  /** Degrees of freedom lost to parameters estimated from the data: the test has $k - 1 - \text{ddof}$ (default 0). */
  ddof?: number
}

/**
 * The goodness-of-fit test with statistic `powerDivergence` at `lambda` against $\chi^2(k - 1 - \text{ddof})$, upper
 * tail. Throws `DomainError` when the expected counts do not sum to the observed total (to $10^{-8}$ relative) or
 * their number differs from the observed.
 *
 * @param observed The observed counts of the $k$ categories.
 * @param options The expected counts or probabilities, and `ddof`.
 * @param lambda 1 for Pearson's $X^2$, 0 for $G$.
 * @param test The registry key of the test, also used in error messages.
 * @returns The test result, with Cohen's $w = \sqrt{\text{statistic}/N}$ as its effect size.
 */
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
 * Pearson's $\chi^2$ goodness-of-fit test (Pearson, 1900): $X^2 = \sum_i (o_i - e_i)^2/e_i$ over $k$ categories,
 * $\chi^2(k - 1 - \text{ddof})$ under the null (asymptotically; every $e_i$ should be at least about 5). Effect size:
 * Cohen's $w = \sqrt{X^2/N}$ with $N$ the total count. As scipy's `chisquare`.
 *
 * @param observed The observed counts of the $k \ge 2$ categories.
 * @param options The null hypothesis's expected counts or probabilities (default: all categories equally likely),
 *   and `ddof`.
 * @returns The test result.
 *
 * @example Is a die fair? 88 rolls that look fair, and 60 that do not
 * const fair = chiSquareGoodnessOfFit([16, 18, 16, 14, 12, 12])
 * print('X2 =', fair.statistic, ' df =', fair.df, ' p =', fair.pValue)
 * const loaded = chiSquareGoodnessOfFit([5, 5, 5, 5, 5, 35])
 * print('X2 =', loaded.statistic, ' p =', loaded.pValue)
 *
 * @example Against given probabilities
 * // 100 plants against Mendel's 9 : 3 : 3 : 1 ratio.
 * const r = chiSquareGoodnessOfFit([55, 20, 16, 9], { probabilities: [9, 3, 3, 1] })
 * print('X2 =', r.statistic, ' p =', r.pValue)
 */
export function chiSquareGoodnessOfFit(observed: VectorLike, options: GoodnessOptions = {}): TestResult {
  return goodnessOfFit(observed, options, 1, 'chiSquareGoodnessOfFit')
}

/**
 * The $G$-test of goodness of fit: $G = 2\sum_i o_i \log(o_i/e_i)$, the log-likelihood ratio of the multinomial, with
 * the same $\chi^2(k - 1 - \text{ddof})$ null as Pearson's test (scipy's `power_divergence` with
 * `lambda_='log-likelihood'`). Effect size: Cohen's $w = \sqrt{G/N}$.
 *
 * @param observed The observed counts of the $k \ge 2$ categories.
 * @param options The null hypothesis's expected counts or probabilities (default: equal), and `ddof`.
 * @returns The test result.
 *
 * @example G is close to Pearson's X2 when the fit is good, and further when it is not
 * for (const counts of [[16, 18, 16, 14, 12, 12], [5, 5, 5, 5, 5, 35]]) {
 *   const g = gTestGoodnessOfFit(counts)
 *   print('G =', g.statistic, ' p =', g.pValue, ' X2 =', chiSquareGoodnessOfFit(counts).statistic)
 * }
 */
export function gTestGoodnessOfFit(observed: VectorLike, options: GoodnessOptions = {}): TestResult {
  return goodnessOfFit(observed, options, 0, 'gTestGoodnessOfFit')
}

/**
 * A contingency table with its expected counts under independence, $e_{ij} = r_i c_j / N$, and degrees of freedom.
 */
export type Table = {
  /** The number of rows $r$. */
  rows: number
  /** The number of columns $c$. */
  cols: number
  /** The observed counts, row-major ($r c$ values). */
  observed: Float64Array
  /** The expected counts under independence, row-major like `observed`. */
  expected: Float64Array
  /** The total count $N$. */
  total: number
  /** The degrees of freedom $(r - 1)(c - 1)$. */
  df: number
}

/**
 * The expected counts $r_i c_j/N$ of an $r \times c$ table under independence ($r_i$ and $c_j$ the row and column
 * totals), and its $(r - 1)(c - 1)$ degrees of freedom (scipy's `contingency.expected_freq`). Throws `DomainError`
 * for a table smaller than $2 \times 2$, a negative count, or a row or column that sums to zero.
 *
 * @param table The observed counts, rows by columns.
 * @returns The table with its expected counts.
 *
 * @example The counts a 2 x 3 table would have under independence
 * const t = expectedCounts([[10, 10, 20], [20, 20, 20]])
 * print('expected (row-major):', t.expected, ' df =', t.df)
 */
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

/**
 * The test of independence with statistic `powerDivergence` at `lambda` against $\chi^2((r - 1)(c - 1))$, upper tail,
 * with Yates' correction when asked for and the table has one degree of freedom.
 *
 * @param table The observed counts, rows by columns.
 * @param correction Whether to apply Yates' correction to a table with one degree of freedom.
 * @param lambda 1 for Pearson's $X^2$, 0 for $G$.
 * @param test The registry key of the test.
 * @returns The test result, with the uncorrected Cramér's $V$ as its effect size.
 */
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
 * Pearson's $\chi^2$ test of independence of the rows and columns of an $r \times c$ table:
 * $X^2 = \sum_{ij} (o_{ij} - e_{ij})^2/e_{ij}$ with $e_{ij} = r_i c_j/N$, $\chi^2((r - 1)(c - 1))$ under the null.
 * For a $2 \times 2$ table Yates' continuity correction is applied unless `correction` is false: each count moves
 * half a unit towards its expectation, never past it (as scipy's `chi2_contingency`). Effect size: Cramér's $V$
 * (uncorrected).
 *
 * @param table The observed counts, rows by columns, at least $2 \times 2$.
 * @param options Options of the test.
 * @param options.correction Apply Yates' correction to a $2 \times 2$ table (ignored for larger ones).
 * @returns The test result.
 *
 * @example A 2 x 3 table consistent with independence
 * const r = chiSquareIndependence([[10, 10, 20], [20, 20, 20]])
 * print('X2 =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 *
 * @example Yates' correction on a 2 x 2 table
 * const t = [[12, 5], [9, 17]]
 * const corrected = chiSquareIndependence(t)
 * const plain = chiSquareIndependence(t, { correction: false })
 * print('Yates-corrected: X2 =', corrected.statistic, ' p =', corrected.pValue)
 * print('uncorrected: X2 =', plain.statistic, ' p =', plain.pValue)
 * print(plain.effectSize)
 */
export function chiSquareIndependence(
  table: MatrixLike,
  { correction = true }: { correction?: boolean } = {},
): TestResult {
  return independence(table, correction, 1, 'chiSquareIndependence')
}

/**
 * The $G$-test of independence: $G = 2\sum_{ij} o_{ij} \log(o_{ij}/e_{ij})$, $\chi^2((r - 1)(c - 1))$ under the
 * null, with Yates-corrected counts for a $2 \times 2$ table unless `correction` is false (scipy's
 * `chi2_contingency` with `lambda_='log-likelihood'`). Effect size: Cramér's $V$ (from the uncorrected $X^2$).
 *
 * @param table The observed counts, rows by columns, at least $2 \times 2$.
 * @param options Options of the test.
 * @param options.correction Apply Yates' correction to a $2 \times 2$ table (ignored for larger ones).
 * @returns The test result.
 *
 * @example The G statistic of two tables
 * const big = gTestIndependence([[10, 10, 20], [20, 20, 20]])
 * print('2 x 3: G =', big.statistic, ' p =', big.pValue)
 * const small = gTestIndependence([[12, 5], [9, 17]])
 * print('2 x 2, corrected: G =', small.statistic, ' p =', small.pValue)
 */
export function gTestIndependence(table: MatrixLike, { correction = true }: { correction?: boolean } = {}): TestResult {
  return independence(table, correction, 0, 'gTestIndependence')
}

/**
 * Cramér's $V = \sqrt{X^2/(N(\min(r, c) - 1))}$ for an $r \times c$ table, from the uncorrected Pearson $X^2$: 0 for
 * independence, 1 for a perfect association (scipy's `contingency.association(method='cramer')`). The registered
 * metric is `aifn-compute/learning/metrics`' `cramersV`; this is the effect size the independence tests report.
 *
 * @param table The observed counts, rows by columns.
 * @returns $V$, in $[0, 1]$.
 */
function cramersV(table: MatrixLike): number {
  const t = expectedCounts(table)
  return Math.sqrt(powerDivergence(t.observed, t.expected, 1) / (t.total * (Math.min(t.rows, t.cols) - 1)))
}

// ── 2 × 2 tables ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The cells $a, b, c, d$ of a $2 \times 2$ table `[[a, b], [c, d]]`, checked to be non-negative integers (a
 * `DomainError` otherwise, or for a table of another shape).
 *
 * @param table The table, as two rows of two counts.
 * @param where The caller's name, for error messages.
 * @returns The cells `[a, b, c, d]`, row by row.
 */
function twoByTwo(table: MatrixLike, where: string): [number, number, number, number] {
  const { data, m, n } = dense.toMatrixF64(table, where)
  if (m !== 2 || n !== 2) throw new DomainError(where, `${where}: needs a 2 × 2 table`)
  for (const v of data)
    if (!(Number.isInteger(v) && v >= 0)) throw new DomainError(where, `${where}: cells must be non-negative integers`)
  return [data[0], data[1], data[2], data[3]]
}

/**
 * Fisher's noncentral hypergeometric law of $a$, the top-left cell given the margins, at odds ratio $\psi$:
 * $\pr(a = x) \propto \binom{K}{x}\binom{N - K}{n - x}\psi^x$ over $\max(0, n + K - N) \le x \le \min(n, K)$, with $N$
 * the total, $K$ the first column's total and $n$ the first row's. Returned as the support's lower end and the
 * probabilities, normalised in logs so that no term overflows.
 *
 * @param N The table's total count.
 * @param K The first column's total.
 * @param n The first row's total.
 * @param logPsi The log odds ratio $\log\psi$ (0 for the central hypergeometric law).
 * @returns `lo`, the smallest possible $a$, and `p`, the probabilities of $a = \text{lo}, \text{lo} + 1, \dots$.
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

/**
 * Solve $f(\log\psi) = \text{target}$ for an $f$ increasing in $\log\psi$, by bisection on $[-60, 60]$ ($\psi$ from
 * $e^{-60}$ to $e^{60}$), to $10^{-13}$ or 200 halvings. A target out of $f$'s range returns an end of the bracket.
 *
 * @param f An increasing function of the log odds ratio.
 * @param target The value to reach.
 * @returns The log odds ratio found.
 */
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
 * The odds ratio of a $2 \times 2$ table `[[a, b], [c, d]]` with a confidence interval. `sample`: $ad/(bc)$, with
 * Woolf's interval $\exp(\log \mathrm{OR} \pm z\sqrt{1/a + 1/b + 1/c + 1/d})$, or $[0, \infty]$ when a cell is 0;
 * `conditional` (default): the conditional maximum-likelihood estimate, the $\psi$ at which the noncentral
 * hypergeometric mean of $a$ equals the observed $a$ (0 or $\infty$ when $a$ is at an end of its range), with the
 * exact interval that inverts the two one-sided Fisher tests (Cornfield, 1956), as scipy's `odds_ratio`.
 *
 * @param table The table, two rows of two non-negative integer counts.
 * @param options The estimate and its interval.
 * @param options.kind `conditional` or `sample`.
 * @param options.level The confidence level, in $(0, 1)$.
 * @param options.alternative `two-sided`, `less` (an upper bound, from 0) or `greater` (a lower bound, to $\infty$).
 * @returns `value`, the odds ratio, and `ci`, its interval.
 *
 * @example The conditional and the sample odds ratio of one table
 * const table = [[7, 15], [58, 472]]
 * print('conditional:', oddsRatio(table))
 * print('sample:', oddsRatio(table, { kind: 'sample' }))
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
 * Fisher's exact test of a $2 \times 2$ table `[[a, b], [c, d]]` (Fisher, 1922): given the margins, $a$ follows the
 * hypergeometric law $\Hypergeom(N, a + c, a + b)$ when rows and columns are independent (odds ratio 1). The
 * statistic is $a$; `greater` tests an odds ratio above 1 (large $a$), and the two-sided p-value sums every table no
 * more likely than the observed one (scipy's `fisher_exact`). The estimate and effect size are the conditional
 * maximum-likelihood odds ratio with its exact interval.
 *
 * @param table The table, two rows of two non-negative integer counts.
 * @param options The alternative and the confidence level of the odds ratio's interval.
 * @returns The test result, with the exact hypergeometric null law.
 *
 * @example Fisher's tea-tasting lady names all 8 cups correctly
 * // Rows: milk or tea poured first; columns: what she guessed.
 * const r = fisherExact([[4, 0], [0, 4]])
 * print('two-sided p =', r.pValue)
 * print('one-sided p =', fisherExact([[4, 0], [0, 4]], { alternative: 'greater' }).pValue)
 *
 * @example A table that is not enough evidence
 * const r = fisherExact([[6, 2], [1, 4]])
 * print('p =', r.pValue, ' odds ratio =', r.estimate, ' interval:', r.ci)
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
