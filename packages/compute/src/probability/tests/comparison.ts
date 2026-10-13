/**
 * Tests for comparing learning algorithms: McNemar's test of two classifiers on one test set (McNemar, 1947;
 * statsmodels' `mcnemar`), the corrected resampled $t$-test (Nadeau and Bengio, 2003), the $5 \times 2$
 * cross-validated paired $t$-test (Dietterich, 1998) with the combined $F$-test (Alpaydin, 1999), Friedman's test of
 * $k$ algorithms over $N$ data sets (Friedman, 1937) with the Iman–Davenport $F$ correction (Iman and Davenport,
 * 1980) and the Nemenyi critical difference (Nemenyi, 1963; Demšar, 2006), and the Bayesian correlated $t$-test
 * (Corani and Benavoli, 2015; Benavoli et al., 2017).
 *
 * Differences of scores measured on resampled splits of one data set are correlated, since the training sets (and
 * test sets) overlap, so the naive variance $\hat\sigma^2/J$ of their mean is too small and a plain paired $t$-test
 * rejects too often. The corrected resampled test inflates it to $(1/J + n_2/n_1)\hat\sigma^2$, the $5 \times 2$
 * tests use five replications of two-fold cross-validation, whose two folds' training sets do not overlap, and the
 * Bayesian test models the correlation $\rho$ directly. Comparisons over several data sets use ranks instead.
 *
 * The Nemenyi critical difference needs the studentized range distribution, the law of
 * $Q = (\max_i X_i - \min_i X_i)/S$ for $k$ independent standard normals and an independent $S$ with $\nu S^2 \sim
 * \chi^2_\nu$; `studentizedRangeNull` evaluates its cdf and survival function by Gauss–Legendre quadrature, as
 * scipy's `studentized_range`.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import {
  dense,
  isTensor,
  tensor,
  toFlat,
  type MatrixLike,
  type Tensor,
  type VectorLike,
} from 'aifn-compute/foundation/tensor'
import { gaussLegendre } from 'aifn-compute/numerics/quadrature'
import { logGamma, normalCdf } from 'aifn-compute/numerics/special'
import { Binomial, ChiSquare, FisherSnedecor, StudentT, type Univariate } from 'aifn-compute/probability/distributions'
import { ranks } from 'aifn-compute/probability/stats'
import {
  checkLevel,
  continuousLaw,
  meanAndVariance,
  pivotInterval,
  pValueOf,
  result,
  sample,
  tailOf,
  type Alternative,
  type TestOptions,
  type TestResult,
} from './protocol'

// ── McNemar's test ───────────────────────────────────────────────────────────────────────────────────────────────────

/** The predicted or true labels of a test set: numbers (an array or a tensor), strings or booleans. */
export type Predictions = VectorLike | readonly (string | boolean)[]

/** Options of `mcnemarTest`. */
export type McnemarOptions = {
  /** The alternative: `greater` when classifier A is the more accurate. One-sided only for the exact test. */
  alternative?: Alternative
  /** Use the exact binomial test (default, as statsmodels) rather than the $\chi^2$ approximation. */
  exact?: boolean
  /** In the $\chi^2$ form, subtract 1 from $\lvert b - c \rvert$ (Edwards' continuity correction; default true). */
  correction?: boolean
}

/**
 * The result of McNemar's test: the protocol's fields and the $2 \times 2$ `table` of agreement with the truth, rows
 * for classifier A (correct, wrong) and columns for B (correct, wrong), so that its off-diagonal cells are the
 * discordant counts $b$ (only A correct) and $c$ (only B correct).
 */
export type McnemarTest = TestResult & { table: Tensor }

/**
 * The values of a set of labels, as an array that can be compared element by element.
 *
 * @param x The labels: an array of numbers, strings or booleans, or a tensor.
 * @returns The labels, in order.
 */
function labels(x: Predictions): readonly unknown[] {
  return isTensor(x) ? toFlat(x) : Array.from(x as ArrayLike<unknown>)
}

/**
 * McNemar's test (McNemar, 1947) of two classifiers A and B scored on the same test set. With $b$ the number of cases
 * only A classifies correctly and $c$ the number only B does, under $H_0$ (equal accuracies) each of the $b + c$
 * discordant cases is equally likely to go either way. The exact test (default) reads $b$ against
 * $\mathrm{Binomial}(b + c, \tfrac12)$, two-sided by doubling the smaller tail (statsmodels' `mcnemar(exact=True)`,
 * whose statistic is $\min(b, c)$; the p-values agree). With `exact: false` it is the $\chi^2$ form
 * $(\lvert b - c \rvert - 1)^2/(b + c)$ on one degree of freedom, or $(b - c)^2/(b + c)$ with `correction: false`,
 * NaN when $b + c = 0$. The estimate is $b/(b + c)$ (one half under the null) and the effect size the matched-pairs
 * odds ratio $b/c$.
 *
 * The data are either three arrays (A's predictions, B's predictions and the true labels) or, in place of them, the
 * $2 \times 2$ table of counts `[[both correct, only A correct], [only B correct, both wrong]]` (statsmodels' layout).
 * Throws `DomainError` for arrays of unequal lengths, a table that is not $2 \times 2$ with non-negative cells (whole
 * numbers for the exact test), or a one-sided alternative with the $\chi^2$ form.
 *
 * @param predA Classifier A's predicted labels.
 * @param predB Classifier B's predicted labels, as long as `predA`.
 * @param y The true labels, as long as `predA`.
 * @param options The alternative (`greater`: A is the more accurate), `exact` and `correction`.
 * @returns The test result, with the table of agreement.
 *
 * @example Two classifiers on twelve cases: A alone is right four times, B alone once
 * const y = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
 * const a = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0]
 * const b = [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]
 * const r = mcnemarTest(a, b, y)
 * print('table:', r.table)
 * print('exact: b =', r.statistic, ' p =', r.pValue)
 * print('chi-square: p =', mcnemarTest(a, b, y, { exact: false }).pValue)
 *
 * @example From the table of counts, as statsmodels takes it
 * const r = mcnemarTest([[101, 59], [121, 33]], { exact: false })
 * print('chi2 =', r.statistic, ' p =', r.pValue)
 * print('A more accurate? p =', mcnemarTest([[101, 59], [121, 33]], { alternative: 'greater' }).pValue)
 */
export function mcnemarTest(
  predA: Predictions,
  predB: Predictions,
  y: Predictions,
  options?: McnemarOptions,
): McnemarTest
export function mcnemarTest(table: MatrixLike, options?: McnemarOptions): McnemarTest
export function mcnemarTest(
  first: Predictions | MatrixLike,
  second?: Predictions | McnemarOptions,
  third?: Predictions,
  fourth?: McnemarOptions,
): McnemarTest {
  const where = 'mcnemarTest'
  let cells: [number, number, number, number]
  let options: McnemarOptions
  if (third === undefined) {
    options = (second as McnemarOptions | undefined) ?? {}
    const { data, m, n } = dense.toMatrixF64(first as MatrixLike, where)
    if (m !== 2 || n !== 2) throw new DomainError(where, `${where}: needs a 2 × 2 table`)
    for (const v of data)
      if (!(Number.isFinite(v) && v >= 0)) throw new DomainError(where, `${where}: cells must be non-negative`)
    cells = [data[0], data[1], data[2], data[3]]
  } else {
    options = fourth ?? {}
    const a = labels(first as Predictions)
    const b = labels(second as Predictions)
    const t = labels(third)
    if (a.length !== t.length || b.length !== t.length)
      throw new DomainError(where, `${where}: the predictions and labels must have equal lengths`)
    cells = [0, 0, 0, 0]
    for (let i = 0; i < t.length; i++) cells[(a[i] === t[i] ? 0 : 2) + (b[i] === t[i] ? 0 : 1)]++
  }
  const { exact = true, correction = true } = options
  const alternative = options.alternative ?? 'two-sided'
  const [, b, c] = cells
  const n = cells[0] + cells[1] + cells[2] + cells[3]
  const table = tensor([
    [cells[0], cells[1]],
    [cells[2], cells[3]],
  ])
  const common = {
    test: 'mcnemarTest',
    alternative,
    estimand: 'share of discordant cases A gets right',
    estimate: b / (b + c),
    nullValue: 0.5,
    effectSize: { name: 'matched-pairs odds ratio', value: b / c },
    n,
  }
  if (exact) {
    if (!(Number.isInteger(b) && Number.isInteger(c)))
      throw new DomainError(where, `${where}: the exact test needs whole-number counts`)
    const law = Binomial(b + c, 0.5)
    const tail = tailOf(alternative)
    const pValue = pValueOf(law, b, tail)
    const r = result({
      ...common,
      method: "McNemar's test (exact)",
      statistic: b,
      symbol: 'b',
      pValue,
      tail,
      null: law,
    })
    return { ...r, table }
  }
  if (alternative !== 'two-sided')
    throw new DomainError(where, `${where}: the chi-square form is two-sided; use the exact test for one side`)
  // Undefined (NaN) with no discordant cases, where the corrected form would otherwise give 1/0.
  const statistic = b + c > 0 ? (Math.abs(b - c) - (correction ? 1 : 0)) ** 2 / (b + c) : NaN
  const law = ChiSquare(1)
  const r = result({
    ...common,
    method: correction ? "McNemar's test (χ², continuity-corrected)" : "McNemar's test (χ²)",
    statistic,
    symbol: '\\chi^2',
    df: 1,
    pValue: Number.isNaN(statistic) ? NaN : pValueOf(law, statistic, 'upper'),
    tail: 'upper',
    null: law,
  })
  return { ...r, table }
}

// ── Resampled and cross-validated t-tests ────────────────────────────────────────────────────────────────────────────

/**
 * The corrected resampled $t$-test (Nadeau and Bengio, 2003) of $H_0$: two algorithms have equal expected scores,
 * from the differences $d_1, \dots, d_J$ of their scores on $J$ random train/test splits of one data set (repeated
 * hold-out or repeated cross-validation, with $n_1$ training and $n_2$ test cases per split). The overlap between
 * training sets correlates the $d_j$, so the variance of $\bar d$ is taken to be $(1/J + n_2/n_1)\hat\sigma^2$
 * rather than $\hat\sigma^2/J$, with $\hat\sigma^2$ the sample variance of the differences:
 *
 * $$t = \frac{\bar d}{\sqrt{(1/J + n_2/n_1)\,\hat\sigma^2}},$$
 *
 * read against Student's $t$ with $J - 1$ degrees of freedom. The interval is for the mean difference, with the same
 * corrected standard error. For $k$-fold cross-validation repeated $r$ times, $J = rk$ and
 * $n_2/n_1 = 1/(k - 1)$. Throws `DomainError` unless both sizes are positive.
 *
 * @param diffs The score differences $d_j$ (A minus B), one per split, at least two.
 * @param options The alternative (`greater`: A scores higher), the confidence level, and the split sizes.
 * @param options.trainSize The number of training cases $n_1$ in each split.
 * @param options.testSize The number of test cases $n_2$ in each split.
 * @returns The test result for the mean difference, with $J - 1$ degrees of freedom.
 *
 * @example Ten-fold cross-validation, ten differences of accuracy
 * const d = [0.021, 0.034, -0.005, 0.018, 0.027, 0.011, 0.009, 0.031, -0.002, 0.016]
 * const r = correctedResampledTTest(d, { trainSize: 900, testSize: 100 })
 * print('corrected: t =', r.statistic, ' p =', r.pValue)
 * print('naive paired t: p =', oneSampleTTest(d).pValue)
 * print('95% interval for the mean difference:', r.ci)
 */
export function correctedResampledTTest(
  diffs: VectorLike,
  options: TestOptions & { trainSize: number; testSize: number },
): TestResult {
  const where = 'correctedResampledTTest'
  const d = sample(diffs, where, 2)
  const { trainSize, testSize } = options
  if (!(trainSize > 0 && testSize > 0))
    throw new DomainError(where, `${where}: the training and test sizes must be positive`)
  const alternative = options.alternative ?? 'two-sided'
  const level = options.level ?? 0.95
  checkLevel(level, where)
  const J = d.length
  const { mean, variance } = meanAndVariance(d)
  const se = Math.sqrt((1 / J + testSize / trainSize) * variance)
  const statistic = mean / se
  const law = StudentT(J - 1)
  const tail = tailOf(alternative)
  return result({
    test: 'correctedResampledTTest',
    method: 'Corrected resampled t-test',
    statistic,
    symbol: 't',
    df: J - 1,
    pValue: Number.isNaN(statistic) ? NaN : pValueOf(law, statistic, tail),
    alternative,
    tail,
    null: law,
    estimand: 'mean difference',
    estimate: mean,
    nullValue: 0,
    ci: pivotInterval(mean, se, law, level, alternative),
    n: J,
  })
}

/**
 * The $5 \times 2$ differences as a row-major array, checked: five replications (rows) of two folds (columns), all
 * finite. Throws `DomainError` otherwise.
 *
 * @param diffs The differences, a $5 \times 2$ matrix.
 * @param where The caller's name, for error messages.
 * @returns The ten differences, row by row.
 */
function fiveByTwo(diffs: MatrixLike, where: string): Float64Array {
  const { data, m, n } = dense.toMatrixF64(diffs, where)
  if (m !== 5 || n !== 2) throw new DomainError(where, `${where}: needs a 5 × 2 matrix of differences`)
  for (const v of data) if (!Number.isFinite(v)) throw new DomainError(where, `${where}: a difference is not finite`)
  return data
}

/**
 * The sum over the five replications of the variance estimate $s_i^2 = (p_i^{(1)} - \bar p_i)^2 +
 * (p_i^{(2)} - \bar p_i)^2$, with $\bar p_i$ the mean of replication $i$'s two differences.
 *
 * @param p The ten differences, row by row.
 * @returns $\sum_{i=1}^{5} s_i^2$.
 */
function replicationVariance(p: Float64Array): number {
  let s = 0
  for (let i = 0; i < 5; i++) {
    const m = (p[2 * i] + p[2 * i + 1]) / 2
    s += (p[2 * i] - m) ** 2 + (p[2 * i + 1] - m) ** 2
  }
  return s
}

/**
 * Alpaydin's (1999) combined $5 \times 2$ cross-validated $F$-test: with $p_i^{(j)}$ the difference on fold $j$ of
 * replication $i$ and $s_i^2$ each replication's variance estimate (as in `fiveByTwoCvTest`),
 *
 * $$F = \frac{\sum_{i=1}^{5} \sum_{j=1}^{2} \left(p_i^{(j)}\right)^2}{2 \sum_{i=1}^{5} s_i^2},$$
 *
 * approximately $F(10, 5)$ under $H_0$, read from the upper tail. It uses all ten differences in the numerator, so it
 * is more stable than Dietterich's $t$, which uses one.
 *
 * @param diffs The differences (A minus B), a $5 \times 2$ matrix: one row per replication of two-fold
 *   cross-validation, one column per fold.
 * @returns The test result, with $F$ and its degrees of freedom $(10, 5)$.
 *
 * @example The combined F beside Dietterich's t on the same differences
 * const d = [[0.042, 0.018], [0.031, 0.025], [0.012, 0.039], [0.028, 0.016], [0.035, 0.021]]
 * const f = combinedFTest(d)
 * print('F =', f.statistic, ' df =', f.df, ' p =', f.pValue)
 * print('5x2cv t: p =', fiveByTwoCvTest(d).pValue)
 */
export function combinedFTest(diffs: MatrixLike): TestResult {
  const p = fiveByTwo(diffs, 'combinedFTest')
  let ss = 0
  for (const v of p) ss += v * v
  const statistic = ss / (2 * replicationVariance(p))
  const law = FisherSnedecor(10, 5)
  return result({
    test: 'combinedFTest',
    method: 'Combined 5×2cv F-test',
    statistic,
    symbol: 'F',
    df: [10, 5],
    pValue: Number.isNaN(statistic) ? NaN : pValueOf(law, statistic, 'upper'),
    alternative: 'two-sided',
    tail: 'upper',
    null: law,
    n: 10,
  })
}

/** The result of the $5 \times 2$ cross-validated $t$-test: the protocol's fields and the combined $F$-test. */
export type FiveByTwoCvTest = TestResult & { combinedF: TestResult }

/**
 * Dietterich's (1998) $5 \times 2$ cross-validated paired $t$-test: two-fold cross-validation is replicated five
 * times, giving differences $p_i^{(j)}$ (replication $i$, fold $j$). With $\bar p_i$ the mean of replication $i$ and
 * $s_i^2 = (p_i^{(1)} - \bar p_i)^2 + (p_i^{(2)} - \bar p_i)^2$,
 *
 * $$t = \frac{p_1^{(1)}}{\sqrt{\tfrac15 \sum_{i=1}^{5} s_i^2}},$$
 *
 * approximately Student's $t$ with 5 degrees of freedom under $H_0$. The numerator is the first fold's difference of
 * the first replication only, as Dietterich defines it. The result carries Alpaydin's combined $F$-test
 * (`combinedFTest`) as `combinedF`.
 *
 * @param diffs The differences (A minus B), a $5 \times 2$ matrix: one row per replication, one column per fold.
 * @param options The alternative (`greater`: A scores higher).
 * @param options.alternative The alternative hypothesis (default `two-sided`).
 * @returns The test result, with $t$ on 5 degrees of freedom and the combined $F$-test.
 *
 * @example Differences of accuracy from five replications of two-fold cross-validation
 * const d = [[0.042, 0.018], [0.031, 0.025], [0.012, 0.039], [0.028, 0.016], [0.035, 0.021]]
 * const r = fiveByTwoCvTest(d)
 * print('t =', r.statistic, ' df =', r.df, ' p =', r.pValue)
 * print('combined F =', r.combinedF.statistic, ' p =', r.combinedF.pValue)
 */
export function fiveByTwoCvTest(
  diffs: MatrixLike,
  { alternative = 'two-sided' }: { alternative?: Alternative } = {},
): FiveByTwoCvTest {
  const p = fiveByTwo(diffs, 'fiveByTwoCvTest')
  const statistic = p[0] / Math.sqrt(replicationVariance(p) / 5)
  const law = StudentT(5)
  const tail = tailOf(alternative)
  const r = result({
    test: 'fiveByTwoCvTest',
    method: '5×2cv paired t-test',
    statistic,
    symbol: 't',
    df: 5,
    pValue: Number.isNaN(statistic) ? NaN : pValueOf(law, statistic, tail),
    alternative,
    tail,
    null: law,
    estimand: 'difference on the first fold',
    estimate: p[0],
    nullValue: 0,
    n: 10,
  })
  return { ...r, combinedF: combinedFTest(diffs) }
}

// ── Friedman's test over several data sets ───────────────────────────────────────────────────────────────────────────

/**
 * The result of Friedman's test: the protocol's fields for the $\chi^2$ statistic, the `averageRanks` $R_j$ of the
 * $k$ algorithms (rank 1 the best, by `higherIsBetter`), and the Iman–Davenport $F$-test as `imanDavenport`.
 */
export type FriedmanTest = TestResult & { averageRanks: Tensor; imanDavenport: TestResult }

/**
 * Friedman's test (Friedman, 1937) of $H_0$: $k$ algorithms perform equally well, from their scores on $N$ data sets.
 * Each data set's scores are ranked (ties share the mean of their ranks), and with $R_j$ the average rank of
 * algorithm $j$,
 *
 * $$\chi^2_F = \frac{12N}{k(k + 1)}\left(\sum_{j=1}^{k} R_j^2 - \frac{k(k + 1)^2}{4}\right),$$
 *
 * divided by $1 - \sum (t^3 - t)/(Nk(k^2 - 1))$ over the groups of $t$ tied scores, and read against $\chi^2_{k-1}$
 * (scipy's `friedmanchisquare`). The result also carries the Iman–Davenport (1980) statistic
 * $F_F = (N - 1)\chi^2_F/(N(k - 1) - \chi^2_F)$, read against $F(k - 1, (k - 1)(N - 1))$, which Demšar (2006)
 * recommends as less conservative. After a rejection, two algorithms whose average ranks differ by at least
 * `nemenyiCriticalDifference(k, N)` are significantly different. Throws `DomainError` for fewer than two data sets or
 * algorithms, or a non-finite score.
 *
 * @param scores The scores, an $N \times k$ matrix: one row per data set, one column per algorithm.
 * @param options How scores are ranked.
 * @param options.higherIsBetter Rank the highest score 1 (default, for accuracies); false ranks the lowest 1 (for
 *   errors and losses). The statistic is the same either way; only the average ranks change.
 * @returns The test result, with the average ranks and the Iman–Davenport test.
 *
 * @example Three classifiers on six data sets
 * const acc = [
 *   [0.81, 0.79, 0.84],
 *   [0.92, 0.90, 0.93],
 *   [0.75, 0.77, 0.80],
 *   [0.68, 0.64, 0.71],
 *   [0.88, 0.85, 0.87],
 *   [0.71, 0.70, 0.76],
 * ]
 * const r = friedmanTest(acc)
 * print('average ranks:', r.averageRanks)
 * print('chi2_F =', r.statistic, ' p =', r.pValue)
 * print('Iman-Davenport F =', r.imanDavenport.statistic, ' p =', r.imanDavenport.pValue)
 * print('Nemenyi CD at 0.05:', nemenyiCriticalDifference(3, 6))
 */
export function friedmanTest(
  scores: MatrixLike,
  { higherIsBetter = true }: { higherIsBetter?: boolean } = {},
): FriedmanTest {
  const where = 'friedmanTest'
  const { data, m: N, n: k } = dense.toMatrixF64(scores, where)
  if (N < 2 || k < 2) throw new DomainError(where, `${where}: needs at least two data sets and two algorithms`)
  for (const v of data) if (!Number.isFinite(v)) throw new DomainError(where, `${where}: a score is not finite`)
  const total = new Float64Array(k)
  let ties = 0
  for (let i = 0; i < N; i++) {
    const row = Array.from(data.subarray(i * k, (i + 1) * k), (v) => (higherIsBetter ? -v : v))
    const r = toFlat(ranks(row))
    for (let j = 0; j < k; j++) total[j] += r[j]
    // Each group of t tied scores shares one midrank, so it adds t³ − t to the correction.
    const sorted = row.slice().sort((a, b) => a - b)
    for (let s = 0; s < k;) {
      let e = s
      while (e + 1 < k && sorted[e + 1] === sorted[s]) e++
      const t = e - s + 1
      ties += t ** 3 - t
      s = e + 1
    }
  }
  let ss = 0
  for (const t of total) ss += t * t
  const c = 1 - ties / (N * k * (k * k - 1))
  const statistic = ((12 / (N * k * (k + 1))) * ss - 3 * N * (k + 1)) / c
  const law = ChiSquare(k - 1)
  const fStatistic = ((N - 1) * statistic) / (N * (k - 1) - statistic)
  const fLaw = FisherSnedecor(k - 1, (k - 1) * (N - 1))
  // χ²_F reaches N(k − 1) when every data set ranks the algorithms alike, and then F_F is infinite.
  const imanDavenport = result({
    test: 'friedmanTest',
    method: 'Iman–Davenport F-test',
    statistic: fStatistic,
    symbol: 'F_F',
    df: [k - 1, (k - 1) * (N - 1)],
    pValue: Number.isNaN(fStatistic) ? NaN : pValueOf(fLaw, fStatistic, 'upper'),
    alternative: 'two-sided',
    tail: 'upper',
    null: fLaw,
    n: N,
  })
  const r = result({
    test: 'friedmanTest',
    method: "Friedman's test",
    statistic,
    symbol: '\\chi^2_F',
    df: k - 1,
    pValue: Number.isNaN(statistic) ? NaN : pValueOf(law, statistic, 'upper'),
    alternative: 'two-sided',
    tail: 'upper',
    null: law,
    n: N,
  })
  return { ...r, averageRanks: tensor(Array.from(total, (t) => t / N)), imanDavenport }
}

// ── The studentized range and the Nemenyi critical difference ────────────────────────────────────────────────────────

/** The points of the Gauss–Legendre rule on each panel of the quadratures below. */
const POINTS = 12
/** The panels of the outer integral over $\\log S$ of the studentized range. */
const OUTER_PANELS = 16

/**
 * A composite Gauss–Legendre rule: `panels` equal panels of $[a, b]$, each with the `POINTS`-point rule.
 *
 * @param a The left end.
 * @param b The right end, above `a`.
 * @param panels The number of panels.
 * @returns The nodes `x` and weights `w`, as Float64Arrays of `panels * POINTS` values.
 */
function compositeRule(a: number, b: number, panels: number): { x: Float64Array; w: Float64Array } {
  const { nodes, weights } = gaussLegendre(POINTS)
  const u = toFlat(nodes)
  const v = toFlat(weights)
  const h = (b - a) / panels
  const x = new Float64Array(panels * POINTS)
  const w = new Float64Array(panels * POINTS)
  for (let p = 0; p < panels; p++)
    for (let i = 0; i < POINTS; i++) {
      x[p * POINTS + i] = a + h * (p + (u[i] + 1) / 2)
      w[p * POINTS + i] = (h / 2) * v[i]
    }
  return { x, w }
}

/** The normal grid of the inner integral: nodes $z$ on $[-9, 9]$, weights times $\varphi(z)$, and $\Phi(\pm z)$. */
type NormalGrid = { z: Float64Array; w: Float64Array; below: Float64Array; above: Float64Array }
let grid: NormalGrid | null = null

/**
 * The quadrature grid for integrals against the standard normal density, built once: 36 panels of width $\tfrac12$
 * on $[-9, 9]$, outside which $\varphi(z) < 10^{-17}$.
 *
 * @returns The nodes, the weights times $\varphi(z)$, $\Phi(z)$ (`below`) and $1 - \Phi(z)$ (`above`).
 */
function normalGrid(): NormalGrid {
  if (grid) return grid
  const { x, w } = compositeRule(-9, 9, 36)
  const below = x.map((z) => normalCdf(z) as number)
  const above = x.map((z) => normalCdf(-z) as number)
  const weights = x.map((z, i) => (w[i] * Math.exp(-0.5 * z * z)) / Math.sqrt(2 * Math.PI))
  grid = { z: x, w: weights, below, above }
  return grid
}

/**
 * The cdf and survival function of the range of $k$ independent standard normals (the studentized range with
 * $\nu = \infty$) at $q$:
 *
 * $$\pr(R \le q) = k\int \varphi(z)\,\bigl(\Phi(z) - \Phi(z - q)\bigr)^{k-1}\,dz,$$
 *
 * and $\pr(R > q) = k\int \varphi(z)\,\Phi(z - q)\sum_{j=0}^{k-2} \Phi(z)^j \bigl(\Phi(z) - \Phi(z - q)\bigr)^{k-2-j}dz$,
 * the same difference of powers factored so that the upper tail keeps its relative accuracy.
 *
 * @param q The range, at least 0.
 * @param k The number of normals, at least 2.
 * @returns The cdf and the survival function at $q$.
 */
function normalRange(q: number, k: number): [number, number] {
  if (!(q > 0)) return [0, 1]
  if (q === Infinity) return [1, 0]
  const g = normalGrid()
  let cdf = 0
  let sf = 0
  for (let i = 0; i < g.z.length; i++) {
    const z = g.z[i]
    const a = g.below[i]
    const tail = normalCdf(z - q) as number
    // Φ(z) − Φ(z − q) from the upper tails when z > 0, where both cdfs are near 1.
    const d = Math.max(0, z > 0 ? (normalCdf(q - z) as number) - g.above[i] : a - tail)
    let s = 1
    let ap = 1
    for (let m = 1; m <= k - 2; m++) {
      ap *= a
      s = d * s + ap
    }
    cdf += g.w[i] * d ** (k - 1)
    sf += g.w[i] * tail * s
  }
  return [Math.min(1, k * cdf), Math.min(1, k * sf)]
}

/**
 * The cdf and survival function of the studentized range $Q = R/S$ with $\nu$ degrees of freedom:
 * $\pr(Q \le q) = \int_0^\infty g_\nu(s) \pr(R \le qs)\,ds$, with $g_\nu$ the density of $S = \sqrt{\chi^2_\nu/\nu}$,
 * $g_\nu(s) = \frac{2(\nu/2)^{\nu/2}}{\Gamma(\nu/2)} s^{\nu-1} e^{-\nu s^2/2}$, integrated by a composite
 * Gauss–Legendre rule between the $10^{-17}$ and $1 - 10^{-17}$ quantiles of $S$.
 *
 * @param q The statistic, at least 0.
 * @param k The number of means, at least 2.
 * @param df The degrees of freedom $\nu$, positive (`Infinity` for the range of normals).
 * @returns The cdf and the survival function at $q$.
 */
function studentizedRange(q: number, k: number, df: number): [number, number] {
  if (df === Infinity) return normalRange(q, k)
  if (!(q > 0)) return [0, 1]
  if (q === Infinity) return [1, 0]
  const chi = ChiSquare(df)
  const lo = Math.sqrt((chi.quantile(1e-17) as number) / df)
  const hi = Math.sqrt((chi.isf(1e-17) as number) / df)
  const { x, w } = compositeRule(lo, hi, OUTER_PANELS)
  const logNorm = Math.LN2 + (df / 2) * Math.log(df / 2) - (logGamma(df / 2) as number)
  let cdf = 0
  let sf = 0
  for (let i = 0; i < x.length; i++) {
    const s = x[i]
    const density = Math.exp(logNorm + (df - 1) * Math.log(s) - (df * s * s) / 2)
    const [c, t] = normalRange(q * s, k)
    cdf += w[i] * density * c
    sf += w[i] * density * t
  }
  return [Math.min(1, cdf), Math.min(1, sf)]
}

/**
 * The studentized range distribution (Student, 1927; Tukey's HSD): the law of $Q = (\max_i X_i - \min_i X_i)/S$ for
 * $k$ independent standard normals $X_i$ and an independent $S$ with $\nu S^2 \sim \chi^2_\nu$, on $[0, \infty)$.
 * Its cdf is the double integral
 *
 * $$\pr(Q \le q) = \int_0^\infty g_\nu(s)\, k\int \varphi(z)\bigl(\Phi(z) - \Phi(z - qs)\bigr)^{k-1}\,dz\,ds,$$
 *
 * with $g_\nu$ the density of $S$, evaluated by composite Gauss–Legendre quadrature (the inner integral on
 * $[-9, 9]$, the outer between the $10^{-17}$ and $1 - 10^{-17}$ quantiles of $S$), as scipy's
 * `studentized_range`; with $\nu = \infty$ (the default) $S = 1$ and only the inner integral remains. The survival
 * function is integrated directly, so upper-tail probabilities keep their relative accuracy. Quantiles invert the cdf
 * by bisection (fast for $\nu = \infty$, a second or so for finite $\nu$). Throws `DomainError` unless $k \ge 2$ is
 * an integer and $\nu > 0$.
 *
 * @param k The number of means compared, an integer at least 2.
 * @param df The degrees of freedom $\nu$ of the variance estimate; `Infinity` for a known variance.
 * @returns The law, a continuous `Univariate` on $[0, \infty)$ (its moments throw).
 *
 * @example The upper 5% points for k = 2 and k = 4, and a tail probability at 10 degrees of freedom
 * print('k = 2: q =', studentizedRangeNull(2).isf(0.05), ' (1.96 sqrt 2 =', 1.959964 * Math.SQRT2, ')')
 * print('k = 4: q =', studentizedRangeNull(4).isf(0.05))
 * print('P(Q > 4.33), k = 4, df = 10:', studentizedRangeNull(4, 10).survival(4.33))
 */
export function studentizedRangeNull(k: number, df = Infinity): Univariate {
  const where = 'studentizedRangeNull'
  if (!(Number.isInteger(k) && k >= 2)) throw new DomainError(where, `${where}: k must be an integer at least 2`)
  if (!(df > 0)) throw new DomainError(where, `${where}: the degrees of freedom must be positive`)
  // Both tails come from one quadrature, so the last point's pair is kept for a cdf and survival call at one point.
  let last = NaN
  let both: [number, number] = [NaN, NaN]
  const at = (q: number) => {
    if (q !== last) [last, both] = [q, studentizedRange(q, k, df)]
    return both
  }
  return continuousLaw({
    name: 'StudentizedRange',
    params: { k, df },
    lower: 0,
    upper: Infinity,
    cdf: (q) => at(q)[0],
    survival: (q) => at(q)[1],
  })
}

/**
 * The critical difference of the Nemenyi test (Nemenyi, 1963; Demšar, 2006) after Friedman's test: two of $k$
 * algorithms compared on $N$ data sets differ significantly at level $\alpha$ when their average ranks differ by at
 * least
 *
 * $$\mathrm{CD} = q_\alpha \sqrt{\frac{k(k + 1)}{6N}},$$
 *
 * where $q_\alpha$ is the upper $\alpha$ point of the studentized range of $k$ means with $\nu = \infty$, divided by
 * $\sqrt 2$ (Demšar's Table 5a: 1.960, 2.343, 2.569, ... for $k = 2, 3, 4, \dots$ at $\alpha = 0.05$). Throws
 * `DomainError` for $k < 2$, $N < 1$ or $\alpha$ outside $(0, 1)$.
 *
 * @param k The number of algorithms, an integer at least 2.
 * @param N The number of data sets, at least 1.
 * @param alpha The significance level $\alpha$, in $(0, 1)$.
 * @returns The critical difference of average ranks.
 *
 * @example Demšar's q values and the critical difference for five algorithms on twenty data sets
 * print('q_0.05, k = 2 ... 5:', [2, 3, 4, 5].map((k) => studentizedRangeNull(k).isf(0.05) / Math.SQRT2))
 * print('CD (k = 5, N = 20, 0.05):', nemenyiCriticalDifference(5, 20))
 * print('CD at 0.10:', nemenyiCriticalDifference(5, 20, 0.1))
 */
export function nemenyiCriticalDifference(k: number, N: number, alpha = 0.05): number {
  const where = 'nemenyiCriticalDifference'
  if (!(Number.isInteger(k) && k >= 2)) throw new DomainError(where, `${where}: k must be an integer at least 2`)
  if (!(N >= 1)) throw new DomainError(where, `${where}: N must be at least 1`)
  if (!(alpha > 0 && alpha < 1)) throw new DomainError(where, `${where}: α must be in (0, 1)`)
  const q = studentizedRangeNull(k).isf(alpha) as number
  return (q / Math.SQRT2) * Math.sqrt((k * (k + 1)) / (6 * N))
}

// ── The Bayesian correlated t-test ───────────────────────────────────────────────────────────────────────────────────

/**
 * The posterior of the Bayesian correlated $t$-test over the mean difference $\mu$, and the posterior probabilities of
 * the three regions the region of practical equivalence (rope) splits the line into.
 */
export type BayesianCorrelatedTTest = {
  /** The posterior of $\mu$: Student's $t$ with `df` degrees of freedom, location `mean` and scale `scale`; null when
   * the scale is 0 (every difference equal), where the posterior is a point mass at `mean`. */
  readonly posterior: Univariate | null
  /** The posterior location, the mean difference $\bar d$. */
  readonly mean: number
  /** The posterior scale $\sqrt{(1/n + \rho/(1 - \rho))\hat\sigma^2}$. */
  readonly scale: number
  /** The posterior degrees of freedom, $n - 1$. */
  readonly df: number
  /** The rope $[l, u]$: differences in it are practically equivalent to none. */
  readonly rope: readonly [number, number]
  /** $\pr(\mu < l)$: B is practically better (the differences are A minus B). */
  readonly pLeft: number
  /** $\pr(l \le \mu \le u)$: the two are practically equivalent. */
  readonly pRope: number
  /** $\pr(\mu > u)$: A is practically better. */
  readonly pRight: number
  /** The number of differences $n$. */
  readonly n: number
}

/**
 * The Bayesian correlated $t$-test (Corani and Benavoli, 2015; Benavoli et al., 2017) of two algorithms from the $n$
 * differences of their scores on resampled splits of one data set, whose correlation $\rho$ comes from overlapping
 * training sets ($\rho = n_2/(n_1 + n_2)$, the test fraction, as Nadeau and Bengio suggest; $1/k$ for $k$-fold
 * cross-validation). Under a flat prior the posterior of the mean difference $\mu$ is Student's $t$ with $n - 1$
 * degrees of freedom, location $\bar d$ and scale
 *
 * $$\sqrt{\left(\frac1n + \frac{\rho}{1 - \rho}\right)\hat\sigma^2},$$
 *
 * $\hat\sigma^2$ the sample variance of the differences. The result reports that posterior and its probabilities left
 * of, inside and right of the rope $[l, u]$, as the `baycomp` package's `CorrelatedTTest`. When every difference is
 * equal the posterior is a point mass at $\bar d$. Throws `DomainError` unless $0 \le \rho < 1$ and $l \le u$.
 *
 * @param diffs The score differences (A minus B), one per split, at least two.
 * @param options The correlation and the rope.
 * @param options.rho The correlation $\rho$ between differences, in $[0, 1)$; the test fraction of each split.
 * @param options.rope The rope: a half-width $r \ge 0$ for $[-r, r]$, or its two ends $[l, u]$. Default 0, so that
 *   `pRope` is 0 and `pLeft`, `pRight` are the posterior probabilities of either sign.
 * @returns The posterior and the probabilities of the three regions.
 *
 * @example Ten-fold cross-validation with a rope of one point of accuracy
 * const d = [0.021, 0.034, -0.005, 0.018, 0.027, 0.011, 0.009, 0.031, -0.002, 0.016]
 * const r = bayesianCorrelatedTTest(d, { rho: 0.1, rope: 0.01 })
 * print('posterior: mean =', r.mean, ' scale =', r.scale, ' df =', r.df)
 * print('P(B better) =', r.pLeft, ' P(equivalent) =', r.pRope, ' P(A better) =', r.pRight)
 */
export function bayesianCorrelatedTTest(
  diffs: VectorLike,
  { rho, rope = 0 }: { rho: number; rope?: number | readonly [number, number] },
): BayesianCorrelatedTTest {
  const where = 'bayesianCorrelatedTTest'
  const d = sample(diffs, where, 2)
  if (!(rho >= 0 && rho < 1)) throw new DomainError(where, `${where}: ρ must be in [0, 1)`)
  const [lower, upper] = typeof rope === 'number' ? [-rope, rope] : rope
  if (!(lower <= upper)) throw new DomainError(where, `${where}: the rope must be an interval [l, u] with l ≤ u`)
  const n = d.length
  const { mean, variance } = meanAndVariance(d)
  const scale = Math.sqrt((1 / n + rho / (1 - rho)) * variance)
  const df = n - 1
  const common = { mean, scale, df, rope: [lower, upper] as const, n }
  if (scale === 0) {
    const side = mean < lower ? 0 : mean > upper ? 2 : 1
    return { ...common, posterior: null, pLeft: +(side === 0), pRope: +(side === 1), pRight: +(side === 2) }
  }
  const posterior = StudentT(df, mean, scale)
  const pLeft = posterior.cdf(lower) as number
  const pRight = posterior.survival(upper) as number
  return { ...common, posterior, pLeft, pRope: Math.max(0, 1 - pLeft - pRight), pRight }
}
