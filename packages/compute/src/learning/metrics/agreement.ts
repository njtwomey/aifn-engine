/**
 * Correlation and agreement: Pearson, Spearman and Kendall correlation as metrics, Lin's concordance correlation,
 * Fleiss' $\kappa$, Krippendorff's $\alpha$ (nominal, ordinal, interval, ratio, with missing ratings), intraclass
 * correlations (Shrout and Fleiss's six forms), and the nominal association measures of a contingency table (Cramér's
 * $V$, Tschuprow's $T$, Pearson's contingency coefficient, Theil's $U$).
 *
 * The correlations compare two equal-length sequences (targets and predictions, or two raters) and are those of
 * `aifn-compute/probability/stats`, registered as metrics. The agreement coefficients read a matrix of ratings or
 * counts, and the association measures a contingency table of counts, whose $\chi^2$ comes from
 * `aifn-compute/probability/tests`. All are higher-is-better.
 */

import { correlation, kendallTau, spearman } from 'aifn-compute/probability/stats'
import { expectedCounts, powerDivergence } from 'aifn-compute/probability/tests'
import { defineMetric, dense, divide, nonEmpty, sameLength, values, type Data, type Rows } from './core'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The registry metadata of a correlation metric: stable, read from `values`, higher is better, range $[-1, 1]$.
 *
 * @param key The metric's registry key (its export name).
 * @param name The metric's display name.
 * @param note The key of the note that explains it.
 * @returns The metadata, with its literal fields kept.
 */
const agreementInfo = (key: string, name: string, note: string) =>
  ({
    key,
    name,
    stability: 'stable',
    inputs: 'values',
    direction: 'higher',
    range: [-1, 1],
    notes: [note],
    capability: 'decide',
  }) as const

/**
 * Pearson's correlation $r$ between targets and predictions (pearson-correlation), as `correlation`. NaN when either
 * is constant; throws `ShapeError` when the lengths differ.
 *
 * @param x The first sequence (the targets).
 * @param y The second sequence (the predictions), of the same length.
 * @returns $r$, in $[-1, 1]$.
 *
 * @example A linear relation, and a monotone one
 * print('linear', pearsonCorrelation([1, 2, 3, 4], [3, 5, 7, 9]))
 * print('cubic', pearsonCorrelation([1, 2, 3, 4], [1, 8, 27, 64]))
 */
export const pearsonCorrelation = defineMetric(
  agreementInfo('pearsonCorrelation', 'Pearson correlation', 'pearson-correlation'),
  (x: Data, y: Data): number => correlation(x, y),
)

/**
 * Spearman's $\rho$, the Pearson correlation of mid-ranks (rank-correlation), as `scipy.stats.spearmanr`. NaN when
 * either sequence is constant.
 *
 * @param x The first sequence.
 * @param y The second sequence, of the same length.
 * @returns $\rho$, in $[-1, 1]$.
 *
 * @example Any monotone relation scores 1
 * print('cubic', spearmanCorrelation([1, 2, 3, 4], [1, 8, 27, 64]))
 * print('one swap', spearmanCorrelation([1, 2, 3, 4], [1, 3, 2, 4]))
 */
export const spearmanCorrelation = defineMetric(
  agreementInfo('spearmanCorrelation', "Spearman's ρ", 'rank-correlation'),
  (x: Data, y: Data): number => spearman(x, y),
)

/**
 * Kendall's $\tau_b$, $(C - D)/\sqrt{(n_0 - T_x)(n_0 - T_y)}$ with $C$ and $D$ the concordant and discordant pairs,
 * $n_0 = n(n - 1)/2$, and $T_x$, $T_y$ the pairs tied in each sequence (rank-correlation). As
 * `scipy.stats.kendalltau`; NaN when either sequence is constant.
 *
 * @param x The first sequence.
 * @param y The second sequence, of the same length.
 * @returns $\tau_b$, in $[-1, 1]$.
 *
 * @example One swapped pair of six
 * print('tau_b', kendallCorrelation([1, 2, 3, 4], [1, 3, 2, 4]))
 */
export const kendallCorrelation = defineMetric(
  agreementInfo('kendallCorrelation', "Kendall's τ_b", 'rank-correlation'),
  (x: Data, y: Data): number => kendallTau(x, y),
)

/**
 * Lin's concordance correlation coefficient $2\sigma_{xy}/(\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2)$ with
 * divisor-$n$ moments (Lin 1989; concordance-correlation-coefficient): 1 exactly when every pair satisfies $y = x$.
 * Unlike Pearson's $r$, it penalises an offset or a change of scale. Throws `ShapeError` when the lengths differ and
 * `DomainError` when they are empty; NaN when both sequences are the same constant.
 *
 * @param x The first sequence (the targets).
 * @param y The second sequence (the predictions), of the same length.
 * @returns The concordance correlation, in $[-1, 1]$.
 *
 * @example Perfectly correlated, but offset by 1
 * const x = [1, 2, 3, 4]
 * print('CCC', concordanceCorrelation(x, [2, 3, 4, 5]))
 * print('Pearson', pearsonCorrelation(x, [2, 3, 4, 5]))
 * print('on the line y = x', concordanceCorrelation(x, x))
 */
export const concordanceCorrelation = defineMetric(
  agreementInfo('concordanceCorrelation', 'Concordance correlation coefficient', 'concordance-correlation-coefficient'),
  (x: Data, y: Data): number => {
    const a = values(x)
    const b = values(y)
    sameLength(a, b, 'concordanceCorrelation')
    nonEmpty(a.length, 'concordanceCorrelation')
    const n = a.length
    const ma = a.reduce((s, v) => s + v, 0) / n
    const mb = b.reduce((s, v) => s + v, 0) / n
    let va = 0
    let vb = 0
    let cov = 0
    for (let i = 0; i < n; i++) {
      va += (a[i] - ma) ** 2
      vb += (b[i] - mb) ** 2
      cov += (a[i] - ma) * (b[i] - mb)
    }
    return divide((2 * cov) / n, va / n + vb / n + (ma - mb) ** 2)
  },
)

/**
 * Fleiss' $\kappa$ (Fleiss 1971; fleiss-kappa) from an $N \times k$ table whose entry $n_{ij}$ counts the raters who
 * put item $i$ in category $j$ (every item rated by the same number $n$ of raters):
 * $(\bar P - \bar P_e)/(1 - \bar P_e)$, with $\bar P$ the mean agreement within items and $\bar P_e$ that expected
 * from the category totals. As statsmodels' `fleiss_kappa`. Throws `ShapeError` when an item's counts do not sum to
 * the first item's $n$.
 *
 * @param counts The $N \times k$ table of counts, one row per item and one column per category.
 * @returns $\kappa$: 1 for full agreement, 0 for agreement at chance.
 *
 * @example Two raters agree on two of three items
 * print('kappa', fleissKappa([
 *   [2, 0],
 *   [0, 2],
 *   [1, 1],
 * ]))
 */
export const fleissKappa = defineMetric(
  {
    key: 'fleissKappa',
    stability: 'stable',
    name: "Fleiss' κ",
    inputs: 'ratings',
    direction: 'higher',
    range: [-1, 1],
    notes: ['fleiss-kappa'],
  },
  (counts: Rows): number => {
    const { rows: N, cols: k, data } = dense(counts, 'fleissKappa')
    let n = 0
    for (let j = 0; j < k; j++) n += data[j]
    const p = new Float64Array(k)
    let pBar = 0
    for (let i = 0; i < N; i++) {
      let s = 0
      let sq = 0
      for (let j = 0; j < k; j++) {
        const v = data[i * k + j]
        s += v
        sq += v * v
        p[j] += v
      }
      if (s !== n) throw new ShapeError('metrics', `metrics: fleissKappa: item ${i} has ${s} ratings, expected ${n}`)
      pBar += (sq - n) / (n * (n - 1))
    }
    pBar /= N
    let pe = 0
    for (let j = 0; j < k; j++) pe += (p[j] / (N * n)) ** 2
    return divide(pBar - pe, 1 - pe)
  },
)

/**
 * The scale of the ratings for Krippendorff's $\alpha$: which differences count as disagreement (see
 * `krippendorffAlpha`).
 */
export type MeasurementLevel = 'nominal' | 'ordinal' | 'interval' | 'ratio'

/**
 * Krippendorff's $\alpha$ (Krippendorff 2011, "Computing Krippendorff's alpha-reliability") from a raters by units
 * matrix of ratings, NaN where a rater did not rate a unit: $1 - D_o/D_e$, from the coincidence matrix of pairable
 * values. Units with fewer than two ratings are ignored. The level sets the difference function $\delta^2$: nominal
 * ($c \ne k$), ordinal (from cumulative frequencies), interval $(c - k)^2$ or ratio $((c - k)/(c + k))^2$. As the
 * `krippendorff` Python package.
 *
 * @param ratings The ratings, one row per rater and one column per unit, NaN for a missing rating.
 * @param options `level`, the scale of the ratings (default `'nominal'`).
 * @returns $\alpha$: 1 for perfect reliability, 0 for agreement at chance.
 *
 * @example Two raters who differ on one unit, and a third with gaps
 * print('nominal', krippendorffAlpha([
 *   [1, 2, 3, 3],
 *   [1, 2, 3, 4],
 * ]))
 * print('interval', krippendorffAlpha([
 *   [1, 2, 3, 3],
 *   [1, 2, 3, 4],
 * ], { level: 'interval' }))
 * print('with missing', krippendorffAlpha([
 *   [1, 2, 3, 3],
 *   [1, 2, 3, 4],
 *   [NaN, 2, 3, NaN],
 * ]))
 */
export const krippendorffAlpha = defineMetric(
  {
    key: 'krippendorffAlpha',
    stability: 'stable',
    name: "Krippendorff's α",
    inputs: 'ratings',
    direction: 'higher',
    range: [-1, 1],
    notes: ['fleiss-kappa'],
  },
  (ratings: Rows, options: { level?: MeasurementLevel } = {}): number => {
    const { rows: raters, cols: units, data } = dense(ratings, 'krippendorffAlpha')
    const level = options.level ?? 'nominal'
    const valueSet = new Set<number>()
    for (const v of data) if (!Number.isNaN(v)) valueSet.add(v)
    const vals = [...valueSet].sort((a, b) => a - b)
    const index = new Map(vals.map((v, i) => [v, i]))
    const V = vals.length
    // Coincidence matrix: each unit with m ≥ 2 values adds 1/(m − 1) for every ordered pair of values from different raters.
    const o = new Float64Array(V * V)
    for (let u = 0; u < units; u++) {
      const present: number[] = []
      for (let r = 0; r < raters; r++) {
        const v = data[r * units + u]
        if (!Number.isNaN(v)) present.push(index.get(v)!)
      }
      const m = present.length
      if (m < 2) continue
      for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) if (a !== b) o[present[a] * V + present[b]] += 1 / (m - 1)
    }
    const nc = new Float64Array(V)
    for (let c = 0; c < V; c++) for (let k = 0; k < V; k++) nc[c] += o[c * V + k]
    const n = nc.reduce((s, v) => s + v, 0)
    const delta = (c: number, k: number): number => {
      if (level === 'nominal') return c === k ? 0 : 1
      if (level === 'interval') return (vals[c] - vals[k]) ** 2
      if (level === 'ratio') return ((vals[c] - vals[k]) / (vals[c] + vals[k])) ** 2
      const [lo, hi] = c < k ? [c, k] : [k, c]
      let s = 0
      for (let g = lo; g <= hi; g++) s += nc[g]
      return (s - (nc[c] + nc[k]) / 2) ** 2
    }
    let observed = 0
    let expected = 0
    for (let c = 0; c < V; c++)
      for (let k = 0; k < V; k++) {
        if (c === k) continue
        const d = delta(c, k)
        observed += o[c * V + k] * d
        expected += nc[c] * nc[k] * d
      }
    return 1 - divide(observed / n, expected / (n * (n - 1)))
  },
)

/**
 * The six intraclass correlations of Shrout and Fleiss (1979): single (`ICC1`, `ICC2`, `ICC3`) or $k$-rater average
 * (`ICC1k`, `ICC2k`, `ICC3k`) ratings, under the one-way random, two-way random and two-way mixed models.
 */
export type IccForm = 'ICC1' | 'ICC2' | 'ICC3' | 'ICC1k' | 'ICC2k' | 'ICC3k'

/**
 * An intraclass correlation from an $n \times k$ matrix of ratings ($n$ targets, $k$ raters, complete), by the two-way
 * ANOVA mean squares (Shrout and Fleiss 1979): ICC1 one-way random, ICC2 two-way random (absolute agreement), ICC3
 * two-way mixed (consistency), and their $k$-rater averages. Default ICC2. As pingouin's `intraclass_corr`. Missing
 * ratings are not handled: a NaN makes the result NaN.
 *
 * @param ratings The $n \times k$ ratings, one row per target and one column per rater.
 * @param options `form`, which of the six ICCs (default `'ICC2'`).
 * @returns The intraclass correlation, at most 1.
 *
 * @example A rater who always scores one higher: consistent, but not in absolute agreement
 * const ratings = [
 *   [1, 2],
 *   [2, 3],
 *   [3, 4],
 * ]
 * print('ICC1', intraclassCorrelation(ratings, { form: 'ICC1' }))
 * print('ICC2', intraclassCorrelation(ratings))
 * print('ICC3', intraclassCorrelation(ratings, { form: 'ICC3' }))
 */
export const intraclassCorrelation = defineMetric(
  {
    key: 'intraclassCorrelation',
    stability: 'stable',
    name: 'Intraclass correlation',
    inputs: 'ratings',
    direction: 'higher',
    range: [-1, 1],
    notes: ['fleiss-kappa'],
  },
  (ratings: Rows, options: { form?: IccForm } = {}): number => {
    const { rows: n, cols: k, data } = dense(ratings, 'intraclassCorrelation')
    let grand = 0
    const rowMean = new Float64Array(n)
    const colMean = new Float64Array(k)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < k; j++) {
        const v = data[i * k + j]
        grand += v / (n * k)
        rowMean[i] += v / k
        colMean[j] += v / n
      }
    let ssTotal = 0
    for (const v of data) ssTotal += (v - grand) ** 2
    const ssRows = k * rowMean.reduce((s, m) => s + (m - grand) ** 2, 0)
    const ssCols = n * colMean.reduce((s, m) => s + (m - grand) ** 2, 0)
    const ssError = ssTotal - ssRows - ssCols
    const msr = ssRows / (n - 1)
    const msc = ssCols / (k - 1)
    const mse = ssError / ((n - 1) * (k - 1))
    const msw = (ssCols + ssError) / (n * (k - 1))
    switch (options.form ?? 'ICC2') {
      case 'ICC1':
        return (msr - msw) / (msr + (k - 1) * msw)
      case 'ICC2':
        return (msr - mse) / (msr + (k - 1) * mse + (k * (msc - mse)) / n)
      case 'ICC3':
        return (msr - mse) / (msr + (k - 1) * mse)
      case 'ICC1k':
        return (msr - msw) / msr
      case 'ICC2k':
        return (msr - mse) / (msr + (msc - mse) / n)
      case 'ICC3k':
        return (msr - mse) / msr
    }
  },
)

// ── Nominal association ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Pearson's $\chi^2 = \sum_{ij} (n_{ij} - e_{ij})^2/e_{ij}$ of a contingency table, with $e_{ij} = n_{i+} n_{+j}/n$,
 * and the table's shape: the expected counts and the power divergence of `aifn-compute/probability/tests` (one
 * definition for the tests and the metrics). No continuity correction. Throws `DomainError` for a table smaller than
 * $2 \times 2$, a negative count, or a row or column that sums to zero.
 *
 * @param table The $r \times c$ table of counts.
 * @returns `chiSquare`, the statistic; `n`, the total count; `rows` and `cols`, $r$ and $c$.
 *
 * @example A 2 x 2 table with a moderate association
 * print(chiSquareStatistic([
 *   [10, 5],
 *   [5, 10],
 * ]))
 */
export function chiSquareStatistic(table: Rows): { chiSquare: number; n: number; rows: number; cols: number } {
  const t = expectedCounts(table)
  return { chiSquare: powerDivergence(t.observed, t.expected, 1), n: t.total, rows: t.rows, cols: t.cols }
}

/**
 * The registry metadata of a nominal association measure: stable, read from `ratings` (a contingency table), higher
 * is better, range $[0, 1]$.
 *
 * @param key The metric's registry key (its export name).
 * @param name The metric's display name.
 * @returns The metadata, with its literal fields kept.
 */
const associationInfo = (key: string, name: string) =>
  ({
    key,
    name,
    stability: 'stable',
    inputs: 'ratings',
    direction: 'higher',
    range: [0, 1],
    notes: ['nominal-association'],
  }) as const

/**
 * Cramér's $V = \sqrt{\chi^2/(n \min(r - 1, c - 1))}$ of an $r \times c$ contingency table (Cramér 1946), as
 * `scipy.stats.contingency.association` with `method='cramer'`. Throws as `chiSquareStatistic`.
 *
 * @param table The $r \times c$ table of counts.
 * @returns $V$, in $[0, 1]$: 0 for independence, 1 for a perfect association.
 *
 * @example Independent, partly and fully associated
 * print('independent', cramersV([[5, 5], [5, 5]]))
 * print('partly', cramersV([[10, 5], [5, 10]]))
 * print('fully', cramersV([[10, 0], [0, 10]]))
 */
export const cramersV = defineMetric(associationInfo('cramersV', "Cramér's V"), (table: Rows): number => {
  const s = chiSquareStatistic(table)
  return Math.sqrt(s.chiSquare / (s.n * Math.min(s.rows - 1, s.cols - 1)))
})

/**
 * Tschuprow's $T = \sqrt{\chi^2/(n\sqrt{(r - 1)(c - 1)})}$, as `scipy.stats.contingency.association` with
 * `method='tschuprow'`. It equals Cramér's $V$ for a square table and is smaller otherwise. Throws as
 * `chiSquareStatistic`.
 *
 * @param table The $r \times c$ table of counts.
 * @returns $T$, in $[0, 1]$.
 *
 * @example Below Cramér's V for a 2 x 3 table
 * const table = [
 *   [10, 5, 5],
 *   [5, 10, 10],
 * ]
 * print('T', tschuprowT(table), 'V', cramersV(table))
 */
export const tschuprowT = defineMetric(associationInfo('tschuprowT', "Tschuprow's T"), (table: Rows): number => {
  const s = chiSquareStatistic(table)
  return Math.sqrt(s.chiSquare / (s.n * Math.sqrt((s.rows - 1) * (s.cols - 1))))
})

/**
 * Pearson's contingency coefficient $C = \sqrt{\chi^2/(\chi^2 + n)}$; its maximum is below 1 (for an $r \times r$
 * table, $\sqrt{(r - 1)/r}$). As `scipy.stats.contingency.association` with `method='pearson'`. Throws as
 * `chiSquareStatistic`.
 *
 * @param table The $r \times c$ table of counts.
 * @returns $C$, in $[0, 1)$.
 *
 * @example A perfect 2 x 2 association reaches only sqrt(1/2)
 * print('C', contingencyCoefficient([[10, 0], [0, 10]]), 'sqrt(1/2) =', Math.SQRT1_2)
 */
export const contingencyCoefficient = defineMetric(
  associationInfo('contingencyCoefficient', 'Contingency coefficient'),
  (table: Rows): number => {
    const s = chiSquareStatistic(table)
    return Math.sqrt(s.chiSquare / (s.chiSquare + s.n))
  },
)

/**
 * Theil's uncertainty coefficient $U(Y \mid X) = I(X; Y)/H(Y)$ (Theil 1970): the fraction of the entropy of $Y$ removed
 * by knowing $X$. With $X$ the row variable, `of: 'columns'` (default) gives $U(\text{columns} \mid \text{rows})$ and
 * `of: 'rows'` $U(\text{rows} \mid \text{columns})$. It is asymmetric. NaN when the predicted variable takes one
 * value.
 *
 * @param table The $r \times c$ table of counts.
 * @param options `of`, the variable whose entropy is explained: `'columns'` (default) or `'rows'`.
 * @returns $U$, in $[0, 1]$.
 *
 * @example The rows determine the columns, but not the reverse
 * const table = [
 *   [10, 0],
 *   [0, 5],
 *   [0, 5],
 * ]
 * print('U(columns | rows)', theilsU(table))
 * print('U(rows | columns)', theilsU(table, { of: 'rows' }))
 */
export const theilsU = defineMetric(
  associationInfo('theilsU', "Theil's U"),
  (table: Rows, options: { of?: 'rows' | 'columns' } = {}): number => {
    const { rows, cols, data } = dense(table, 'theilsU')
    const r = new Float64Array(rows)
    const c = new Float64Array(cols)
    let n = 0
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        r[i] += data[i * cols + j]
        c[j] += data[i * cols + j]
        n += data[i * cols + j]
      }
    let mi = 0
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        const v = data[i * cols + j]
        if (v > 0) mi += (v / n) * Math.log((n * v) / (r[i] * c[j]))
      }
    const h = (m: Float64Array) => m.reduce((s, v) => (v > 0 ? s - (v / n) * Math.log(v / n) : s), 0)
    return divide(mi, h((options.of ?? 'columns') === 'columns' ? c : r))
  },
)
