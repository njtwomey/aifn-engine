/**
 * Correlation and agreement: Pearson, Spearman and Kendall correlation as metrics, Lin's concordance correlation,
 * Fleiss' κ, Krippendorff's α (nominal, ordinal, interval, ratio, with missing ratings), intraclass correlations
 * (Shrout and Fleiss's six forms), and the nominal association measures of a contingency table (Cramér's V,
 * Tschuprow's T, Pearson's contingency coefficient, Theil's U).
 */

import { correlation, kendallTau, spearman } from 'aifn-compute/probability/stats'
import { expectedCounts, powerDivergence } from 'aifn-compute/probability/tests'
import { defineMetric, dense, divide, nonEmpty, sameLength, values, type Data, type Rows } from './core'
import { ShapeError } from 'aifn-compute/foundation/errors'

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

/** Pearson's correlation r between targets and predictions (pearson-correlation). */
export const pearsonCorrelation = defineMetric(
  agreementInfo('pearsonCorrelation', 'Pearson correlation', 'pearson-correlation'),
  (x: Data, y: Data): number => correlation(x, y),
)

/** Spearman's ρ, the Pearson correlation of mid-ranks (rank-correlation). */
export const spearmanCorrelation = defineMetric(
  agreementInfo('spearmanCorrelation', "Spearman's ρ", 'rank-correlation'),
  (x: Data, y: Data): number => spearman(x, y),
)

/** Kendall's τ_b, (C − D)/√((n₀ − Tₓ)(n₀ − T_y)) with tie corrections (rank-correlation). */
export const kendallCorrelation = defineMetric(
  agreementInfo('kendallCorrelation', "Kendall's τ_b", 'rank-correlation'),
  (x: Data, y: Data): number => kendallTau(x, y),
)

/**
 * Lin's concordance correlation coefficient 2σ_xy/(σ_x² + σ_y² + (μ_x − μ_y)²) with divisor-n moments (Lin 1989;
 * concordance-correlation-coefficient): 1 exactly when every pair satisfies y = x.
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
 * Fleiss' κ (Fleiss 1971; fleiss-kappa) from an N × k table whose entry nᵢⱼ counts the raters who put item i in
 * category j (every item rated by the same number n of raters): (P̄ − P̄ₑ)/(1 − P̄ₑ).
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

/** The scale of the ratings for Krippendorff's α: which differences count as disagreement. */
export type MeasurementLevel = 'nominal' | 'ordinal' | 'interval' | 'ratio'

/**
 * Krippendorff's α (Krippendorff 2011, "Computing Krippendorff's alpha-reliability") from a raters × units matrix of
 * ratings, NaN where a rater did not rate a unit: 1 − Dₒ/Dₑ, from the coincidence matrix of pairable values. Units
 * with fewer than two ratings are ignored. The level sets the difference function δ²: nominal (c ≠ k), ordinal (from
 * cumulative frequencies), interval (c − k)² or ratio ((c − k)/(c + k))².
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

/** The six intraclass correlations of Shrout and Fleiss (1979): single or average ratings, one- or two-way models. */
export type IccForm = 'ICC1' | 'ICC2' | 'ICC3' | 'ICC1k' | 'ICC2k' | 'ICC3k'

/**
 * An intraclass correlation from an n × k matrix of ratings (n targets, k raters, complete), by the two-way ANOVA
 * mean squares (Shrout and Fleiss 1979): ICC1 one-way random, ICC2 two-way random (absolute agreement), ICC3 two-way
 * mixed (consistency), and their k-rater averages. Default ICC2.
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
 * Pearson's χ² = Σ (nᵢⱼ − eᵢⱼ)²/eᵢⱼ of a contingency table, with eᵢⱼ = nᵢ₊n₊ⱼ/n, and the table's shape: the
 * expected counts and the power divergence of `aifn-compute/probability/tests` (one definition for the tests and the metrics).
 */
export function chiSquareStatistic(table: Rows): { chiSquare: number; n: number; rows: number; cols: number } {
  const t = expectedCounts(table)
  return { chiSquare: powerDivergence(t.observed, t.expected, 1), n: t.total, rows: t.rows, cols: t.cols }
}

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

/** Cramér's V = √(χ²/(n·min(r − 1, c − 1))) of an r × c contingency table (Cramér 1946). */
export const cramersV = defineMetric(associationInfo('cramersV', "Cramér's V"), (table: Rows): number => {
  const s = chiSquareStatistic(table)
  return Math.sqrt(s.chiSquare / (s.n * Math.min(s.rows - 1, s.cols - 1)))
})

/** Tschuprow's T = √(χ²/(n√((r − 1)(c − 1)))). */
export const tschuprowT = defineMetric(associationInfo('tschuprowT', "Tschuprow's T"), (table: Rows): number => {
  const s = chiSquareStatistic(table)
  return Math.sqrt(s.chiSquare / (s.n * Math.sqrt((s.rows - 1) * (s.cols - 1))))
})

/** Pearson's contingency coefficient C = √(χ²/(χ² + n)); its maximum is below 1. */
export const contingencyCoefficient = defineMetric(
  associationInfo('contingencyCoefficient', 'Contingency coefficient'),
  (table: Rows): number => {
    const s = chiSquareStatistic(table)
    return Math.sqrt(s.chiSquare / (s.chiSquare + s.n))
  },
)

/**
 * Theil's uncertainty coefficient U(Y | X) = I(X; Y)/H(Y) (Theil 1970): the fraction of the entropy of Y removed by
 * knowing X. With X the row variable, `of: 'columns'` (default) gives U(columns | rows) and `of: 'rows'` U(rows |
 * columns).
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
