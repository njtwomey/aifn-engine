/**
 * Curves of a scoring classifier over all thresholds, as the contract's `Curve` (`x`, `y`, `thresholds`, `area`,
 * narrowed by `curve`) that charts draw directly, and the areas and operating points read off them: ROC (with
 * tie-aware AUROC, partial AUROC, the convex hull and multiclass AUROC), precision–recall and average precision, DET,
 * cost curves, cumulative gain and lift, precision–recall–gain, equal error rate, Youden's $J$ and rates at a
 * constraint. Also the closed forms of the binormal model used by the site's figures.
 *
 * Cases with score $s_i \ge t$ are predicted positive; thresholds run from $+\infty$ (nothing positive) downwards,
 * one per distinct score, so tied scores move a curve in one step. The positive class is `positive` when given, else
 * as `positiveOf` chooses it from the true labels. A NaN score throws `DomainError`, and a rate with no cases to
 * divide by is NaN. The empirical curves are those of sklearn.metrics without dropping intermediate points.
 */

import { trapezoidSamples } from 'aifn-compute/numerics/quadrature'
import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'
import { ranks } from 'aifn-compute/probability/stats'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Curve } from 'aifn-compute/foundation/contracts'
import {
  binaryTruth,
  classesOf,
  defineMetric,
  dense,
  divide,
  encodeLabels,
  isMatrixLike,
  labelList,
  orderDescending,
  sameLength,
  values,
  vector,
  type Data,
  type Label,
  type Labels,
  type Rows,
} from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Cumulative counts at each distinct threshold, from the highest score down. */
type Sweep = {
  /** The distinct scores, in decreasing order. */
  thresholds: number[]
  /** True positives among cases with score $s_i \ge t$, for each threshold $t$. */
  tps: number[]
  /** False positives among cases with score $s_i \ge t$, for each threshold $t$. */
  fps: number[]
  /** The number of positive cases. */
  positives: number
  /** The number of negative cases. */
  negatives: number
}

/**
 * Walk down the cases in decreasing score order and record the counts at each distinct score. Tied scores move in one
 * step, so a tie between a positive and a negative moves the ROC curve diagonally. Inputs of different lengths throw
 * `ShapeError`, and a NaN score `DomainError`.
 *
 * @param yTrue The true labels.
 * @param scores A score per case.
 * @param positive The positive class; left out, chosen by `positiveOf` from the labels of `yTrue`.
 * @returns The distinct thresholds with the cumulative counts at each, and the totals of positives and negatives.
 */
function sweep(yTrue: Labels, scores: Data, positive?: Label): Sweep {
  const { y } = binaryTruth(yTrue, positive)
  const s = values(scores)
  sameLength(y, s, 'curve')
  for (let i = 0; i < s.length; i++)
    if (Number.isNaN(s[i])) throw new DomainError('metrics', 'metrics: curve: a score is NaN')
  const order = orderDescending(s)
  const thresholds: number[] = []
  const tps: number[] = []
  const fps: number[] = []
  let tp = 0
  let fp = 0
  for (let k = 0; k < order.length; k++) {
    const i = order[k]
    if (y[i]) tp++
    else fp++
    if (k === order.length - 1 || s[order[k + 1]] !== s[i]) {
      thresholds.push(s[i])
      tps.push(tp)
      fps.push(fp)
    }
  }
  return { thresholds, tps, fps, positives: tp, negatives: fp }
}

// ── ROC ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A ROC curve: `x` the false-positive rate and `y` the true-positive rate at each threshold, from $(0, 0)$ at
 * threshold $+\infty$ to $(1, 1)$; `area` is the trapezoidal AUROC, equal to the tie-aware Mann–Whitney statistic.
 */
export type RocCurve = Curve<'roc'> & {
  /** The threshold of each point: $+\infty$, then the distinct scores in decreasing order. */
  readonly thresholds: Tensor
  /** The area under the curve by the trapezoidal rule. */
  readonly area: number
  /** The number of positive cases. */
  readonly positives: number
  /** The number of negative cases. */
  readonly negatives: number
}

/**
 * The empirical ROC curve of binary labels and scores (Fawcett, 2006): $(\mathrm{FPR}, \mathrm{TPR})$ at every
 * distinct score as the threshold, plus $(0, 0)$ at $+\infty$. Tied scores give one diagonal step. The same points as
 * scikit-learn's `roc_curve` with `drop_intermediate=False`. With no negatives (or no positives) the rates are NaN.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The curve, with its thresholds, its area (the AUROC) and the counts of positives and negatives.
 *
 * @example The scikit-learn example
 * const c = rocCurve([1, 1, 2, 2], [0.1, 0.4, 0.35, 0.8], { positive: 2 })
 * print('FPR =', c.x)
 * print('TPR =', c.y)
 * print('thresholds =', c.thresholds)
 * print('AUROC =', c.area)
 */
export function rocCurve(yTrue: Labels, scores: Data, options: { positive?: Label } = {}): RocCurve {
  const sw = sweep(yTrue, scores, options.positive)
  const fpr = [0, ...sw.fps.map((v) => divide(v, sw.negatives))]
  const tpr = [0, ...sw.tps.map((v) => divide(v, sw.positives))]
  return {
    kind: 'curve',
    curve: 'roc',
    x: vector(fpr),
    y: vector(tpr),
    thresholds: vector([Infinity, ...sw.thresholds]),
    area: trapezoidSamples(tpr, fpr),
    positives: sw.positives,
    negatives: sw.negatives,
  }
}

/**
 * Binary AUROC by the Mann–Whitney statistic: $(\sum_{i : y_i = 1} r_i - n_+(n_+ + 1)/2) / (n_+ n_-)$, with $r_i$ the
 * average ranks, so a tied positive–negative pair counts one half (Mann and Whitney, 1947; Hanley and McNeil, 1982).
 *
 * @param y The truth as 0/1, one per case.
 * @param s The scores, one per case.
 * @returns The AUROC; NaN when either class is empty.
 */
function binaryAuroc(y: Uint8Array, s: Float64Array): number {
  const r = toFlat(ranks(s))
  let sum = 0
  let np = 0
  for (let i = 0; i < y.length; i++)
    if (y[i]) {
      sum += r[i]
      np++
    }
  const nn = y.length - np
  return divide(sum - (np * (np + 1)) / 2, np * nn)
}

/** Options of `auroc`. */
export type AurocOptions = {
  /** Binary: the positive class. */
  positive?: Label
  /**
   * Multiclass (scores as an $n \times K$ matrix): one-against-rest (default) or one-against-one (Hand and Till,
   * 2001).
   */
  multiClass?: 'ovr' | 'ovo'
  /** Multiclass: `macro` (default) or `weighted` by class prevalence (one-against-rest only). */
  average?: 'macro' | 'weighted'
  /** Multiclass: the classes in the order of the score columns; default the sorted labels. */
  labels?: readonly Label[]
}

/**
 * The area under the ROC curve, $\pr(s(X^+) > s(X^-)) + \frac{1}{2} \pr(s(X^+) = s(X^-))$
 * (receiver-operating-characteristic-curve-and-area), as scikit-learn's `roc_auc_score`. Binary scores are a vector;
 * for $K$ classes pass an $n \times K$ matrix of class scores (columns in label order) and choose one-against-rest or
 * the Hand–Till one-against-one average. A matrix whose column count is not the number of classes throws
 * `ShapeError`.
 *
 * @param yTrue The true labels.
 * @param scores A score per case (binary), or an $n \times K$ matrix with a column of scores per class.
 * @param options The positive class (binary), and the multiclass scheme, averaging and class order (`AurocOptions`).
 * @returns The AUROC: 1 when every positive outscores every negative, 0.5 for scores that ignore the class.
 *
 * @example Binary scores (the scikit-learn example)
 * print(auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]))
 *
 * @example A tie counts one half
 * print(auroc([0, 1], [0.5, 0.5]))
 *
 * @example Three classes, one against the rest and one against one
 * const yTrue = [0, 1, 2, 2]
 * const scores = [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6], [0.3, 0.4, 0.3]]
 * print('ovr =', auroc(yTrue, scores))
 * print('ovo =', auroc(yTrue, scores, { multiClass: 'ovo' }))
 */
export const auroc = defineMetric(
  {
    key: 'auroc',
    stability: 'stable',
    name: 'Area under the ROC curve',
    inputs: 'scores',
    direction: 'higher',
    range: [0, 1],
    notes: ['receiver-operating-characteristic-curve-and-area'],
    capability: 'score',
  },
  (yTrue: Labels, scores: Data | Rows, options: AurocOptions = {}): number => {
    if (!isMatrixLike(scores)) {
      const { y } = binaryTruth(yTrue, options.positive)
      const s = values(scores as Data)
      sameLength(y, s, 'auroc')
      return binaryAuroc(y, s)
    }
    const t = labelList(yTrue)
    const S = dense(scores as Rows, 'auroc scores')
    const classes = options.labels ? [...options.labels] : classesOf(t)
    if (classes.length !== S.cols)
      throw new ShapeError('metrics', `metrics: auroc: ${S.cols} score columns for ${classes.length} classes`)
    const ti = encodeLabels(t, classes)
    const column = (k: number, rows?: number[]) =>
      Float64Array.from(rows ?? Array.from({ length: S.rows }, (_, i) => i), (i) => S.data[i * S.cols + k])
    const K = classes.length
    if ((options.multiClass ?? 'ovr') === 'ovr') {
      let s = 0
      let w = 0
      for (let k = 0; k < K; k++) {
        const y = Uint8Array.from(ti, (v) => (v === k ? 1 : 0))
        const count = y.reduce((a, b) => a + b, 0)
        const wk = options.average === 'weighted' ? count : 1
        s += wk * binaryAuroc(y, column(k))
        w += wk
      }
      return s / w
    }
    // Hand and Till (2001): M = (2 / K(K − 1)) Σ_{j<k} (A(j|k) + A(k|j)) / 2, each A on the cases of the two classes.
    let s = 0
    for (let j = 0; j < K; j++)
      for (let k = j + 1; k < K; k++) {
        const rows = Array.from(ti.keys()).filter((i) => ti[i] === j || ti[i] === k)
        const yj = Uint8Array.from(rows, (i) => (ti[i] === j ? 1 : 0))
        const yk = Uint8Array.from(rows, (i) => (ti[i] === k ? 1 : 0))
        s += (binaryAuroc(yj, column(j, rows)) + binaryAuroc(yk, column(k, rows))) / 2
      }
    return (2 * s) / (K * (K - 1))
  },
)

/**
 * The area under the ROC curve for $\mathrm{FPR} \le m$, $m$ = `maxFpr`, with the curve interpolated linearly at
 * $m$. By default it is standardised by McClish's correction,
 * $\frac{1}{2}(1 + (A - A_{\min}) / (A_{\max} - A_{\min}))$ with $A_{\min} = m^2/2$ and $A_{\max} = m$, so that chance
 * scores 0.5 and perfect 1 (McClish, 1989), as scikit-learn's `roc_auc_score(max_fpr=…)`. A `maxFpr` outside
 * $(0, 1]$ throws `DomainError`.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `maxFpr`, the largest false-positive rate $m$ (required); `standardised`, false for the raw area
 *   $A$ (default true); and `positive`, the positive class.
 * @returns The standardised partial AUROC in $[0, 1]$, or the raw area in $[0, m]$.
 *
 * @example Standardised and raw, up to a false-positive rate of 0.5
 * const yTrue = [0, 0, 1, 1]
 * const scores = [0.1, 0.4, 0.35, 0.8]
 * print('standardised =', partialAuroc(yTrue, scores, { maxFpr: 0.5 }))
 * print('raw =', partialAuroc(yTrue, scores, { maxFpr: 0.5, standardised: false }))
 */
export const partialAuroc = defineMetric(
  {
    key: 'partialAuroc',
    stability: 'stable',
    name: 'Partial AUROC',
    inputs: 'scores',
    direction: 'higher',
    range: [0, 1],
    notes: ['receiver-operating-characteristic-curve-and-area'],
    capability: 'score',
  },
  (yTrue: Labels, scores: Data, options: { maxFpr: number; standardised?: boolean; positive?: Label }): number => {
    const { maxFpr } = options
    if (!(maxFpr > 0 && maxFpr <= 1))
      throw new DomainError('metrics', 'metrics: partialAuroc: maxFpr must lie in (0, 1]')
    const c = rocCurve(yTrue, scores, options)
    const fpr = c.x.data as Float64Array
    const tpr = c.y.data as Float64Array
    // The first point beyond maxFpr, and the curve cut there.
    let stop = fpr.findIndex((v) => v > maxFpr)
    if (stop < 0) stop = fpr.length
    const x = Array.from(fpr.subarray(0, stop))
    const y = Array.from(tpr.subarray(0, stop))
    if (stop < fpr.length) {
      const t = (maxFpr - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
      x.push(maxFpr)
      y.push(tpr[stop - 1] + t * (tpr[stop] - tpr[stop - 1]))
    }
    const area = trapezoidSamples(y, x)
    if (options.standardised === false) return area
    const minArea = 0.5 * maxFpr * maxFpr
    return 0.5 * (1 + (area - minArea) / (maxFpr - minArea))
  },
)

/**
 * The upper convex hull of a ROC curve, from $(0, 0)$ to $(1, 1)$ (Provost and Fawcett, 2001): the operating points
 * that are optimal for some costs and class ratio, as a ROC `Curve` without thresholds; `area` is the hull's AUROC.
 *
 * @param curve A ROC curve, as `rocCurve` returns it; only its points are read.
 * @returns The hull's vertices in increasing FPR, and the area under it.
 *
 * @example The hull skips the concave point
 * const c = rocCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
 * const hull = rocConvexHull(c)
 * print('hull FPR =', hull.x, ' TPR =', hull.y)
 * print('AUROC =', c.area, ' hull =', hull.area)
 */
export function rocConvexHull(curve: RocCurve): Curve<'roc'> & { readonly area: number } {
  const fpr = curve.x.data as Float64Array
  const tpr = curve.y.data as Float64Array
  const pts = Array.from(fpr, (x, i): [number, number] => [x, tpr[i]])
  pts.push([0, 0], [1, 1])
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const hull: [number, number][] = []
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  for (const p of pts) {
    while (hull.length >= 2 && cross(hull[hull.length - 2], hull[hull.length - 1], p) >= 0) hull.pop()
    hull.push(p)
  }
  const x = hull.map((p) => p[0])
  const y = hull.map((p) => p[1])
  return { kind: 'curve', curve: 'roc', x: vector(x), y: vector(y), area: trapezoidSamples(y, x) }
}

// ── Precision–recall ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A precision–recall curve: `x` recall and `y` precision, from recall 0 (precision 1 by convention, threshold
 * $+\infty$) to recall 1; `area` is the average precision $\sum_n (R_n - R_{n-1}) P_n$ and `prevalence` the curve's
 * chance level.
 */
export type PrecisionRecallCurve = Curve<'pr'> & {
  /** The threshold of each point: $+\infty$, then the distinct scores in decreasing order. */
  readonly thresholds: Tensor
  /** The average precision; NaN when there are no positives. */
  readonly area: number
  /** The fraction of cases that are positive. */
  readonly prevalence: number
}

/**
 * The precision–recall curve of binary labels and scores (precision-recall-curve-and-average-precision): precision and
 * recall at every distinct score as the threshold, preceded by (recall 0, precision 1). The same points as
 * scikit-learn's `precision_recall_curve`, in the reverse order.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The curve, with its thresholds, its area (the average precision) and the prevalence.
 *
 * @example Four cases
 * const c = precisionRecallCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
 * print('recall =', c.x)
 * print('precision =', c.y)
 * print('thresholds =', c.thresholds)
 * print('AP =', c.area, ' prevalence =', c.prevalence)
 */
export function precisionRecallCurve(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label } = {},
): PrecisionRecallCurve {
  const sw = sweep(yTrue, scores, options.positive)
  const precision = [1, ...sw.tps.map((tp, i) => tp / (tp + sw.fps[i]))]
  const recall = [0, ...sw.tps.map((tp) => divide(tp, sw.positives))]
  let ap = 0
  for (let i = 1; i < recall.length; i++) ap += (recall[i] - recall[i - 1]) * precision[i]
  return {
    kind: 'curve',
    curve: 'pr',
    x: vector(recall),
    y: vector(precision),
    thresholds: vector([Infinity, ...sw.thresholds]),
    area: sw.positives > 0 ? ap : NaN,
    prevalence: sw.positives / (sw.positives + sw.negatives),
  }
}

/**
 * Average precision, $\mathrm{AP} = \sum_n (R_n - R_{n-1}) P_n$ over the thresholds in decreasing order (Boyd et al.,
 * 2013): the step-wise area under the precision–recall curve, without interpolation, as scikit-learn's
 * `average_precision_score`. For an $n \times K$ score matrix, the one-against-rest average (`macro` default, or
 * `weighted` by prevalence). NaN when a class has no positives.
 *
 * @param yTrue The true labels.
 * @param scores A score per case (binary), or an $n \times K$ matrix with a column of scores per class.
 * @param options `positive`, the positive class (binary); `average`, `macro` or `weighted` (matrix); and `labels`,
 *   the classes in the order of the columns (default the sorted labels of `yTrue`). A column count other than the
 *   number of classes, or a row count other than the number of cases, throws `ShapeError`.
 * @returns The average precision, in $[0, 1]$.
 *
 * @example Binary scores (the scikit-learn example)
 * print(averagePrecision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]))
 *
 * @example Perfect and reversed rankings
 * print('perfect =', averagePrecision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]))
 * print('reversed =', averagePrecision([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]))
 */
export const averagePrecision = defineMetric(
  {
    key: 'averagePrecision',
    stability: 'stable',
    name: 'Average precision',
    inputs: 'scores',
    direction: 'higher',
    range: [0, 1],
    notes: ['precision-recall-curve-and-average-precision'],
    capability: 'score',
  },
  (
    yTrue: Labels,
    scores: Data | Rows,
    options: { positive?: Label; average?: 'macro' | 'weighted'; labels?: readonly Label[] } = {},
  ): number => {
    if (!isMatrixLike(scores)) return precisionRecallCurve(yTrue, scores as Data, options).area
    const t = labelList(yTrue)
    const S = dense(scores as Rows, 'averagePrecision scores')
    sameLength(t, { length: S.rows }, 'averagePrecision')
    const classes = options.labels ? [...options.labels] : classesOf(t)
    if (classes.length !== S.cols)
      throw new ShapeError(
        'metrics',
        `metrics: averagePrecision: ${S.cols} score columns for ${classes.length} classes`,
      )
    let s = 0
    let w = 0
    for (let k = 0; k < classes.length; k++) {
      const y = t.map((v) => (v === classes[k] ? 1 : 0))
      const col = Float64Array.from({ length: S.rows }, (_, i) => S.data[i * S.cols + k])
      const wk = options.average === 'weighted' ? y.reduce((a: number, b) => a + b, 0) : 1
      s += wk * precisionRecallCurve(y, col).area
      w += wk
    }
    return s / w
  },
)

/**
 * The trapezoidal area under a precision–recall curve. Straight lines between precision–recall points are not
 * achievable and are optimistic (Davis and Goadrich, 2006), so `averagePrecision` is preferred; the two may differ
 * either way.
 *
 * @param curve A precision–recall curve, as `precisionRecallCurve` returns it.
 * @returns The area under its points by the trapezoidal rule.
 *
 * @example Trapezoidal area and average precision
 * const c = precisionRecallCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
 * print('trapezoid =', precisionRecallTrapezoid(c), ' AP =', c.area)
 */
export function precisionRecallTrapezoid(curve: PrecisionRecallCurve): number {
  return trapezoidSamples(curve.y, curve.x)
}

/**
 * The precision–recall–gain curve (Flach and Kull, 2015): precision gain $(P - \pi) / ((1 - \pi) P)$ against recall
 * gain $(R - \pi) / ((1 - \pi) R)$, with $P$ precision, $R$ recall and $\pi$ the prevalence, at every threshold with
 * at least one true positive: `x` recall gain, `y` precision gain. Gains below 0 (worse than always-positive) are
 * kept, so a chart can clip them.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The curve with the threshold of each point and the prevalence $\pi$.
 *
 * @example Gains with balanced classes
 * const c = precisionRecallGainCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
 * print('recall gain =', c.x)
 * print('precision gain =', c.y)
 * print('thresholds =', c.thresholds)
 */
export function precisionRecallGainCurve(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label } = {},
): Curve<'prg'> & { readonly thresholds: Tensor; readonly prevalence: number } {
  const sw = sweep(yTrue, scores, options.positive)
  const pi = sw.positives / (sw.positives + sw.negatives)
  const pg: number[] = []
  const rg: number[] = []
  const th: number[] = []
  sw.tps.forEach((tp, i) => {
    if (tp === 0) return
    const fn = sw.positives - tp
    pg.push(1 - (pi / (1 - pi)) * (sw.fps[i] / tp))
    rg.push(1 - (pi / (1 - pi)) * (fn / tp))
    th.push(sw.thresholds[i])
  })
  return {
    kind: 'curve',
    curve: 'prg',
    x: vector(rg),
    y: vector(pg),
    thresholds: vector(th),
    prevalence: pi,
  }
}

// ── DET, operating points ────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The detection error trade-off curve: `x` the false-positive rate against `y` the false-negative rate at every
 * threshold (the ROC curve with $\mathrm{FNR} = 1 - \mathrm{TPR}$). Charts usually draw both on a probit scale,
 * `normalQuantile(rate)` from `aifn-compute/numerics/special`.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The curve with the threshold of each point.
 *
 * @example False negatives fall as false positives rise
 * const c = detCurve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
 * print('FPR =', c.x)
 * print('FNR =', c.y)
 */
export function detCurve(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label } = {},
): Curve<'det'> & { readonly thresholds: Tensor } {
  const c = rocCurve(yTrue, scores, options)
  return {
    kind: 'curve',
    curve: 'det',
    x: c.x,
    y: vector(Array.from(c.y.data, (v) => 1 - v)),
    thresholds: c.thresholds,
  }
}

/**
 * An operating point on a ROC curve: its `threshold`, and the false-positive rate `fpr`, true-positive rate `tpr` and
 * `precision` of predicting positive at scores at least that threshold.
 */
export type OperatingPoint = { threshold: number; fpr: number; tpr: number; precision: number }

/**
 * The operating point at one threshold of a sweep.
 *
 * @param sw The sweep.
 * @param i The index of the threshold in `sw.thresholds`.
 * @returns The threshold with its rates and precision.
 */
function pointAt(sw: Sweep, i: number): OperatingPoint {
  return {
    threshold: sw.thresholds[i],
    fpr: divide(sw.fps[i], sw.negatives),
    tpr: divide(sw.tps[i], sw.positives),
    precision: sw.tps[i] / (sw.tps[i] + sw.fps[i]),
  }
}

/**
 * The equal error rate (operating-points-and-equal-error-rate): the error rate where $\mathrm{FPR} = \mathrm{FNR}$ on
 * the empirical ROC curve, found by linear interpolation between the two adjacent points where
 * $\mathrm{FNR} - \mathrm{FPR}$ changes sign. The threshold is interpolated the same way (the second point's
 * threshold when the first is at $+\infty$). NaN when the rates are undefined.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The equal error `rate` and the `threshold` where it is reached.
 *
 * @example Both error rates are one third at threshold 0.6
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * print(equalErrorRate(yTrue, scores))
 */
export function equalErrorRate(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label } = {},
): { rate: number; threshold: number } {
  const c = rocCurve(yTrue, scores, options)
  const fpr = c.x.data as Float64Array
  const tpr = c.y.data as Float64Array
  const th = c.thresholds.data as Float64Array
  for (let i = 1; i < fpr.length; i++) {
    const d0 = 1 - tpr[i - 1] - fpr[i - 1]
    const d1 = 1 - tpr[i] - fpr[i]
    if (d0 >= 0 && d1 <= 0) {
      const t = d0 === d1 ? 0 : d0 / (d0 - d1)
      const threshold = Number.isFinite(th[i - 1]) ? th[i - 1] + t * (th[i] - th[i - 1]) : th[i]
      return { rate: fpr[i - 1] + t * (fpr[i] - fpr[i - 1]), threshold }
    }
  }
  return { rate: NaN, threshold: NaN }
}

/**
 * The equal error rate as a metric (lower is better): the `rate` of `equalErrorRate`.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The equal error rate, in $[0, 1]$.
 *
 * @example One third
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * print(eer(yTrue, scores))
 */
export const eer = defineMetric(
  {
    key: 'eer',
    stability: 'stable',
    name: 'Equal error rate',
    inputs: 'scores',
    direction: 'lower',
    range: [0, 1],
    notes: ['operating-points-and-equal-error-rate'],
    capability: 'score',
  },
  (yTrue: Labels, scores: Data, options: { positive?: Label } = {}): number =>
    equalErrorRate(yTrue, scores, options).rate,
)

/**
 * Youden's index: the threshold maximising $J = \mathrm{TPR} - \mathrm{FPR}$, and the point there (Youden, 1950).
 * The first maximum in decreasing threshold order is kept.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The operating point at the best threshold, with its $J$ as `j`.
 *
 * @example Two of three positives above every negative
 * print(youdenPoint([0, 0, 0, 1, 1, 1], [0.1, 0.3, 0.6, 0.2, 0.7, 0.9]))
 */
export function youdenPoint(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label } = {},
): OperatingPoint & { j: number } {
  const sw = sweep(yTrue, scores, options.positive)
  const P = sw.positives
  const N = sw.negatives
  // Compare J PN = TP N - FP P, exact in integers, so that equal J are equal and the first maximum is kept (TP/P - FP/N
  // in floating point can differ in the last bit between equal values).
  if (!(P > 0 && N > 0)) return { ...pointAt(sw, 0), j: -Infinity }
  let best = 0
  let bestKey = -Infinity
  for (let i = 0; i < sw.thresholds.length; i++) {
    const key = sw.tps[i] * N - sw.fps[i] * P
    if (key > bestKey) {
      bestKey = key
      best = i
    }
  }
  const point = pointAt(sw, best)
  return { ...point, j: point.tpr - point.fpr }
}

/**
 * The operating point that meets a constraint on one rate and is best on the other
 * (operating-points-and-equal-error-rate): with `minTpr`, the highest threshold whose TPR is at least `minTpr` (best
 * FPR); with `maxFpr`, the lowest threshold whose FPR is at most `maxFpr` (best TPR); with `minPrecision`, the lowest
 * threshold whose precision is at least `minPrecision` (best recall). Undefined (NaN rates) when no threshold
 * qualifies. Only the distinct scores are candidate thresholds, not $+\infty$.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param constraint The rate to constrain: one of `minTpr`, `maxFpr` or `minPrecision`.
 * @param options `positive`, the positive class.
 * @returns The operating point, or one of NaN values when no threshold meets the constraint.
 *
 * @example Three constraints on one classifier
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * print('TPR at least 1:', operatingPoint(yTrue, scores, { minTpr: 1 }))
 * print('no false positives:', operatingPoint(yTrue, scores, { maxFpr: 0 }))
 * print('precision at least 0.75:', operatingPoint(yTrue, scores, { minPrecision: 0.75 }))
 */
export function operatingPoint(
  yTrue: Labels,
  scores: Data,
  constraint: { minTpr: number } | { maxFpr: number } | { minPrecision: number },
  options: { positive?: Label } = {},
): OperatingPoint {
  const sw = sweep(yTrue, scores, options.positive)
  const none = { threshold: NaN, fpr: NaN, tpr: NaN, precision: NaN }
  const pts = sw.thresholds.map((_, i) => pointAt(sw, i))
  if ('minTpr' in constraint) return pts.find((p) => p.tpr >= constraint.minTpr) ?? none
  if ('maxFpr' in constraint) {
    const ok = pts.filter((p) => p.fpr <= constraint.maxFpr)
    return ok.length ? ok[ok.length - 1] : none
  }
  const ok = pts.filter((p) => p.precision >= constraint.minPrecision)
  return ok.length ? ok.reduce((a, b) => (b.tpr >= a.tpr ? b : a)) : none
}

/**
 * Specificity ($1 - \mathrm{FPR}$) at the operating point with sensitivity at least `sensitivity`: the highest such
 * threshold, as `operatingPoint` with `minTpr`. NaN when none qualifies.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `sensitivity`, the smallest acceptable true-positive rate (required), and `positive`, the positive
 *   class.
 * @returns The specificity there, in $[0, 1]$.
 *
 * @example Specificity when every positive must be found
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * print(specificityAtSensitivity(yTrue, scores, { sensitivity: 1 }))
 */
export const specificityAtSensitivity = defineMetric(
  {
    key: 'specificityAtSensitivity',
    stability: 'stable',
    name: 'Specificity at a sensitivity',
    inputs: 'scores',
    direction: 'higher',
    range: [0, 1],
    notes: ['operating-points-and-equal-error-rate'],
    capability: 'score',
  },
  (yTrue: Labels, scores: Data, options: { sensitivity: number; positive?: Label }): number =>
    1 - operatingPoint(yTrue, scores, { minTpr: options.sensitivity }, options).fpr,
)

/**
 * True-positive rate at the operating point with FPR at most `fpr`: the lowest such threshold, as `operatingPoint`
 * with `maxFpr`. NaN when none qualifies.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `fpr`, the largest acceptable false-positive rate (required), and `positive`, the positive class.
 * @returns The true-positive rate there, in $[0, 1]$.
 *
 * @example Recall with no false alarms, and with one in three
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * print('FPR 0:', tprAtFpr(yTrue, scores, { fpr: 0 }))
 * print('FPR 1/3:', tprAtFpr(yTrue, scores, { fpr: 1 / 3 }))
 */
export const tprAtFpr = defineMetric(
  {
    key: 'tprAtFpr',
    stability: 'stable',
    name: 'TPR at a false-positive rate',
    inputs: 'scores',
    direction: 'higher',
    range: [0, 1],
    notes: ['operating-points-and-equal-error-rate'],
    capability: 'score',
  },
  (yTrue: Labels, scores: Data, options: { fpr: number; positive?: Label }): number =>
    operatingPoint(yTrue, scores, { maxFpr: options.fpr }, options).tpr,
)

/**
 * Recall at the operating point with precision at least `precision` (best recall among those), as `operatingPoint`
 * with `minPrecision`. NaN when none qualifies.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `precision`, the smallest acceptable precision (required), and `positive`, the positive class.
 * @returns The recall there, in $[0, 1]$.
 *
 * @example Recall at precision 0.75 and at 1
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * print('precision 0.75:', recallAtPrecision(yTrue, scores, { precision: 0.75 }))
 * print('precision 1:', recallAtPrecision(yTrue, scores, { precision: 1 }))
 */
export const recallAtPrecision = defineMetric(
  {
    key: 'recallAtPrecision',
    stability: 'stable',
    name: 'Recall at a precision',
    inputs: 'scores',
    direction: 'higher',
    range: [0, 1],
    notes: ['operating-points-and-equal-error-rate'],
    capability: 'score',
  },
  (yTrue: Labels, scores: Data, options: { precision: number; positive?: Label }): number =>
    operatingPoint(yTrue, scores, { minPrecision: options.precision }, options).tpr,
)

// ── Cost curves, gain and lift ───────────────────────────────────────────────────────────────────────────────────────

/**
 * Normalised expected cost
 * $\mathrm{NE}[C] = \mathrm{FNR} \cdot \mathrm{PC}(+) + \mathrm{FPR} \cdot (1 - \mathrm{PC}(+))$ of an operating
 * point at probability cost $\mathrm{PC}(+)$ (Drummond and Holte, 2006).
 *
 * @param fpr The operating point's false-positive rate.
 * @param fnr The operating point's false-negative rate.
 * @param probabilityCost The probability cost $\mathrm{PC}(+)$ in $[0, 1]$: the prevalence times the cost of a false
 *   negative, normalised by the expected cost of the two errors.
 * @returns The normalised expected cost.
 *
 * @example Equal costs and priors
 * print(normalisedExpectedCost(0.2, 0.3, 0.5))
 */
export function normalisedExpectedCost(fpr: number, fnr: number, probabilityCost: number): number {
  return fnr * probabilityCost + fpr * (1 - probabilityCost)
}

/**
 * The cost curve of a scoring classifier (cost-curves): each ROC point is a line $\mathrm{NE}[C](\mathrm{PC})$ from
 * $(0, \mathrm{FPR})$ to $(1, \mathrm{FNR})$, and the classifier's cost curve is their lower envelope, sampled at
 * `points` probability costs (default 101): `x` the probability cost, `y` the normalised expected cost, `area` the
 * expected cost over uniform $\mathrm{PC}$ (trapezoidal). `fpr` and `fnr` are the ROC points, one line each.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class, and `points`, the number of probability costs sampled evenly on
 *   $[0, 1]$ (at least 2).
 * @returns The lower envelope with its area, and the ROC points behind it.
 *
 * @example Five probability costs
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * const c = costCurve(yTrue, scores, { points: 5 })
 * print('PC =', c.x)
 * print('cost =', c.y)
 * print('area =', c.area)
 */
export function costCurve(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label; points?: number } = {},
): Curve<'cost'> & { readonly area: number; readonly fpr: Tensor; readonly fnr: Tensor } {
  const c = rocCurve(yTrue, scores, options)
  const fpr = c.x.data as Float64Array
  const fnr = Float64Array.from(c.y.data, (v) => 1 - v)
  const m = options.points ?? 101
  const pc = Float64Array.from({ length: m }, (_, i) => i / (m - 1))
  const cost = Float64Array.from(pc, (x) => {
    let best = Infinity
    for (let i = 0; i < fpr.length; i++) best = Math.min(best, normalisedExpectedCost(fpr[i], fnr[i], x))
    return best
  })
  return {
    kind: 'curve',
    curve: 'cost',
    x: vector(pc),
    y: vector(cost),
    fpr: c.x,
    fnr: vector(fnr),
    area: trapezoidSamples(cost, pc),
  }
}

/**
 * The cumulative gain and lift curves: targeting the top fraction $q$ of cases by score captures a fraction
 * $\mathrm{gain}(q)$ of the positives, and $\mathrm{lift}(q) = \mathrm{gain}(q) / q$. One point per distinct score,
 * preceded by $(0, 0)$ (lift undefined there): `x` the fraction targeted, `y` the gain.
 *
 * @param yTrue The true labels; the positive class is chosen by `positiveOf` from them unless `options.positive`
 *   is given.
 * @param scores A score per case, higher meaning more likely positive. A NaN score throws `DomainError`.
 * @param options `positive`, the positive class.
 * @returns The gain curve with the threshold and the lift of each point.
 *
 * @example Half the cases hold two thirds of the positives
 * const yTrue = [0, 0, 0, 1, 1, 1]
 * const scores = [0.1, 0.3, 0.6, 0.4, 0.7, 0.9]
 * const c = gainCurve(yTrue, scores)
 * print('targeted =', c.x)
 * print('gain =', c.y)
 * print('lift =', c.lift)
 */
export function gainCurve(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label } = {},
): Curve<'gain'> & { readonly thresholds: Tensor; readonly lift: Tensor } {
  const sw = sweep(yTrue, scores, options.positive)
  const n = sw.positives + sw.negatives
  const fraction = [0, ...sw.tps.map((tp, i) => (tp + sw.fps[i]) / n)]
  const gain = [0, ...sw.tps.map((tp) => divide(tp, sw.positives))]
  return {
    kind: 'curve',
    curve: 'gain',
    x: vector(fraction),
    y: vector(gain),
    lift: vector(gain.map((g, i) => divide(g, fraction[i]))),
    thresholds: vector([Infinity, ...sw.thresholds]),
  }
}

// ── The binormal model ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The binormal model of the site's classification figures: negative scores $\sim \Gauss(0, 1)$, positive scores
 * $\sim \Gauss(d, 1)$, prevalence $\pi$, predict positive when the score exceeds $t$. Closed-form rates at a
 * threshold: $\mathrm{TPR} = 1 - \Phi(t - d)$ and $\mathrm{FPR} = 1 - \Phi(t)$.
 *
 * @param threshold The threshold $t$.
 * @param separation The separation $d$ of the two class means, in standard deviations.
 * @returns The true- and false-positive rates at $t$.
 *
 * @example Halfway between the means
 * print(binormalRates(1, 2))
 */
export function binormalRates(threshold: number, separation: number): { tpr: number; fpr: number } {
  return { tpr: 1 - normalCdf(threshold - separation), fpr: 1 - normalCdf(threshold) }
}

/**
 * The binormal ROC and precision–recall curves on `points` thresholds (default 400), evenly spaced from
 * $\max(4, d + 4)$ down to $\min(-4, d - 4)$. Arguments are named, so the separation and the prevalence cannot be
 * swapped. Precision is $\pi \mathrm{TPR} / (\pi \mathrm{TPR} + (1 - \pi) \mathrm{FPR})$, and 1 where nothing is
 * predicted positive.
 *
 * @param options `separation`, the separation $d$ of the class means; `prevalence`, the fraction $\pi$ of positives;
 *   and `points`, the number of thresholds (at least 2).
 * @returns The thresholds in decreasing order with the TPR, FPR and precision at each.
 *
 * @example Five thresholds
 * const c = binormalCurves({ separation: 2, prevalence: 0.5, points: 5 })
 * print('thresholds =', c.thresholds)
 * print('TPR =', c.tpr)
 * print('FPR =', c.fpr)
 * print('precision =', c.precision)
 */
export function binormalCurves(options: { separation: number; prevalence: number; points?: number }): {
  thresholds: Tensor
  tpr: Tensor
  fpr: Tensor
  precision: Tensor
} {
  const { separation: d, prevalence: pi } = options
  const m = options.points ?? 400
  const lo = Math.min(-4, d - 4)
  const hi = Math.max(4, d + 4)
  const th = Float64Array.from({ length: m }, (_, i) => hi - ((hi - lo) * i) / (m - 1))
  const tpr = new Float64Array(m)
  const fpr = new Float64Array(m)
  const precision = new Float64Array(m)
  th.forEach((t, i) => {
    const r = binormalRates(t, d)
    tpr[i] = r.tpr
    fpr[i] = r.fpr
    const denominator = pi * r.tpr + (1 - pi) * r.fpr
    precision[i] = denominator > 0 ? (pi * r.tpr) / denominator : 1
  })
  return { thresholds: vector(th), tpr: vector(tpr), fpr: vector(fpr), precision: vector(precision) }
}

/**
 * Binormal AUROC, $\Phi(d / \sqrt{2})$: the difference of a positive and a negative score is $\Gauss(d, 2)$.
 *
 * @param separation The separation $d$ of the class means, in standard deviations.
 * @returns The AUROC of the binormal model.
 *
 * @example Chance, and growing separation
 * print('d = 0:', binormalAuroc(0))
 * print('d = 1:', binormalAuroc(1))
 * print('d = 2:', binormalAuroc(2))
 */
export function binormalAuroc(separation: number): number {
  return normalCdf(separation / Math.SQRT2)
}

/**
 * Binormal equal error rate, $\Phi(-d/2)$, reached at $t = d/2$.
 *
 * @param separation The separation $d$ of the class means, in standard deviations.
 * @returns The equal error rate of the binormal model.
 *
 * @example The two error rates meet at t = d/2
 * print('EER =', binormalEqualErrorRate(2))
 * const r = binormalRates(1, 2)
 * print('FPR =', r.fpr, ' FNR =', 1 - r.tpr)
 */
export function binormalEqualErrorRate(separation: number): number {
  return normalCdf(-separation / 2)
}

/**
 * Binormal average precision, the integral $\int P \, dR = \int P(t) \, \phi(t - d) \, dt$ with $P(t)$ the
 * precision at threshold $t$, by Simpson's rule on $t \in [d - 12, d + 12]$ (2,000 intervals); accurate to about
 * $10^{-9}$ for moderate $d$.
 *
 * @param options `separation`, the separation $d$ of the class means, and `prevalence`, the fraction $\pi$ of
 *   positives.
 * @returns The average precision of the binormal model.
 *
 * @example Chance is the prevalence
 * print('d = 0:', binormalAveragePrecision({ separation: 0, prevalence: 0.3 }))
 * print('d = 2:', binormalAveragePrecision({ separation: 2, prevalence: 0.3 }))
 */
export function binormalAveragePrecision(options: { separation: number; prevalence: number }): number {
  const { separation: d, prevalence: pi } = options
  const m = 2000
  const a = d - 12
  const h = 24 / m
  let s = 0
  for (let i = 0; i <= m; i++) {
    const t = a + i * h
    const { tpr, fpr } = binormalRates(t, d)
    const denominator = pi * tpr + (1 - pi) * fpr
    const prec = denominator > 0 ? (pi * tpr) / denominator : 1
    const f = prec * normalPdf(t - d)
    s += (i === 0 || i === m ? 1 : i % 2 ? 4 : 2) * f
  }
  return (s * h) / 3
}
