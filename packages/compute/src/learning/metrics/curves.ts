/**
 * Curves of a scoring classifier over all thresholds, as the contract's `Curve` (`x`, `y`, `thresholds`, `area`,
 * narrowed by `curve`) that charts draw directly, and the areas and
 * operating points read off them: ROC (with tie-aware AUROC, partial AUROC, the convex hull and multiclass AUROC),
 * precision–recall and average precision, DET, cost curves, cumulative gain and lift, precision–recall–gain, equal
 * error rate, Youden's J and rates at a constraint. Also the closed forms of the binormal model used by the site's
 * figures. Cases with score ≥ threshold are predicted positive; thresholds run from +∞ (nothing positive) downwards.
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
  thresholds: number[]
  /** True positives among cases with score ≥ threshold. */
  tps: number[]
  /** False positives among cases with score ≥ threshold. */
  fps: number[]
  positives: number
  negatives: number
}

/**
 * Walk down the cases in decreasing score order and record the counts at each distinct score. Tied scores move in one
 * step, so a tie between a positive and a negative moves the ROC curve diagonally.
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
 * A ROC curve: `x` the false-positive rate and `y` the true-positive rate at each threshold, from (0, 0) at threshold
 * +∞ to (1, 1); `area` is the trapezoidal AUROC, equal to the tie-aware Mann–Whitney statistic.
 */
export type RocCurve = Curve<'roc'> & {
  readonly thresholds: Tensor
  readonly area: number
  readonly positives: number
  readonly negatives: number
}

/**
 * The empirical ROC curve of binary labels and scores (Fawcett 2006): (FPR, TPR) at every distinct score as the
 * threshold, plus (0, 0) at +∞. Tied scores give one diagonal step. The same points as scikit-learn's `roc_curve`
 * with `drop_intermediate=False`.
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
 * Binary AUROC by the Mann–Whitney statistic: (Σ ranks of positives − n₊(n₊ + 1)/2)/(n₊n₋), with average ranks, so a
 * tied positive–negative pair counts one half (Mann and Whitney 1947; Hanley and McNeil 1982).
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
  /** Multiclass (scores as an n × K matrix): one-against-rest (default) or one-against-one (Hand and Till 2001). */
  multiClass?: 'ovr' | 'ovo'
  /** Multiclass: `macro` (default) or `weighted` by class prevalence (one-against-rest only). */
  average?: 'macro' | 'weighted'
  /** Multiclass: the classes in the order of the score columns; default the sorted labels. */
  labels?: readonly Label[]
}

/**
 * The area under the ROC curve, P(s(X⁺) > s(X⁻)) + ½ P(s(X⁺) = s(X⁻))
 * (receiver-operating-characteristic-curve-and-area). Binary scores are a vector; for K classes pass an n × K matrix
 * of class scores (columns in label order) and choose one-against-rest or the Hand–Till one-against-one average.
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
 * The area under the ROC curve for FPR ≤ `maxFpr`, with the curve interpolated linearly at `maxFpr`. By default it is
 * standardised by McClish's correction, ½(1 + (A − A_min)/(A_max − A_min)) with A_min = maxFpr²/2 and A_max = maxFpr,
 * so that chance scores 0.5 and perfect 1 (McClish 1989), as scikit-learn's `roc_auc_score(max_fpr=…)`.
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
 * The upper convex hull of a ROC curve, from (0, 0) to (1, 1) (Provost and Fawcett 2001): the operating points that
 * are optimal for some costs and class ratio, as a ROC `Curve` without thresholds; `area` is the hull's AUROC.
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
 * A precision–recall curve: `x` recall and `y` precision, from recall 0 (precision 1 by convention, threshold +∞) to
 * recall 1; `area` is the average precision Σ (Rₙ − Rₙ₋₁) Pₙ and `prevalence` the curve's chance level.
 */
export type PrecisionRecallCurve = Curve<'pr'> & {
  readonly thresholds: Tensor
  readonly area: number
  readonly prevalence: number
}

/**
 * The precision–recall curve of binary labels and scores (precision-recall-curve-and-average-precision): precision and
 * recall at every distinct score as the threshold, preceded by (recall 0, precision 1). The same points as
 * scikit-learn's `precision_recall_curve`, in the reverse order.
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
 * Average precision, AP = Σₙ (Rₙ − Rₙ₋₁) Pₙ over the thresholds in decreasing order (Boyd et al. 2013): the step-wise
 * area under the precision–recall curve, without interpolation, as scikit-learn's `average_precision_score`. For an
 * n × K score matrix, the one-against-rest average (`macro` default, or `weighted` by prevalence).
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
    const classes = options.labels ? [...options.labels] : classesOf(t)
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

/** The trapezoidal area under a precision–recall curve, which overestimates it (Davis and Goadrich 2006). */
export function precisionRecallTrapezoid(curve: PrecisionRecallCurve): number {
  return trapezoidSamples(curve.y, curve.x)
}

/**
 * The precision–recall–gain curve (Flach and Kull 2015): precision gain (prec − π)/((1 − π) prec) against recall gain
 * (rec − π)/((1 − π) rec), with π the prevalence, at every threshold with at least one true positive: `x` recall gain,
 * `y` precision gain. Gains below 0 (worse than always-positive) are kept, so a chart can clip them.
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
 * threshold (the ROC curve with FNR = 1 − TPR). Charts usually draw both on a probit scale, `normalQuantile(rate)` from
 * `aifn-compute/numerics/special`.
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

/** An operating point on a ROC curve. */
export type OperatingPoint = { threshold: number; fpr: number; tpr: number; precision: number }

function pointAt(sw: Sweep, i: number): OperatingPoint {
  return {
    threshold: sw.thresholds[i],
    fpr: divide(sw.fps[i], sw.negatives),
    tpr: divide(sw.tps[i], sw.positives),
    precision: sw.tps[i] / (sw.tps[i] + sw.fps[i]),
  }
}

/**
 * The equal error rate (operating-points-and-equal-error-rate): the error rate where FPR = FNR on the empirical ROC
 * curve, found by linear interpolation between the two adjacent points where FNR − FPR changes sign. The threshold is
 * interpolated the same way (+∞ at the first point).
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

/** The equal error rate as a metric (lower is better). */
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

/** Youden's index: the threshold maximising J = TPR − FPR, and the point there (Youden 1950). */
export function youdenPoint(
  yTrue: Labels,
  scores: Data,
  options: { positive?: Label } = {},
): OperatingPoint & { j: number } {
  const sw = sweep(yTrue, scores, options.positive)
  let best = 0
  let bestJ = -Infinity
  for (let i = 0; i < sw.thresholds.length; i++) {
    const p = pointAt(sw, i)
    if (p.tpr - p.fpr > bestJ) {
      bestJ = p.tpr - p.fpr
      best = i
    }
  }
  return { ...pointAt(sw, best), j: bestJ }
}

/**
 * The operating point that meets a constraint on one rate and is best on the other
 * (operating-points-and-equal-error-rate): with `minTpr`, the highest threshold whose TPR ≥ minTpr (best FPR); with
 * `maxFpr`, the lowest threshold whose FPR ≤ maxFpr (best TPR); with `minPrecision`, the lowest threshold whose
 * precision ≥ minPrecision (best recall). Undefined (NaN rates) when no threshold qualifies.
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

/** Specificity (1 − FPR) at the operating point with sensitivity at least `sensitivity`. */
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

/** True-positive rate at the operating point with FPR at most `fpr`. */
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

/** Recall at the operating point with precision at least `precision` (best recall among those). */
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
 * Normalised expected cost NE[C] = FNR·PC(+) + FPR·(1 − PC(+)) of an operating point at probability cost PC(+)
 * (Drummond and Holte 2006).
 */
export function normalisedExpectedCost(fpr: number, fnr: number, probabilityCost: number): number {
  return fnr * probabilityCost + fpr * (1 - probabilityCost)
}

/**
 * The cost curve of a scoring classifier (cost-curves): each ROC point is a line NE[C](PC) from (0, FPR) to (1, FNR),
 * and the classifier's cost curve is their lower envelope, sampled at `points` probability costs (default 101): `x` the
 * probability cost, `y` the normalised expected cost, `area` the expected cost over uniform PC. `fpr` and `fnr` are
 * the ROC points, one line each.
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
 * The cumulative gain and lift curves: targeting the top fraction q of cases by score captures a fraction gain(q) of
 * the positives, and lift(q) = gain(q)/q. One point per distinct score, preceded by (0, 0) (lift undefined there):
 * `x` the fraction targeted, `y` the gain.
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
 * The binormal model of the site's classification figures: negative scores ~ N(0, 1), positive scores ~ N(d, 1),
 * prevalence π, predict positive when the score exceeds t. Closed-form rates at a threshold.
 */
export function binormalRates(threshold: number, separation: number): { tpr: number; fpr: number } {
  return { tpr: 1 - normalCdf(threshold - separation), fpr: 1 - normalCdf(threshold) }
}

/**
 * The binormal ROC and precision–recall curves on `points` thresholds (default 400) from max(4, d + 4) down to
 * min(−4, d − 4). Arguments are named, so the separation and the prevalence cannot be swapped.
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

/** Binormal AUROC, Φ(d/√2): the difference of a positive and a negative score is N(d, 2). */
export function binormalAuroc(separation: number): number {
  return normalCdf(separation / Math.SQRT2)
}

/** Binormal equal error rate, Φ(−d/2), reached at t = d/2. */
export function binormalEqualErrorRate(separation: number): number {
  return normalCdf(-separation / 2)
}

/**
 * Binormal average precision, the integral ∫ precision d(recall) = ∫ prec(t) φ(t − d) dt, by Simpson's rule on
 * t ∈ [d − 12, d + 12] (2,000 intervals); accurate to about 1e-9 for moderate d.
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
