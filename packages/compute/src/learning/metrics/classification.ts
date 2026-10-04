/**
 * Threshold metrics of predicted labels: accuracy, balanced accuracy, precision, recall, F-β, specificity, predictive
 * values, Jaccard, MCC, Cohen's κ (unweighted and weighted), Hamming loss and exact match, likelihood ratios, and the
 * ordinal metrics. Binary, multiclass and multi-label inputs share one averaging rule (see `Average` in confusion.ts).
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import {
  binaryCounts,
  confusionMatrix,
  averaged,
  perClass,
  tallies,
  type AverageOptions,
  type CountStatistic,
} from './confusion'
import {
  caseWeights,
  defineMetric,
  dense,
  divide,
  isMatrixLike,
  labelList,
  nonEmpty,
  sameLength,
  vector,
  type Data,
  type Label,
  type Labels,
  type Rows,
} from './core'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** Labels, or for multi-label metrics an n × L matrix of 0/1 (rows are cases, columns labels). */
export type ClassificationInput = Labels | Rows

// ── Count statistics (TP, FP, FN, TN) → value ────────────────────────────────────────────────────────────────────────

const precisionStat: CountStatistic = (tp, fp, _fn, _tn, z) => divide(tp, tp + fp, z)
const recallStat: CountStatistic = (tp, _fp, fn, _tn, z) => divide(tp, tp + fn, z)
const specificityStat: CountStatistic = (_tp, fp, _fn, tn, z) => divide(tn, tn + fp, z)
const npvStat: CountStatistic = (_tp, _fp, fn, tn, z) => divide(tn, tn + fn, z)
const fprStat: CountStatistic = (_tp, fp, _fn, tn, z) => divide(fp, fp + tn, z)
const fnrStat: CountStatistic = (tp, _fp, fn, _tn, z) => divide(fn, fn + tp, z)
const jaccardStat: CountStatistic = (tp, fp, fn, _tn, z) => divide(tp, tp + fp + fn, z)
const fStat =
  (beta: number): CountStatistic =>
  (tp, fp, fn, _tn, z) => {
    // Fβ = (1 + β²) TP / ((1 + β²) TP + β² FN + FP): the count form, defined whenever any of TP, FP, FN is non-zero.
    const b2 = beta * beta
    return divide((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp, z)
  }
const informednessStat: CountStatistic = (tp, fp, fn, tn, z) =>
  recallStat(tp, fp, fn, tn, z) + specificityStat(tp, fp, fn, tn, z) - 1
const markednessStat: CountStatistic = (tp, fp, fn, tn, z) =>
  precisionStat(tp, fp, fn, tn, z) + npvStat(tp, fp, fn, tn, z) - 1

function averagedMetric(
  stat: CountStatistic,
  yTrue: ClassificationInput,
  yPred: ClassificationInput,
  o: AverageOptions,
) {
  return averaged(stat, tallies(yTrue, yPred, o), o)
}

// ── Accuracy ─────────────────────────────────────────────────────────────────────────────────────────────────────────

function accuracyOf(yTrue: ClassificationInput, yPred: ClassificationInput, sampleWeight?: Data): number {
  if (isMatrixLike(yTrue) || isMatrixLike(yPred)) {
    // Multi-label accuracy is subset accuracy: a case counts only when every label is right.
    const t = dense(yTrue as Rows, 'accuracy')
    const p = dense(yPred as Rows, 'accuracy')
    const w = caseWeights(sampleWeight, t.rows, 'accuracy')
    let hit = 0
    let total = 0
    for (let i = 0; i < t.rows; i++) {
      let same = true
      for (let l = 0; l < t.cols; l++)
        if ((t.data[i * t.cols + l] !== 0) !== (p.data[i * t.cols + l] !== 0)) same = false
      const wi = w ? w[i] : 1
      if (same) hit += wi
      total += wi
    }
    return divide(hit, total)
  }
  const t = labelList(yTrue as Labels)
  const p = labelList(yPred as Labels)
  sameLength(t, p, 'accuracy')
  nonEmpty(t.length, 'accuracy')
  const w = caseWeights(sampleWeight, t.length, 'accuracy')
  let hit = 0
  let total = 0
  for (let i = 0; i < t.length; i++) {
    const wi = w ? w[i] : 1
    if (t[i] === p[i]) hit += wi
    total += wi
  }
  return hit / total
}

/**
 * Accuracy: the fraction of cases whose predicted label equals the true label (accuracy-and-balanced-accuracy). For
 * multi-label rows it is subset accuracy (exact match).
 */
export const accuracy = defineMetric(
  {
    key: 'accuracy',
    stability: 'stable',
    name: 'Accuracy',
    inputs: 'labels',
    direction: 'higher',
    range: [0, 1],
    notes: ['accuracy-and-balanced-accuracy'],
    capability: 'decide',
  },
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: { sampleWeight?: Data } = {}): number =>
    accuracyOf(yTrue, yPred, options.sampleWeight),
)

/** Error rate: 1 − accuracy. */
export const errorRate = defineMetric(
  {
    key: 'errorRate',
    stability: 'stable',
    name: 'Error rate',
    inputs: 'labels',
    direction: 'lower',
    range: [0, 1],
    notes: ['accuracy-and-balanced-accuracy'],
    capability: 'decide',
  },
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: { sampleWeight?: Data } = {}): number =>
    1 - accuracyOf(yTrue, yPred, options.sampleWeight),
)

/**
 * Balanced accuracy: the mean of the per-class recalls over the classes present in `yTrue`
 * (accuracy-and-balanced-accuracy), as scikit-learn's `balanced_accuracy_score`. With `adjusted`, rescaled so that
 * chance (1/K) scores 0 and perfect scores 1.
 */
export const balancedAccuracy = defineMetric(
  {
    key: 'balancedAccuracy',
    stability: 'stable',
    name: 'Balanced accuracy',
    inputs: 'labels',
    direction: 'higher',
    range: [0, 1],
    notes: ['accuracy-and-balanced-accuracy'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { adjusted?: boolean; sampleWeight?: Data } = {}): number => {
    const t = tallies(yTrue, yPred, { sampleWeight: options.sampleWeight })
    let s = 0
    let K = 0
    for (let k = 0; k < t.classes.length; k++) {
      if (t.support[k] === 0) continue
      s += t.tp[k] / t.support[k]
      K++
    }
    const score = s / K
    return options.adjusted ? (score - 1 / K) / (1 - 1 / K) : score
  },
)

// ── Precision, recall and friends ────────────────────────────────────────────────────────────────────────────────────

const averagedInfo = (key: string, name: string, note: string) =>
  ({
    key,
    name,
    stability: 'stable',
    inputs: 'labels',
    direction: 'higher',
    range: [0, 1],
    notes: [note],
    capability: 'decide',
  }) as const

/** Precision, TP / (TP + FP): the fraction of predicted positives that are positive (precision-recall-and-f-score). */
export const precision = defineMetric(
  averagedInfo('precision', 'Precision', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(precisionStat, yTrue, yPred, options),
)

/** Recall (sensitivity, true-positive rate), TP / (TP + FN) (precision-recall-and-f-score). */
export const recall = defineMetric(
  averagedInfo('recall', 'Recall', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(recallStat, yTrue, yPred, options),
)

/**
 * The F-β score, (1 + β²)·precision·recall / (β²·precision + recall), computed from counts as
 * (1 + β²)TP / ((1 + β²)TP + β²FN + FP) (van Rijsbergen 1979). β > 1 weights recall more. Default β = 1.
 */
export const fBeta = defineMetric(
  averagedInfo('fBeta', 'F-β score', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions & { beta?: number } = {}): number =>
    averagedMetric(fStat(options.beta ?? 1), yTrue, yPred, options),
)

/** The F₁ score, the harmonic mean of precision and recall: 2TP / (2TP + FP + FN). */
export const f1 = defineMetric(
  averagedInfo('f1', 'F₁ score', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(fStat(1), yTrue, yPred, options),
)

/** Specificity (true-negative rate), TN / (TN + FP) (sensitivity-specificity-and-predictive-values). */
export const specificity = defineMetric(
  averagedInfo('specificity', 'Specificity', 'sensitivity-specificity-and-predictive-values'),
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(specificityStat, yTrue, yPred, options),
)

/** Negative predictive value, TN / (TN + FN) (sensitivity-specificity-and-predictive-values). */
export const negativePredictiveValue = defineMetric(
  averagedInfo('negativePredictiveValue', 'Negative predictive value', 'sensitivity-specificity-and-predictive-values'),
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(npvStat, yTrue, yPred, options),
)

/** False-positive rate (fall-out), FP / (FP + TN) = 1 − specificity. */
export const falsePositiveRate = defineMetric(
  {
    ...averagedInfo('falsePositiveRate', 'False-positive rate', 'sensitivity-specificity-and-predictive-values'),
    direction: 'lower',
  },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(fprStat, yTrue, yPred, options),
)

/** False-negative rate (miss rate), FN / (FN + TP) = 1 − recall. */
export const falseNegativeRate = defineMetric(
  {
    ...averagedInfo('falseNegativeRate', 'False-negative rate', 'sensitivity-specificity-and-predictive-values'),
    direction: 'lower',
  },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(fnrStat, yTrue, yPred, options),
)

/** Informedness (Youden's J), TPR + TNR − 1 (Youden 1950). 0 for any classifier that ignores its input. */
export const informedness = defineMetric(
  { ...averagedInfo('informedness', 'Informedness', 'sensitivity-specificity-and-predictive-values'), range: [-1, 1] },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(informednessStat, yTrue, yPred, options),
)

/** Markedness, PPV + NPV − 1 (Powers 2011). */
export const markedness = defineMetric(
  { ...averagedInfo('markedness', 'Markedness', 'sensitivity-specificity-and-predictive-values'), range: [-1, 1] },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(markednessStat, yTrue, yPred, options),
)

/**
 * The Jaccard index TP / (TP + FP + FN) (hamming-jaccard-and-exact-match), averaged like precision; with multi-label
 * rows and `samples` averaging it is the mean per-case |Ŷ ∩ Y| / |Ŷ ∪ Y|.
 */
export const jaccardScore = defineMetric(
  averagedInfo('jaccardScore', 'Jaccard index', 'hamming-jaccard-and-exact-match'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(jaccardStat, yTrue, yPred, options),
)

/** Per-class precision, recall, F-β and support, aligned with `classes` (scikit-learn's `average=None`). */
export function precisionRecallFscoreSupport(
  yTrue: ClassificationInput,
  yPred: ClassificationInput,
  options: { beta?: number; labels?: readonly Label[]; zeroDivision?: number; sampleWeight?: Data } = {},
): { classes: Label[]; precision: Tensor; recall: Tensor; fScore: Tensor; support: Tensor } {
  const t = tallies(yTrue, yPred, options)
  const z = options.zeroDivision ?? NaN
  return {
    classes: t.classes,
    precision: perClass(precisionStat, t, z),
    recall: perClass(recallStat, t, z),
    fScore: perClass(fStat(options.beta ?? 1), t, z),
    support: vector(t.support),
  }
}

// ── Chance-corrected agreement ───────────────────────────────────────────────────────────────────────────────────────

/**
 * The Matthews correlation coefficient (Matthews 1975): for two classes (TP·TN − FP·FN)/√(product of the margins),
 * the Pearson correlation of the 0/1 truth and prediction; for K classes the Gorodkin form
 * (c·n − Σₖ pₖtₖ)/√((n² − Σpₖ²)(n² − Σtₖ²)) with c correct, pₖ predicted and tₖ true counts. NaN when a margin is
 * empty (libraries return 0).
 */
export const matthewsCorrelation = defineMetric(
  {
    key: 'matthewsCorrelation',
    stability: 'stable',
    name: 'Matthews correlation coefficient',
    inputs: 'labels',
    direction: 'higher',
    range: [-1, 1],
    notes: ['matthews-correlation-coefficient'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { labels?: readonly Label[]; sampleWeight?: Data } = {}): number => {
    const cm = confusionMatrix(yTrue, yPred, options)
    const K = cm.classes.length
    const c = cm.matrix.data as Float64Array
    let correct = 0
    let n = 0
    let sumPT = 0
    let sumP2 = 0
    let sumT2 = 0
    for (let k = 0; k < K; k++) {
      let row = 0
      let col = 0
      for (let j = 0; j < K; j++) {
        row += c[k * K + j]
        col += c[j * K + k]
      }
      correct += c[k * K + k]
      n += row
      sumPT += row * col
      sumP2 += col * col
      sumT2 += row * row
    }
    return divide(correct * n - sumPT, Math.sqrt((n * n - sumP2) * (n * n - sumT2)))
  },
)

/** Disagreement weights for Cohen's κ: none (0/1), linear |j − k|, quadratic (j − k)², or an explicit K × K matrix. */
export type KappaWeights = 'none' | 'linear' | 'quadratic' | Rows

/**
 * Weighted κ from a K × K agreement table C (rows: rater one, columns: rater two): 1 − Σ wⱼₖCⱼₖ / Σ wⱼₖEⱼₖ with
 * Eⱼₖ = rⱼcₖ/n the chance-expected counts (Cohen 1960; Cohen 1968). Unweighted κ is the 0/1 weight.
 */
export function kappaFromTable(table: Rows, weights: KappaWeights = 'none'): number {
  const { rows: K, cols, data: c } = dense(table, 'kappaFromTable')
  if (K !== cols) throw new ShapeError('metrics', 'metrics: kappaFromTable needs a square table')
  const w = new Float64Array(K * K)
  if (typeof weights === 'string') {
    for (let j = 0; j < K; j++)
      for (let k = 0; k < K; k++)
        w[j * K + k] = weights === 'none' ? (j === k ? 0 : 1) : weights === 'linear' ? Math.abs(j - k) : (j - k) ** 2
  } else {
    const m = dense(weights, 'kappa weights')
    if (m.rows !== K || m.cols !== K) throw new ShapeError('metrics', 'metrics: kappa weights must match the table')
    w.set(m.data)
  }
  const row = new Float64Array(K)
  const col = new Float64Array(K)
  let n = 0
  for (let j = 0; j < K; j++)
    for (let k = 0; k < K; k++) {
      row[j] += c[j * K + k]
      col[k] += c[j * K + k]
      n += c[j * K + k]
    }
  let observed = 0
  let expected = 0
  for (let j = 0; j < K; j++)
    for (let k = 0; k < K; k++) {
      observed += w[j * K + k] * c[j * K + k]
      expected += (w[j * K + k] * row[j] * col[k]) / n
    }
  return 1 - divide(observed, expected)
}

/**
 * Cohen's κ between two labellings (true labels and predictions, or two raters): (pₒ − pₑ)/(1 − pₑ), with pₑ from
 * each labelling's own marginal frequencies (cohens-kappa). `weights` gives weighted κ for ordinal classes, the order
 * being `labels` (default the sorted labels).
 */
export const cohensKappa = defineMetric(
  {
    key: 'cohensKappa',
    stability: 'stable',
    name: "Cohen's κ",
    inputs: 'labels',
    direction: 'higher',
    range: [-1, 1],
    notes: ['cohens-kappa'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { weights?: KappaWeights; labels?: readonly Label[] } = {}): number =>
    kappaFromTable(confusionMatrix(yTrue, yPred, { labels: options.labels }).matrix, options.weights ?? 'none'),
)

// ── Multi-label ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Hamming loss: the fraction of individual label decisions that are wrong, Σ|Ŷᵢ △ Yᵢ| / (nL) for multi-label rows
 * (hamming-jaccard-and-exact-match); for single labels, the fraction of wrong labels.
 */
export const hammingLoss = defineMetric(
  {
    key: 'hammingLoss',
    stability: 'stable',
    name: 'Hamming loss',
    inputs: 'sets',
    direction: 'lower',
    range: [0, 1],
    notes: ['hamming-jaccard-and-exact-match'],
    capability: 'decide',
  },
  (yTrue: ClassificationInput, yPred: ClassificationInput): number => {
    if (!(isMatrixLike(yTrue) || isMatrixLike(yPred))) return 1 - accuracyOf(yTrue, yPred)
    const t = tallies(yTrue, yPred)
    let wrong = 0
    for (let l = 0; l < t.classes.length; l++) wrong += t.fp[l] + t.fn[l]
    return wrong / (t.n * t.classes.length)
  },
)

/** Exact match (subset accuracy): the fraction of cases whose whole label set is right. */
export const exactMatch = defineMetric(
  {
    key: 'exactMatch',
    stability: 'stable',
    name: 'Exact match',
    inputs: 'sets',
    direction: 'higher',
    range: [0, 1],
    notes: ['hamming-jaccard-and-exact-match'],
    capability: 'decide',
  },
  (yTrue: ClassificationInput, yPred: ClassificationInput): number => accuracyOf(yTrue, yPred),
)

// ── Likelihood ratios ────────────────────────────────────────────────────────────────────────────────────────────────

/** LR⁺ = TPR / FPR, the factor by which a positive result multiplies the odds of the positive class. Binary. */
export const positiveLikelihoodRatio = defineMetric(
  {
    key: 'positiveLikelihoodRatio',
    stability: 'stable',
    name: 'Positive likelihood ratio',
    inputs: 'labels',
    direction: 'higher',
    range: [0, Infinity],
    notes: ['sensitivity-specificity-and-predictive-values'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { positive?: Label } = {}): number => {
    const { tp, fp, fn, tn } = binaryCounts(yTrue, yPred, options)
    return divide(divide(tp, tp + fn), divide(fp, fp + tn))
  },
)

/** LR⁻ = FNR / TNR, the factor by which a negative result multiplies the odds. Binary; lower is better. */
export const negativeLikelihoodRatio = defineMetric(
  {
    key: 'negativeLikelihoodRatio',
    stability: 'stable',
    name: 'Negative likelihood ratio',
    inputs: 'labels',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['sensitivity-specificity-and-predictive-values'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { positive?: Label } = {}): number => {
    const { tp, fp, fn, tn } = binaryCounts(yTrue, yPred, options)
    return divide(divide(fn, fn + tp), divide(tn, tn + fp))
  },
)

/** The diagnostic odds ratio LR⁺ / LR⁻ = (TP·TN)/(FP·FN). Binary. */
export const diagnosticOddsRatio = defineMetric(
  {
    key: 'diagnosticOddsRatio',
    stability: 'stable',
    name: 'Diagnostic odds ratio',
    inputs: 'labels',
    direction: 'higher',
    range: [0, Infinity],
    notes: ['sensitivity-specificity-and-predictive-values'],
    capability: 'decide',
  },
  (yTrue: Labels, yPred: Labels, options: { positive?: Label } = {}): number => {
    const { tp, fp, fn, tn } = binaryCounts(yTrue, yPred, options)
    return divide(tp * tn, fp * fn)
  },
)
