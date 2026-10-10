/**
 * Threshold metrics of predicted labels: accuracy, balanced accuracy, precision, recall, $F_\beta$, specificity,
 * predictive values, Jaccard, MCC, Cohen's $\kappa$ (unweighted and weighted), Hamming loss and exact match, and
 * likelihood ratios. Binary, multiclass and multi-label inputs share one averaging rule (see `Average` in
 * confusion.ts).
 *
 * Each metric is a function of the true labels and the predicted labels, as in sklearn.metrics. A metric built from
 * the binary counts $\mathrm{TP}$, $\mathrm{FP}$, $\mathrm{FN}$ and $\mathrm{TN}$ is computed per class, one class
 * against the rest, and combined by `AverageOptions.average`: `binary` (the positive class; the default for two
 * classes), `micro`, `macro` (the default otherwise), `weighted` or `samples`. A ratio whose denominator is 0 is NaN
 * unless `zeroDivision` is given, so an undefined score is visible rather than reported as 0.
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

/** Labels, or for multi-label metrics an $n \times L$ matrix of 0/1 (rows are cases, columns labels). */
export type ClassificationInput = Labels | Rows

// ── Count statistics (TP, FP, FN, TN) → value ────────────────────────────────────────────────────────────────────────

/**
 * Precision, $\mathrm{TP} / (\mathrm{TP} + \mathrm{FP})$.
 *
 * @param tp True positives.
 * @param fp False positives.
 * @param _fn False negatives (not read).
 * @param _tn True negatives (not read).
 * @param z The value returned when the denominator is 0.
 * @returns The statistic, or `z` when its denominator is 0.
 */
const precisionStat: CountStatistic = (tp, fp, _fn, _tn, z) => divide(tp, tp + fp, z)
/**
 * Recall, $\mathrm{TP} / (\mathrm{TP} + \mathrm{FN})$.
 *
 * @param tp True positives.
 * @param _fp False positives (not read).
 * @param fn False negatives.
 * @param _tn True negatives (not read).
 * @param z The value returned when the denominator is 0.
 * @returns The statistic, or `z` when its denominator is 0.
 */
const recallStat: CountStatistic = (tp, _fp, fn, _tn, z) => divide(tp, tp + fn, z)
/**
 * Specificity, $\mathrm{TN} / (\mathrm{TN} + \mathrm{FP})$.
 *
 * @param _tp True positives (not read).
 * @param fp False positives.
 * @param _fn False negatives (not read).
 * @param tn True negatives.
 * @param z The value returned when the denominator is 0.
 * @returns The statistic, or `z` when its denominator is 0.
 */
const specificityStat: CountStatistic = (_tp, fp, _fn, tn, z) => divide(tn, tn + fp, z)
/**
 * Negative predictive value, $\mathrm{TN} / (\mathrm{TN} + \mathrm{FN})$.
 *
 * @param _tp True positives (not read).
 * @param _fp False positives (not read).
 * @param fn False negatives.
 * @param tn True negatives.
 * @param z The value returned when the denominator is 0.
 * @returns The statistic, or `z` when its denominator is 0.
 */
const npvStat: CountStatistic = (_tp, _fp, fn, tn, z) => divide(tn, tn + fn, z)
/**
 * False-positive rate, $\mathrm{FP} / (\mathrm{FP} + \mathrm{TN})$.
 *
 * @param _tp True positives (not read).
 * @param fp False positives.
 * @param _fn False negatives (not read).
 * @param tn True negatives.
 * @param z The value returned when the denominator is 0.
 * @returns The statistic, or `z` when its denominator is 0.
 */
const fprStat: CountStatistic = (_tp, fp, _fn, tn, z) => divide(fp, fp + tn, z)
/**
 * False-negative rate, $\mathrm{FN} / (\mathrm{FN} + \mathrm{TP})$.
 *
 * @param tp True positives.
 * @param _fp False positives (not read).
 * @param fn False negatives.
 * @param _tn True negatives (not read).
 * @param z The value returned when the denominator is 0.
 * @returns The statistic, or `z` when its denominator is 0.
 */
const fnrStat: CountStatistic = (tp, _fp, fn, _tn, z) => divide(fn, fn + tp, z)
/**
 * Jaccard index, $\mathrm{TP} / (\mathrm{TP} + \mathrm{FP} + \mathrm{FN})$.
 *
 * @param tp True positives.
 * @param fp False positives.
 * @param fn False negatives.
 * @param _tn True negatives (not read).
 * @param z The value returned when the denominator is 0.
 * @returns The statistic, or `z` when its denominator is 0.
 */
const jaccardStat: CountStatistic = (tp, fp, fn, _tn, z) => divide(tp, tp + fp + fn, z)
/**
 * The $F_\beta$ count statistic,
 * $(1 + \beta^2)\mathrm{TP} / ((1 + \beta^2)\mathrm{TP} + \beta^2\mathrm{FN} + \mathrm{FP})$.
 *
 * @param beta The weight $\beta$ of recall against precision.
 * @returns The statistic for that $\beta$, `zero` when $\mathrm{TP}$, $\mathrm{FP}$ and $\mathrm{FN}$ are all 0.
 */
const fStat =
  (beta: number): CountStatistic =>
  (tp, fp, fn, _tn, z) => {
    // Fβ = (1 + β²) TP / ((1 + β²) TP + β² FN + FP): the count form, defined whenever any of TP, FP, FN is non-zero.
    const b2 = beta * beta
    return divide((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp, z)
  }
/**
 * Informedness, recall plus specificity minus 1.
 *
 * @param tp True positives.
 * @param fp False positives.
 * @param fn False negatives.
 * @param tn True negatives.
 * @param z The value of an undefined recall or specificity.
 * @returns The informedness.
 */
const informednessStat: CountStatistic = (tp, fp, fn, tn, z) =>
  recallStat(tp, fp, fn, tn, z) + specificityStat(tp, fp, fn, tn, z) - 1
/**
 * Markedness, precision plus negative predictive value minus 1.
 *
 * @param tp True positives.
 * @param fp False positives.
 * @param fn False negatives.
 * @param tn True negatives.
 * @param z The value of an undefined precision or negative predictive value.
 * @returns The markedness.
 */
const markednessStat: CountStatistic = (tp, fp, fn, tn, z) =>
  precisionStat(tp, fp, fn, tn, z) + npvStat(tp, fp, fn, tn, z) - 1

/**
 * A count statistic of labels and predictions, tallied and averaged.
 *
 * @param stat The count statistic.
 * @param yTrue The true labels, or multi-label rows.
 * @param yPred The predicted labels, or multi-label rows.
 * @param o The averaging options; `labels` and `sampleWeight` are used to tally, the rest to average.
 * @returns The averaged statistic.
 */
function averagedMetric(
  stat: CountStatistic,
  yTrue: ClassificationInput,
  yPred: ClassificationInput,
  o: AverageOptions,
) {
  return averaged(stat, tallies(yTrue, yPred, o), o)
}

// ── Accuracy ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Accuracy, or subset accuracy for multi-label rows (a case counts when every label is right; any non-zero entry is
 * 1). Single-label inputs of different lengths throw `ShapeError` and an empty one `DomainError`.
 *
 * @param yTrue The true labels, or multi-label rows.
 * @param yPred The predicted labels, or multi-label rows of the same shape (not checked).
 * @param sampleWeight A weight per case; left out, every case has weight 1.
 * @returns The weighted fraction of cases predicted right (NaN for empty multi-label input).
 */
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
 * Accuracy: the fraction of cases whose predicted label equals the true label (accuracy-and-balanced-accuracy), as
 * scikit-learn's `accuracy_score`. For multi-label rows it is subset accuracy (exact match).
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `sampleWeight`, a weight per case (default 1 each).
 * @returns The (weighted) fraction of cases predicted right, in $[0, 1]$.
 *
 * @example Two of four right
 * print(accuracy([0, 1, 2, 3], [0, 2, 1, 3]))
 *
 * @example Multi-label rows: both labels must be right
 * print(accuracy([[0, 1], [1, 1]], [[1, 1], [1, 1]]))
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

/**
 * Error rate: $1 -$ accuracy, the (weighted) fraction of cases predicted wrong.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `sampleWeight`, a weight per case (default 1 each).
 * @returns The error rate, in $[0, 1]$.
 *
 * @example One of four wrong
 * print(errorRate(['a', 'b', 'b', 'a'], ['a', 'b', 'a', 'a']))
 */
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
 * chance ($1/K$, $K$ the number of classes present in `yTrue`) scores 0 and perfect scores 1.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options `adjusted` rescales for chance as above; `sampleWeight` is a weight per case.
 * @returns The balanced accuracy, in $[0, 1]$ (adjusted: at most 1, and negative when worse than chance).
 *
 * @example Imbalanced classes (the scikit-learn example)
 * const yTrue = [0, 1, 0, 0, 1, 0]
 * const yPred = [0, 1, 0, 0, 0, 1]
 * print('balanced accuracy =', balancedAccuracy(yTrue, yPred))
 * print('adjusted =', balancedAccuracy(yTrue, yPred, { adjusted: true }))
 * print('accuracy =', accuracy(yTrue, yPred))
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

/**
 * The registry metadata shared by the averaged metrics: stable, of labels, higher is better, range $[0, 1]$.
 *
 * @param key The metric's key, its export name.
 * @param name The metric's display name.
 * @param note The slug of the note that defines it.
 * @returns The metric's spec, for `defineMetric`.
 */
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

/**
 * Precision, $\mathrm{TP} / (\mathrm{TP} + \mathrm{FP})$: the fraction of predicted positives that are positive
 * (precision-recall-and-f-score), as scikit-learn's `precision_score`.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged precision; NaN (or `zeroDivision`) when nothing is predicted positive.
 *
 * @example A binary problem
 * print(precision([0, 1, 1, 0, 1], [0, 1, 0, 1, 1]))
 *
 * @example Three classes, averaged three ways (the scikit-learn example)
 * const yTrue = [0, 1, 2, 0, 1, 2]
 * const yPred = [0, 2, 1, 0, 0, 1]
 * print('macro =', precision(yTrue, yPred, { average: 'macro' }))
 * print('micro =', precision(yTrue, yPred, { average: 'micro' }))
 * print('weighted =', precision(yTrue, yPred, { average: 'weighted' }))
 */
export const precision = defineMetric(
  averagedInfo('precision', 'Precision', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(precisionStat, yTrue, yPred, options),
)

/**
 * Recall (sensitivity, true-positive rate), $\mathrm{TP} / (\mathrm{TP} + \mathrm{FN})$
 * (precision-recall-and-f-score), as scikit-learn's `recall_score`.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged recall; NaN (or `zeroDivision`) for a class with no true cases.
 *
 * @example A binary problem
 * print(recall([0, 1, 1, 0, 1], [0, 1, 0, 1, 1]))
 *
 * @example Three classes, averaged three ways (the scikit-learn example)
 * const yTrue = [0, 1, 2, 0, 1, 2]
 * const yPred = [0, 2, 1, 0, 0, 1]
 * print('macro =', recall(yTrue, yPred, { average: 'macro' }))
 * print('micro =', recall(yTrue, yPred, { average: 'micro' }))
 * print('weighted =', recall(yTrue, yPred, { average: 'weighted' }))
 */
export const recall = defineMetric(
  averagedInfo('recall', 'Recall', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(recallStat, yTrue, yPred, options),
)

/**
 * The $F_\beta$ score, $(1 + \beta^2) P R / (\beta^2 P + R)$ for precision $P$ and recall $R$, computed from
 * counts as $(1 + \beta^2)\mathrm{TP} / ((1 + \beta^2)\mathrm{TP} + \beta^2\mathrm{FN} + \mathrm{FP})$ (van
 * Rijsbergen, 1979), so it is defined whenever any of the three counts is non-zero. $\beta > 1$ weights recall more.
 * Default $\beta = 1$.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `beta`, the weight $\beta$ of recall, and the averaging options of `AverageOptions`.
 * @returns The averaged $F_\beta$ score, in $[0, 1]$.
 *
 * @example Precision 1 and recall 0.5, weighted three ways
 * const yTrue = [1, 1, 1, 1, 0, 0]
 * const yPred = [1, 1, 0, 0, 0, 0]
 * print('beta = 0.5:', fBeta(yTrue, yPred, { beta: 0.5 }))
 * print('beta = 1:  ', fBeta(yTrue, yPred))
 * print('beta = 2:  ', fBeta(yTrue, yPred, { beta: 2 }))
 */
export const fBeta = defineMetric(
  averagedInfo('fBeta', 'F-β score', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions & { beta?: number } = {}): number =>
    averagedMetric(fStat(options.beta ?? 1), yTrue, yPred, options),
)

/**
 * The $F_1$ score, the harmonic mean of precision and recall:
 * $2\mathrm{TP} / (2\mathrm{TP} + \mathrm{FP} + \mathrm{FN})$, as scikit-learn's `f1_score`.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged $F_1$ score, in $[0, 1]$.
 *
 * @example Binary, and three classes (the scikit-learn example)
 * print('binary =', f1([0, 1, 1, 0, 1], [0, 1, 0, 1, 1]))
 * const yTrue = [0, 1, 2, 0, 1, 2]
 * const yPred = [0, 2, 1, 0, 0, 1]
 * print('macro =', f1(yTrue, yPred, { average: 'macro' }))
 * print('micro =', f1(yTrue, yPred, { average: 'micro' }))
 */
export const f1 = defineMetric(
  averagedInfo('f1', 'F₁ score', 'precision-recall-and-f-score'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(fStat(1), yTrue, yPred, options),
)

/**
 * Specificity (true-negative rate), $\mathrm{TN} / (\mathrm{TN} + \mathrm{FP})$
 * (sensitivity-specificity-and-predictive-values).
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged specificity, in $[0, 1]$.
 *
 * @example Three of four negatives found
 * print(specificity([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
export const specificity = defineMetric(
  averagedInfo('specificity', 'Specificity', 'sensitivity-specificity-and-predictive-values'),
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(specificityStat, yTrue, yPred, options),
)

/**
 * Negative predictive value, $\mathrm{TN} / (\mathrm{TN} + \mathrm{FN})$
 * (sensitivity-specificity-and-predictive-values).
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged negative predictive value, in $[0, 1]$.
 *
 * @example Three of four predicted negatives are negative
 * print(negativePredictiveValue([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
export const negativePredictiveValue = defineMetric(
  averagedInfo('negativePredictiveValue', 'Negative predictive value', 'sensitivity-specificity-and-predictive-values'),
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(npvStat, yTrue, yPred, options),
)

/**
 * False-positive rate (fall-out), $\mathrm{FP} / (\mathrm{FP} + \mathrm{TN}) = 1 - \mathrm{specificity}$.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged false-positive rate, in $[0, 1]$; lower is better.
 *
 * @example One of four negatives flagged
 * print(falsePositiveRate([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
export const falsePositiveRate = defineMetric(
  {
    ...averagedInfo('falsePositiveRate', 'False-positive rate', 'sensitivity-specificity-and-predictive-values'),
    direction: 'lower',
  },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(fprStat, yTrue, yPred, options),
)

/**
 * False-negative rate (miss rate), $\mathrm{FN} / (\mathrm{FN} + \mathrm{TP}) = 1 - \mathrm{recall}$.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged false-negative rate, in $[0, 1]$; lower is better.
 *
 * @example One of two positives missed
 * print(falseNegativeRate([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
export const falseNegativeRate = defineMetric(
  {
    ...averagedInfo('falseNegativeRate', 'False-negative rate', 'sensitivity-specificity-and-predictive-values'),
    direction: 'lower',
  },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(fnrStat, yTrue, yPred, options),
)

/**
 * Informedness (Youden's $J$), $\mathrm{TPR} + \mathrm{TNR} - 1$ (Youden, 1950). 0 for any classifier that ignores
 * its input.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged informedness, in $[-1, 1]$.
 *
 * @example A classifier, and one that always says positive
 * const yTrue = [0, 0, 0, 0, 1, 1]
 * print('classifier:', informedness(yTrue, [0, 0, 0, 1, 1, 0]))
 * print('always positive:', informedness(yTrue, [1, 1, 1, 1, 1, 1]))
 */
export const informedness = defineMetric(
  { ...averagedInfo('informedness', 'Informedness', 'sensitivity-specificity-and-predictive-values'), range: [-1, 1] },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(informednessStat, yTrue, yPred, options),
)

/**
 * Markedness, $\mathrm{PPV} + \mathrm{NPV} - 1$ (Powers, 2011): informedness with the roles of truth and prediction
 * exchanged.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged markedness, in $[-1, 1]$.
 *
 * @example Precision 0.5 and NPV 0.75
 * print(markedness([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
export const markedness = defineMetric(
  { ...averagedInfo('markedness', 'Markedness', 'sensitivity-specificity-and-predictive-values'), range: [-1, 1] },
  (yTrue: Labels, yPred: Labels, options: AverageOptions = {}): number =>
    averagedMetric(markednessStat, yTrue, yPred, options),
)

/**
 * The Jaccard index $\mathrm{TP} / (\mathrm{TP} + \mathrm{FP} + \mathrm{FN})$ (hamming-jaccard-and-exact-match),
 * averaged like precision, as scikit-learn's `jaccard_score`; with multi-label rows and `samples` averaging it is the
 * mean per-case $\lvert \hat{Y}_i \cap Y_i \rvert / \lvert \hat{Y}_i \cup Y_i \rvert$.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options The averaging rule, positive class, classes, `zeroDivision` and case weights (`AverageOptions`).
 * @returns The averaged Jaccard index, in $[0, 1]$.
 *
 * @example Multi-label rows, by label and by case (the scikit-learn example)
 * const yTrue = [[0, 1, 1], [1, 1, 0]]
 * const yPred = [[1, 1, 1], [1, 0, 0]]
 * print('macro over labels =', jaccardScore(yTrue, yPred, { average: 'macro' }))
 * print('mean over cases =', jaccardScore(yTrue, yPred, { average: 'samples' }))
 */
export const jaccardScore = defineMetric(
  averagedInfo('jaccardScore', 'Jaccard index', 'hamming-jaccard-and-exact-match'),
  (yTrue: ClassificationInput, yPred: ClassificationInput, options: AverageOptions = {}): number =>
    averagedMetric(jaccardStat, yTrue, yPred, options),
)

/**
 * Per-class precision, recall, $F_\beta$ and support, aligned with `classes` (scikit-learn's
 * `precision_recall_fscore_support` with `average=None`).
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @param options `beta`, the weight $\beta$ of recall in the F-score (default 1); `labels`, the classes and their
 *   order (default those present); `zeroDivision`, the value of an undefined ratio (default NaN); and `sampleWeight`,
 *   a weight per case.
 * @returns The `classes` and, as tensors aligned with them, each class's `precision`, `recall`, `fScore` and
 *   `support` (its true cases).
 *
 * @example Three classes (the scikit-learn example)
 * const yTrue = ['cat', 'dog', 'pig', 'cat', 'dog', 'pig']
 * const yPred = ['cat', 'pig', 'dog', 'cat', 'cat', 'dog']
 * const r = precisionRecallFscoreSupport(yTrue, yPred)
 * print('classes =', r.classes)
 * print('precision =', r.precision, ' recall =', r.recall)
 * print('F1 =', r.fScore, ' support =', r.support)
 */
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
 * The Matthews correlation coefficient (Matthews, 1975): for two classes
 * $(\mathrm{TP} \cdot \mathrm{TN} - \mathrm{FP} \cdot \mathrm{FN}) / \sqrt{\text{product of the margins}}$, the
 * Pearson correlation of the 0/1 truth and prediction; for $K$ classes the Gorodkin form
 * $(cn - \sum_k p_k t_k) / \sqrt{(n^2 - \sum_k p_k^2)(n^2 - \sum_k t_k^2)}$ with $c$ correct, $p_k$ predicted and
 * $t_k$ true counts. NaN when a margin is empty (scikit-learn returns 0).
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options `labels`, the classes to count (cases with other labels are left out), and `sampleWeight`, a weight
 *   per case.
 * @returns The coefficient, in $[-1, 1]$.
 *
 * @example Worse than chance (the scikit-learn example)
 * print(matthewsCorrelation([1, 1, 1, -1], [1, -1, 1, 1]))
 *
 * @example Three classes
 * print(matthewsCorrelation([0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2]))
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

/**
 * Disagreement weights for Cohen's $\kappa$: none (0/1), linear $\lvert j - k \rvert$, quadratic $(j - k)^2$, or an
 * explicit $K \times K$ matrix.
 */
export type KappaWeights = 'none' | 'linear' | 'quadratic' | Rows

/**
 * Weighted $\kappa$ from a $K \times K$ agreement table $\Cmat$ (rows: rater one, columns: rater two):
 * $1 - \sum_{jk} w_{jk} C_{jk} / \sum_{jk} w_{jk} E_{jk}$ with $E_{jk} = r_j c_k / n$ the chance-expected counts, $r_j$
 * and $c_k$ the row and column totals (Cohen, 1960; Cohen, 1968). Unweighted $\kappa$ is the 0/1 weight. A table or
 * weight matrix that is not $K \times K$ throws `ShapeError`.
 *
 * @param table The agreement table $\Cmat$: counts, rows for the first rater's classes and columns for the second's,
 *   in the same order.
 * @param weights The disagreement weights $w_{jk}$: `none`, `linear`, `quadratic` (by class index), or a $K \times K$
 *   matrix.
 * @returns $\kappa$: 1 for perfect agreement, 0 for chance agreement.
 *
 * @example A two-class table
 * print(kappaFromTable([[20, 5], [10, 15]]))
 *
 * @example Ordinal classes: near misses cost less
 * const table = [[10, 4, 1], [3, 12, 3], [0, 4, 13]]
 * print('unweighted =', kappaFromTable(table))
 * print('linear =', kappaFromTable(table, 'linear'))
 * print('quadratic =', kappaFromTable(table, 'quadratic'))
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
 * Cohen's $\kappa$ between two labellings (true labels and predictions, or two raters):
 * $(p_o - p_e) / (1 - p_e)$, with $p_e$ from each labelling's own marginal frequencies (cohens-kappa), as
 * scikit-learn's `cohen_kappa_score`. `weights` gives weighted $\kappa$ for ordinal classes, the order being `labels`
 * (default the sorted labels).
 *
 * @param yTrue The first labelling (true labels, or one rater's).
 * @param yPred The second labelling, one label per case.
 * @param options `weights`, the disagreement weights (`none`, `linear`, `quadratic` or a matrix; default `none`), and
 *   `labels`, the classes and their order (cases with other labels are left out).
 * @returns $\kappa$, at most 1; 0 for chance agreement.
 *
 * @example Two raters (the scikit-learn example)
 * const a = ['negative', 'positive', 'negative', 'neutral', 'positive']
 * const b = ['negative', 'positive', 'negative', 'neutral', 'negative']
 * print(cohensKappa(a, b))
 *
 * @example Ordinal grades, weighted
 * const a = [1, 2, 3, 3, 2, 1]
 * const b = [1, 3, 3, 2, 2, 1]
 * print('unweighted =', cohensKappa(a, b))
 * print('quadratic =', cohensKappa(a, b, { weights: 'quadratic' }))
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
 * Hamming loss: the fraction of individual label decisions that are wrong,
 * $\sum_i \lvert \hat{Y}_i \mathbin{\triangle} Y_i \rvert / (nL)$ for multi-label rows
 * (hamming-jaccard-and-exact-match), as scikit-learn's `hamming_loss`; for single labels, the fraction of wrong
 * labels.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @returns The fraction of wrong label decisions, in $[0, 1]$.
 *
 * @example Single labels, and multi-label rows (the scikit-learn examples)
 * print('labels:', hammingLoss([1, 2, 3, 4], [2, 2, 3, 4]))
 * print('rows:', hammingLoss([[0, 1], [1, 1]], [[0, 0], [0, 0]]))
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

/**
 * Exact match (subset accuracy): the fraction of cases whose whole label set is right. For single labels it is
 * accuracy.
 *
 * @param yTrue The true labels, or an $n \times L$ matrix of 0/1 label sets.
 * @param yPred The predicted labels, matching `yTrue`.
 * @returns The fraction of cases predicted exactly, in $[0, 1]$.
 *
 * @example One of two label sets exactly right
 * print(exactMatch([[0, 1], [1, 1]], [[0, 1], [1, 0]]))
 */
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

/**
 * $\mathrm{LR}^+ = \mathrm{TPR} / \mathrm{FPR}$, the factor by which a positive result multiplies the odds of the
 * positive class. Binary; NaN when there are no negatives, no positives or no false positives.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options `positive`, the positive class (default as `positiveOf` chooses it).
 * @returns $\mathrm{LR}^+$, at least 0; higher is better.
 *
 * @example Recall 0.5 and false-positive rate 0.25
 * print(positiveLikelihoodRatio([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
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

/**
 * $\mathrm{LR}^- = \mathrm{FNR} / \mathrm{TNR}$, the factor by which a negative result multiplies the odds. Binary;
 * lower is better.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options `positive`, the positive class (default as `positiveOf` chooses it).
 * @returns $\mathrm{LR}^-$, at least 0.
 *
 * @example Miss rate 0.5 and specificity 0.75
 * print(negativeLikelihoodRatio([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
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

/**
 * The diagnostic odds ratio
 * $\mathrm{LR}^+ / \mathrm{LR}^- = (\mathrm{TP} \cdot \mathrm{TN}) / (\mathrm{FP} \cdot \mathrm{FN})$. Binary;
 * NaN when $\mathrm{FP}$ or $\mathrm{FN}$ is 0.
 *
 * @param yTrue The true labels.
 * @param yPred The predicted labels, one per case.
 * @param options `positive`, the positive class (default as `positiveOf` chooses it).
 * @returns The odds ratio, at least 0.
 *
 * @example One each of TP, FP and FN, three TN
 * print(diagnosticOddsRatio([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 0]))
 */
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
