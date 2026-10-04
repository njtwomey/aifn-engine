/**
 * Confusion matrices and the per-class counts every threshold metric is built from: binary, multiclass (one against
 * the rest) and multi-label, with the averaging rules of scikit-learn (micro, macro, weighted, samples).
 */

import type { Tensor } from 'aifn-compute/foundation/tensor'
import {
  binaryTruth,
  caseWeights,
  classesOf,
  compareLabels,
  dense,
  divide,
  encodeLabels,
  isMatrixLike,
  labelList,
  matrix,
  positiveOf,
  sameLength,
  values,
  vector,
  type Data,
  type Label,
  type Labels,
  type Rows,
} from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A confusion matrix: `matrix[j][k]` counts cases of true class `classes[j]` predicted as `classes[k]`. */
export type ConfusionMatrix = {
  /** The class labels, in the order of the rows and columns. */
  classes: Label[]
  /** K × K counts (or weighted counts, or proportions when normalised). Rows are true classes. */
  matrix: Tensor
  /** Number of cases (sum of weights when weighted). */
  n: number
}

/** Options for building a confusion matrix. */
export type ConfusionOptions = {
  /** The classes and their order; default the sorted labels present in either input. */
  labels?: readonly Label[]
  /** A weight per case; counts become sums of weights. */
  sampleWeight?: Data
  /** Divide each row (`true`), column (`pred`) or the whole matrix (`all`) by its total; default counts. */
  normalise?: 'true' | 'pred' | 'all'
}

/**
 * The confusion matrix C of labels `yTrue` and predictions `yPred` (n each): C[j][k] counts the cases of true class j
 * predicted as class k (Fawcett 2006). Cases whose label is not in `labels` are left out.
 */
export function confusionMatrix(yTrue: Labels, yPred: Labels, options: ConfusionOptions = {}): ConfusionMatrix {
  const t = labelList(yTrue)
  const p = labelList(yPred)
  sameLength(t, p, 'confusionMatrix')
  const classes = options.labels ? [...options.labels] : classesOf(t, p)
  const K = classes.length
  const ti = encodeLabels(t, classes)
  const pi = encodeLabels(p, classes)
  const w = caseWeights(options.sampleWeight, t.length, 'confusionMatrix')
  const c = new Float64Array(K * K)
  let n = 0
  for (let i = 0; i < t.length; i++) {
    if (ti[i] < 0 || pi[i] < 0) continue
    const wi = w ? w[i] : 1
    c[ti[i] * K + pi[i]] += wi
    n += wi
  }
  if (options.normalise === 'all') for (let k = 0; k < K * K; k++) c[k] = divide(c[k], n)
  if (options.normalise === 'true')
    for (let j = 0; j < K; j++) {
      let s = 0
      for (let k = 0; k < K; k++) s += c[j * K + k]
      for (let k = 0; k < K; k++) c[j * K + k] = divide(c[j * K + k], s)
    }
  if (options.normalise === 'pred')
    for (let k = 0; k < K; k++) {
      let s = 0
      for (let j = 0; j < K; j++) s += c[j * K + k]
      for (let j = 0; j < K; j++) c[j * K + k] = divide(c[j * K + k], s)
    }
  return { classes, matrix: matrix(c, K, K), n }
}

/**
 * The margins of a confusion matrix and the rates read along them (confusion-matrix): each row's total and the share
 * of it on the diagonal, each column's total and the share of it on the diagonal, and the grand total.
 *
 * For two classes with the positive class k: row k's rate is the TPR (recall, sensitivity) and its complement the FNR;
 * the other row's rate is the TNR (specificity) and its complement the FPR; column k's rate is the PPV (precision)
 * and its complement the FDR; the other column's rate is the NPV and its complement the FOR. With K classes the same
 * quantities are each class's one-against-rest recall and precision.
 */
export type ConfusionMargins = {
  /** The class labels, in the order of the rows and columns. */
  classes: Label[]
  /** Row totals [K]: the cases of each actual class. */
  actual: Tensor
  /** Column totals [K]: the cases predicted as each class. */
  predicted: Tensor
  /** The grand total. */
  n: number
  /** Cⱼⱼ / row j [K]: the recall of each class (TPR and TNR for two classes). */
  rowRate: Tensor
  /** 1 − rowRate [K]: the miss rate of each class (FNR and FPR for two classes). */
  rowMiss: Tensor
  /** Cₖₖ / column k [K]: the precision of each class (PPV and NPV for two classes). */
  columnRate: Tensor
  /** 1 − columnRate [K]: the false discovery rate of each class (FDR and FOR for two classes). */
  columnMiss: Tensor
  /** Row total / n [K]: each class's share of the cases (the prevalence of the positive class). */
  prevalence: Tensor
  /** The diagonal's share of the cases. */
  accuracy: number
}

/**
 * The margins of a confusion matrix of counts: row and column totals, the diagonal's share of each (recall and
 * precision per class, with their complements) and the accuracy. Undefined ratios (an empty row or column) are NaN.
 */
export function confusionMargins(cm: ConfusionMatrix): ConfusionMargins {
  const K = cm.classes.length
  const c = values(cm.matrix)
  const rows = new Float64Array(K)
  const cols = new Float64Array(K)
  let n = 0
  let diagonal = 0
  for (let j = 0; j < K; j++)
    for (let k = 0; k < K; k++) {
      const v = c[j * K + k]
      rows[j] += v
      cols[k] += v
      n += v
      if (j === k) diagonal += v
    }
  const d = (j: number) => c[j * K + j]
  return {
    classes: [...cm.classes],
    actual: vector(rows),
    predicted: vector(cols),
    n,
    rowRate: vector(Array.from(rows, (r, j) => divide(d(j), r))),
    rowMiss: vector(Array.from(rows, (r, j) => divide(r - d(j), r))),
    columnRate: vector(Array.from(cols, (s, k) => divide(d(k), s))),
    columnMiss: vector(Array.from(cols, (s, k) => divide(s - d(k), s))),
    prevalence: vector(Array.from(rows, (r) => divide(r, n))),
    accuracy: divide(diagonal, n),
  }
}

/** The four cells of a binary confusion matrix. */
export type BinaryCounts = { tp: number; fp: number; fn: number; tn: number }

/**
 * The binary confusion counts of labels and predictions for one positive class (default `1`, else `true`, else the
 * last label in order).
 */
export function binaryCounts(
  yTrue: Labels,
  yPred: Labels,
  options: { positive?: Label; sampleWeight?: Data } = {},
): BinaryCounts {
  const t = labelList(yTrue)
  const p = labelList(yPred)
  sameLength(t, p, 'binaryCounts')
  const pos = positiveOf(classesOf(t, p), options.positive)
  const w = caseWeights(options.sampleWeight, t.length, 'binaryCounts')
  const out = { tp: 0, fp: 0, fn: 0, tn: 0 }
  for (let i = 0; i < t.length; i++) {
    const wi = w ? w[i] : 1
    const a = t[i] === pos
    const b = p[i] === pos
    if (a && b) out.tp += wi
    else if (b) out.fp += wi
    else if (a) out.fn += wi
    else out.tn += wi
  }
  return out
}

/**
 * The binary confusion counts when cases with `score ≥ threshold` are predicted positive (the convention of
 * scikit-learn's curves, so that a threshold equal to a score includes that case).
 */
export function countsAtThreshold(
  yTrue: Labels,
  scores: Data,
  threshold: number,
  options: { positive?: Label } = {},
): BinaryCounts {
  const { y } = binaryTruth(yTrue, options.positive)
  const s = values(scores)
  sameLength(y, s, 'countsAtThreshold')
  const out = { tp: 0, fp: 0, fn: 0, tn: 0 }
  for (let i = 0; i < y.length; i++) {
    const predicted = s[i] >= threshold
    if (y[i] && predicted) out.tp++
    else if (predicted) out.fp++
    else if (y[i]) out.fn++
    else out.tn++
  }
  return out
}

/** Per-class one-against-rest counts read off a confusion matrix, as tensors of length K. */
export type OneVsRestCounts = {
  classes: Label[]
  tp: Tensor
  fp: Tensor
  fn: Tensor
  tn: Tensor
  /** Cases of each true class (row sums). */
  support: Tensor
}

/**
 * The one-against-rest binary counts of each class of a K-class confusion matrix: TPₖ = Cₖₖ, FNₖ the rest of row k,
 * FPₖ the rest of column k, TNₖ everything else.
 */
export function oneVsRest(cm: ConfusionMatrix): OneVsRestCounts {
  const t = talliesFromMatrix(cm)
  return {
    classes: t.classes,
    tp: vector(t.tp),
    fp: vector(t.fp),
    fn: vector(t.fn),
    tn: vector(t.tn),
    support: vector(t.support),
  }
}

/** Every rate of a binary confusion matrix (NaN where a denominator is 0). */
export type BinaryRates = {
  n: number
  prevalence: number
  accuracy: number
  errorRate: number
  balancedAccuracy: number
  /** Recall, sensitivity, true-positive rate: TP / P. */
  recall: number
  /** Specificity, true-negative rate: TN / N. */
  specificity: number
  falsePositiveRate: number
  falseNegativeRate: number
  /** Precision, positive predictive value: TP / (TP + FP). */
  precision: number
  negativePredictiveValue: number
  falseDiscoveryRate: number
  f1: number
  /** TP / (TP + FP + FN), the Jaccard index of the positive class. */
  jaccard: number
  matthewsCorrelation: number
  cohensKappa: number
  /** Youden's J: TPR + TNR − 1. */
  informedness: number
  /** PPV + NPV − 1. */
  markedness: number
  positiveLikelihoodRatio: number
  negativeLikelihoodRatio: number
  diagnosticOddsRatio: number
  /** Fraction predicted positive, (TP + FP) / n. */
  selectionRate: number
}

/**
 * Every threshold metric of a binary confusion matrix, as in the notes' summary table (confusion-matrix,
 * sensitivity-specificity-and-predictive-values). Undefined ratios are NaN.
 */
export function binaryRates({ tp, fp, fn, tn }: BinaryCounts): BinaryRates {
  const n = tp + fp + fn + tn
  const P = tp + fn
  const N = fp + tn
  const recall = divide(tp, P)
  const specificity = divide(tn, N)
  const precision = divide(tp, tp + fp)
  const npv = divide(tn, tn + fn)
  const accuracy = divide(tp + tn, n)
  const expected = divide((tp + fp) * P + (fn + tn) * N, n * n)
  return {
    n,
    prevalence: divide(P, n),
    accuracy,
    errorRate: 1 - accuracy,
    balancedAccuracy: (recall + specificity) / 2,
    recall,
    specificity,
    falsePositiveRate: divide(fp, N),
    falseNegativeRate: divide(fn, P),
    precision,
    negativePredictiveValue: npv,
    falseDiscoveryRate: divide(fp, tp + fp),
    f1: divide(2 * tp, 2 * tp + fp + fn),
    jaccard: divide(tp, tp + fp + fn),
    matthewsCorrelation: divide(tp * tn - fp * fn, Math.sqrt((tp + fp) * P * N * (tn + fn))),
    cohensKappa: divide(accuracy - expected, 1 - expected),
    informedness: recall + specificity - 1,
    markedness: precision + npv - 1,
    positiveLikelihoodRatio: divide(recall, divide(fp, N)),
    negativeLikelihoodRatio: divide(divide(fn, P), specificity),
    diagnosticOddsRatio: divide(tp * tn, fp * fn),
    selectionRate: divide(tp + fp, n),
  }
}

// ── Tallies: the per-class counts the averaged metrics use ───────────────────────────────────────────────────────────

/**
 * Per-class one-against-rest counts, and for multi-label data the per-case counts that `samples` averaging needs.
 * Internal to the module.
 */
export type Tallies = {
  classes: Label[]
  tp: Float64Array
  fp: Float64Array
  fn: Float64Array
  tn: Float64Array
  support: Float64Array
  /** Number of cases. */
  n: number
  multilabel: boolean
  /** Multi-label only: per-case TP, FP, FN, TN over the labels. */
  cases?: { tp: Float64Array; fp: Float64Array; fn: Float64Array; tn: Float64Array }
}

function talliesFromMatrix(cm: ConfusionMatrix): Tallies {
  const K = cm.classes.length
  const c = cm.matrix.data as Float64Array
  const tp = new Float64Array(K)
  const fp = new Float64Array(K)
  const fn = new Float64Array(K)
  const tn = new Float64Array(K)
  const support = new Float64Array(K)
  let total = 0
  for (let k = 0; k < K * K; k++) total += c[k]
  for (let k = 0; k < K; k++) {
    let row = 0
    let col = 0
    for (let j = 0; j < K; j++) {
      row += c[k * K + j]
      col += c[j * K + k]
    }
    tp[k] = c[k * K + k]
    fn[k] = row - tp[k]
    fp[k] = col - tp[k]
    tn[k] = total - row - col + tp[k]
    support[k] = row
  }
  return { classes: cm.classes, tp, fp, fn, tn, support, n: total, multilabel: false }
}

/**
 * Tallies of single-label predictions (vectors of labels) or multi-label predictions (n × L matrices of 0/1, rows as
 * cases and columns as labels).
 */
export function tallies(
  yTrue: Labels | Rows,
  yPred: Labels | Rows,
  options: { labels?: readonly Label[]; sampleWeight?: Data } = {},
): Tallies {
  if (isMatrixLike(yTrue) || isMatrixLike(yPred)) {
    const t = dense(yTrue as Rows, 'multi-label truth')
    const p = dense(yPred as Rows, 'multi-label prediction')
    if (t.rows !== p.rows || t.cols !== p.cols)
      throw new ShapeError(
        'metrics',
        `metrics: multi-label inputs have shapes ${t.rows}×${t.cols} and ${p.rows}×${p.cols}`,
      )
    const L = t.cols
    const w = caseWeights(options.sampleWeight, t.rows, 'multi-label metric')
    const tp = new Float64Array(L)
    const fp = new Float64Array(L)
    const fn = new Float64Array(L)
    const tn = new Float64Array(L)
    const cases = {
      tp: new Float64Array(t.rows),
      fp: new Float64Array(t.rows),
      fn: new Float64Array(t.rows),
      tn: new Float64Array(t.rows),
    }
    for (let i = 0; i < t.rows; i++) {
      const wi = w ? w[i] : 1
      for (let l = 0; l < L; l++) {
        const a = t.data[i * L + l] !== 0
        const b = p.data[i * L + l] !== 0
        const cell = a && b ? 'tp' : b ? 'fp' : a ? 'fn' : 'tn'
        ;({ tp, fp, fn, tn })[cell][l] += wi
        cases[cell][i] += 1
      }
    }
    const support = Float64Array.from(tp, (v, l) => v + fn[l])
    const classes = Array.from({ length: L }, (_, l) => l)
    return { classes, tp, fp, fn, tn, support, n: t.rows, multilabel: true, cases }
  }
  return talliesFromMatrix(
    confusionMatrix(yTrue as Labels, yPred as Labels, { labels: options.labels, sampleWeight: options.sampleWeight }),
  )
}

/**
 * How per-class values combine into one number (averaging-multiclass-metrics):
 * - `binary`: the positive class only (two-class problems).
 * - `micro`: pool the counts over classes, then compute once.
 * - `macro`: the unweighted mean over classes.
 * - `weighted`: the mean weighted by each class's support (true count).
 * - `samples`: multi-label only, compute per case over its labels and average over cases.
 */
export type Average = 'binary' | 'micro' | 'macro' | 'weighted' | 'samples'

/** Options shared by the averaged threshold metrics. */
export type AverageOptions = {
  /** Default `binary` for two classes and `macro` otherwise (and for multi-label data). */
  average?: Average
  /** The positive class for `binary` averaging. */
  positive?: Label
  /** The classes to include and their order; default those present in either input. */
  labels?: readonly Label[]
  /** The value of a ratio whose denominator is 0; default NaN, so an undefined value is visible. */
  zeroDivision?: number
  /** A weight per case. */
  sampleWeight?: Data
}

/** A metric of one binary table: (TP, FP, FN, TN) → value, NaN or `zero` where undefined. */
export type CountStatistic = (tp: number, fp: number, fn: number, tn: number, zero: number) => number

/** Apply a count statistic to tallies with an averaging rule. */
export function averaged(stat: CountStatistic, t: Tallies, options: AverageOptions): number {
  const zero = options.zeroDivision ?? NaN
  const K = t.classes.length
  const average = options.average ?? (t.multilabel || K > 2 ? 'macro' : 'binary')
  const sum = (a: Float64Array) => a.reduce((s, v) => s + v, 0)
  switch (average) {
    case 'binary': {
      if (t.multilabel) throw new DomainError('metrics', "metrics: 'binary' averaging needs single-label input")
      if (K > 2)
        throw new DomainError('metrics', `metrics: 'binary' averaging needs two classes, got ${K}; choose an average`)
      const pos = positiveOf(t.classes, options.positive)
      const k = t.classes.findIndex((c) => compareLabels(c, pos) === 0)
      if (k < 0) return stat(0, 0, 0, t.n, zero)
      return stat(t.tp[k], t.fp[k], t.fn[k], t.tn[k], zero)
    }
    case 'micro':
      return stat(sum(t.tp), sum(t.fp), sum(t.fn), sum(t.tn), zero)
    case 'macro':
    case 'weighted': {
      let s = 0
      let w = 0
      for (let k = 0; k < K; k++) {
        const wk = average === 'macro' ? 1 : t.support[k]
        if (wk === 0) continue
        s += wk * stat(t.tp[k], t.fp[k], t.fn[k], t.tn[k], zero)
        w += wk
      }
      return divide(s, w)
    }
    case 'samples': {
      if (!t.cases) throw new DomainError('metrics', "metrics: 'samples' averaging needs multi-label input")
      const c = t.cases
      let s = 0
      for (let i = 0; i < t.n; i++) s += stat(c.tp[i], c.fp[i], c.fn[i], c.tn[i], zero)
      return divide(s, t.n)
    }
  }
}

/** Per-class values of a count statistic, as a tensor aligned with `tallies.classes`. */
export function perClass(stat: CountStatistic, t: Tallies, zeroDivision = NaN): Tensor {
  return vector(Array.from(t.classes, (_, k) => stat(t.tp[k], t.fp[k], t.fn[k], t.tn[k], zeroDivision)))
}
