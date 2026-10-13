/**
 * Cross-validation (plan §5.5; Stone, 1974, "Cross-validatory choice and assessment of statistical predictions",
 * JRSS B 36; Hastie, Tibshirani and Friedman, 2009, "The Elements of Statistical Learning", §7.10): fit on each
 * split's training rows, evaluate on its test rows, keep everything. Metrics are registered metrics (those of
 * `aifn-compute/learning/metrics`, or any function carrying a `MetricInfo`), read by their `info.capability` and keyed
 * by `info.key`. Randomness comes from one stream, split into a child for the splitter and one per fold's fit, so a
 * cross-validation is reproducible from its seed.
 *
 * Classification metrics are told the classes of the whole data set, not those of the fold: a metric that reads
 * labels, scores or probabilities (by its `info.inputs`) gets `{ labels, positive }` as its options, with `labels`
 * the sorted classes of all of `y` and, for two classes, `positive` the positive class chosen as the metrics module
 * chooses it (`1`, else `true`, else the last class). A test fold that holds only negatives then scores recall for the
 * positive class (NaN, as nothing is positive) instead of silently for the negative one. Option `classes: 'fold'`
 * restores the per-fold behaviour.
 *
 * With `pointwise`, each metric is also evaluated on every test row by itself, which gives the held-out loss of each
 * row (the input of Bates, Hastie and Tibshirani's nested cross-validation, and of a standard error over rows). That
 * is meaningful only for a metric whose value on a set is the mean of its values on the rows, $m(\mathcal{T}) =
 * \frac{1}{\abs{\mathcal{T}}} \sum_{i \in \mathcal{T}} m(\{i\})$: accuracy and the error rate, mean squared and
 * absolute errors and the other mean losses, log loss, the Brier score. Such a metric is detected by checking that
 * equation on every fold; one that fails it (root mean squared error, $R^2$, recall, AUROC) is left out.
 */

import type { MetricCapability } from 'aifn-compute/foundation/contracts'
import {
  hasTraining,
  metricInput,
  outputFor,
  outputs,
  rowCount,
  score,
  takeData,
  takeRows,
  type CapabilityOf,
  type Column,
  type Features,
  type FitOptions,
  type Outputs,
  type Requirement,
  type ServedMetric,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import { child, type Stream } from 'aifn-compute/foundation/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { now, type Trace } from 'aifn-compute/foundation/trace'
import { assignment, type Split, type Splitter } from './splitters'

/** A dataset for cross-validation: inputs `x`, targets `y` (a tensor) and optional group labels `groups`. */
export type CrossValidationData<X extends Features> = Supervised<X, Tensor> & { readonly groups?: Column }

/** Something with `fit` on supervised data: an estimator, a pipeline or a search's estimator. */
export type Fittable<X extends Features, M> = {
  /** A readable name. */
  readonly name: string
  /** Fit on supervised data, returning the fitted model `M`. */
  fit(data: Supervised<X, Tensor>, options?: FitOptions): M
}

/** One fold of a cross-validation. */
export interface Fold<M> {
  /** The fold's position $f$ among the splits, from 0. */
  index: number
  /** Sorted int32 row indices of the training set. */
  train: Tensor
  /** Sorted int32 row indices of the test set. */
  test: Tensor
  /** The model fitted on the training rows, with all its fitted state. */
  model: M
  /** The model's outputs on the test rows, by capability (those the metrics need). */
  predictions: Outputs
  /** Each metric on the test rows, by `info.key`. */
  metrics: Record<string, number>
  /** Each metric on the training rows, when `trainMetrics` is set. */
  trainMetrics?: Record<string, number>
  /** The model's training trace, when it kept one. */
  training?: Trace<unknown>
  /** Wall time of the fit, in milliseconds. */
  fitMs: number
  /**
   * With `pointwise`: each pointwise metric on each test row by itself, by `info.key`, in the order of `test` (their
   * mean is the fold's value).
   */
  pointwise?: Record<string, Tensor>
}

/** The result of `crossValidate`. */
export interface CrossValidation<M> {
  /** The splitter's name. */
  splitter: string
  /** The splits, as the splitter returned them. */
  splits: Split[]
  /** Fold assignment, $k \times n$ for $k$ folds: 1 test, 0 train, $-1$ unused (see `assignment`). */
  assignment: Tensor
  /** Every fold, in split order. */
  folds: Fold<M>[]
  /** Each metric's value per fold, a vector of $k$ values, by `info.key`. */
  scores: Record<string, Tensor>
  /** Each metric's mean over folds, by `info.key`. */
  mean: Record<string, number>
  /** Each metric's sample standard deviation over folds (divided by $k - 1$; NaN for one fold), by `info.key`. */
  std: Record<string, number>
  /** Each metric's direction, by `info.key`. */
  directions: Record<string, 'higher' | 'lower'>
  /**
   * Out-of-fold predictions, $n$ values, for each capability the metrics read other than `'predictive'` (`'decide'`,
   * and `'score'` when scores are one value per row): each row's prediction from the model that did not train on it;
   * NaN for rows never tested. With repeated splits the last wins. A capability whose outputs are not one value per
   * test row is left out.
   */
  outOfFold: Partial<Record<MetricCapability, Tensor>>
  /**
   * The classes the classification metrics were given (`labels`), sorted, and the positive class of a two-class
   * problem; absent with `classes: 'fold'` or when no metric reads labels, scores or probabilities.
   */
  classes?: { labels: ClassLabel[]; positive?: ClassLabel }
  /**
   * With `pointwise`: each pointwise metric's held-out value on each row, $n$ values by `info.key` (NaN for rows never
   * tested; with repeated splits the last wins). Per fold, see `Fold.pointwise`. Metrics whose fold values are not the
   * means of their row values are left out.
   */
  pointwise?: Record<string, Tensor>
}

/** A class label, as the metrics take them. */
export type ClassLabel = string | number | boolean

/** Options of `crossValidate`. */
export interface CrossValidateOptions {
  /** Randomness: the splitter gets `child(stream, 'split')`, fold f's fit `child(stream, 'fold', f)`. */
  stream?: Stream
  /** Also evaluate each fold's metrics on its training rows (default false). */
  trainMetrics?: boolean
  /**
   * Where classification metrics take their classes from: `'data'` (default), the classes of the whole `y`, passed to
   * every metric that reads labels, scores or probabilities as `{ labels, positive }`; or `'fold'`, the classes of
   * each fold, as the metric finds them (which changes the positive class of a fold with one class).
   */
  classes?: 'data' | 'fold'
  /** The positive class of a two-class problem; default `1`, else `true`, else the last class. */
  positive?: ClassLabel
  /** Also evaluate each metric on every test row by itself, giving `pointwise` values (default false). */
  pointwise?: boolean
}

/** The options a classification metric is given: the classes, in order, and the positive class. */
type ClassOptions = { labels: ClassLabel[]; positive?: ClassLabel }

/**
 * Order of labels as the metrics order them: booleans (false first), then numbers ascending, then strings by code
 * point.
 *
 * @param a The first label.
 * @param b The second label.
 * @returns A negative number when `a` comes first, a positive one when `b` does, 0 when they are equal.
 */
function compareLabels(a: ClassLabel, b: ClassLabel): number {
  const rank = (v: ClassLabel) => (typeof v === 'boolean' ? 0 : typeof v === 'number' ? 1 : 2)
  const d = rank(a) - rank(b)
  if (d !== 0) return d
  if (typeof a === 'string') return a < (b as string) ? -1 : a > (b as string) ? 1 : 0
  return Number(a) - Number(b)
}

/**
 * The classes of the whole data set for the classification metrics, or undefined when none of the metrics reads
 * labels, scores or probabilities.
 *
 * @param y The targets of every row.
 * @param metrics The metrics.
 * @param positive The positive class asked for, if any.
 * @returns The sorted class list and, for two classes or when asked, the positive class.
 */
function classOptions(y: unknown, metrics: readonly ServedMetric[], positive?: ClassLabel): ClassOptions | undefined {
  if (!metrics.some(readsClasses)) return undefined
  const list: ClassLabel[] = isTensor(y) ? Array.from(toFlat(y)) : Array.from(y as ArrayLike<ClassLabel>)
  const labels = [...new Set(list)].filter((v) => v === v).sort(compareLabels)
  if (positive !== undefined) return { labels, positive }
  if (labels.length !== 2) return { labels }
  // As the metrics' `positiveOf`: 1, else true, else the last class.
  return { labels, positive: labels.includes(1) ? 1 : labels.includes(true) ? true : labels[1] }
}

/**
 * Whether a metric reads classes: its `info.inputs` is labels, scores or probabilities.
 *
 * @param metric The metric.
 * @returns True for a classification metric that takes `labels` and `positive` options.
 */
function readsClasses(metric: ServedMetric): boolean {
  const inputs = (metric.info as { inputs?: string }).inputs
  return inputs === 'labels' || inputs === 'scores' || inputs === 'probabilities'
}

/**
 * Each metric's value on targets `y` given the outputs, with the class options passed to the classification metrics.
 *
 * @param metrics The metrics.
 * @param y The targets of the rows.
 * @param out The model's outputs on the rows, by capability.
 * @param classes The class options, or undefined to call every metric with two arguments.
 * @returns Each metric's value, by `info.key`.
 */
function scoreWith(
  metrics: readonly ServedMetric[],
  y: Tensor,
  out: Outputs,
  classes: ClassOptions | undefined,
): Record<string, number> {
  if (!classes) return score(metrics, y, out)
  const result: Record<string, number> = {}
  for (const m of metrics) {
    if (!readsClasses(m)) {
      Object.assign(result, score([m], y, out))
      continue
    }
    const output = out[m.info.capability]
    if (output === undefined) return score(metrics, y, out) // throws the missing-output error
    const f = m as unknown as (y: unknown, p: unknown, o: ClassOptions) => number
    result[m.info.key] = f(y, metricInput(m, output), classes)
  }
  return result
}

/**
 * The outputs a model gives on some rows for the metrics, and the metrics' values on them.
 *
 * @param model The fitted model; it must have every capability the metrics read.
 * @param x The inputs of the rows.
 * @param y The targets of the rows.
 * @param metrics The metrics to compute.
 * @param classes The class options for the classification metrics; left out, every metric is called with two
 *   arguments.
 * @returns The outputs by capability (`predictions`), and each metric's value by `info.key` (`values`).
 */
function evaluateOn(
  model: unknown,
  x: unknown,
  y: Tensor,
  metrics: readonly ServedMetric[],
  classes?: ClassOptions,
): { predictions: Outputs; values: Record<string, number> } {
  const predictions = outputs(model, metrics, x)
  return { predictions, values: scoreWith(metrics, y, predictions, classes) }
}

/**
 * Each metric on every test row by itself: tensor outputs are cut to the row, and a predictive distribution is asked
 * of the model again on the row's inputs. A metric that throws on one row gets NaN there.
 *
 * @param model The fitted model.
 * @param x The inputs of the test rows.
 * @param y The targets of the test rows.
 * @param predictions The model's outputs on the test rows, by capability.
 * @param metrics The metrics.
 * @param classes The class options for the classification metrics, if any.
 * @returns Each metric's value on each row, by `info.key`, in row order.
 */
function perRow(
  model: unknown,
  x: unknown,
  y: Tensor,
  predictions: Outputs,
  metrics: readonly ServedMetric[],
  classes: ClassOptions | undefined,
): Record<string, Float64Array> {
  const m = y.shape[0] ?? 1
  const values: Record<string, Float64Array> = {}
  for (const { info } of metrics) values[info.key] = new Float64Array(m)
  for (let r = 0; r < m; r++) {
    const row: Outputs = {}
    for (const [capability, output] of Object.entries(predictions) as [MetricCapability, Outputs[MetricCapability]][]) {
      if (output === undefined) continue
      row[capability] = isTensor(output)
        ? takeRows(output, [r])
        : outputFor(model, capability, takeRows(x as Tensor, [r]))
    }
    const yr = takeRows(y, [r])
    for (const metric of metrics) {
      let v: number
      try {
        v = scoreWith([metric], yr, row, classes)[metric.info.key]
      } catch {
        v = NaN
      }
      values[metric.info.key][r] = v
    }
  }
  return values
}

/**
 * Whether row values average to the fold's value: $\abs{\bar v - m} \le 10^{-9}(1 + \abs m)$, or both NaN.
 *
 * @param rows The metric on each test row.
 * @param value The metric on the whole test fold.
 * @returns True when the metric is pointwise on this fold.
 */
function averagesTo(rows: Float64Array, value: number): boolean {
  const mean = rows.reduce((a, b) => a + b, 0) / rows.length
  if (Number.isNaN(value)) return Number.isNaN(mean)
  return Math.abs(mean - value) <= 1e-9 * (1 + Math.abs(value))
}

/**
 * Cross-validate an estimator (or a pipeline): for each split of `splitter`, fit on the training rows and evaluate
 * `metrics` on the test rows. The fitted model must have every capability the metrics need (a compile error
 * otherwise). Everything is kept: the fold assignment matrix, each fold's fitted model, predictions, metrics and
 * training trace, per-metric scores over folds with their mean and standard deviation, and out-of-fold predictions.
 * Folds are fitted in turn, on this thread; whatever the splitter or the fit throws is not caught.
 *
 * @param estimator What to fit on each training set: anything with `name` and `fit(data, options)`.
 * @param data The rows: inputs `x`, targets `y` and, for a grouped splitter, `groups`. Its row-aligned fields are
 *   subset per fold; it is not modified.
 * @param splitter How to split the rows into training and test sets; it is given the row count, `y` and `groups`.
 * @param metrics The metrics to compute on every test set, each a function of the targets and one model output
 *   carrying a `MetricInfo` (`key`, `capability`, `direction`).
 * @param options The stream, whether to score the training rows too, where the classes come from (and the positive
 *   class), and whether to keep each row's held-out value of the pointwise metrics.
 * @returns Every fold, the splits and assignment, each metric's scores per fold with their mean, standard deviation
 *   and direction, the out-of-fold predictions, the classes given to the metrics, and the pointwise values.
 *
 * @example A mean predictor, checked by hand
 * // The model predicts the training mean; each fold's mean squared error can be checked by eye.
 * const meanModel = {
 *   name: 'mean',
 *   fit: (d) => {
 *     const m = mean(d.y)
 *     return { decide: (x) => full([x.shape[0]], m) }
 *   },
 * }
 * const mse = Object.assign((y, p) => mean(square(sub(y, p))), {
 *   info: { key: 'mse', capability: 'decide', direction: 'lower' },
 * })
 * const data = { x: tensor([[0], [1], [2], [3], [4], [5]]), y: tensor([1, 2, 3, 4, 5, 6]) }
 * const cv = crossValidate(meanModel, data, kFold({ k: 3 }), [mse])
 * print('test rows:', cv.splits.map((s) => s.test))
 * print('mse per fold:', cv.scores.mse)
 * print('mean, std:', cv.mean.mse, cv.std.mse)
 * print('out-of-fold predictions:', cv.outOfFold.decide)
 *
 * @example A fold with no positives scores recall for class 1, not class 0
 * // Recall of the positive class the metric is given, or of the largest label it sees when it is given none.
 * const recall = Object.assign(
 *   (y, d, o = {}) => {
 *     const [t, p] = [toFlat(y), toFlat(d)]
 *     const pos = o.positive ?? Math.max(...t, ...p)
 *     return t.filter((v, i) => v === pos && p[i] === pos).length / t.filter((v) => v === pos).length
 *   },
 *   { info: { key: 'recall', capability: 'decide', inputs: 'labels', direction: 'higher' } },
 * )
 * // A model that always predicts 0; the second fold's rows are all of class 0.
 * const zeros = { name: 'zeros', fit: () => ({ decide: (x) => full([x.shape[0]], 0) }) }
 * const data = { x: tensor([[0], [1], [2], [3]]), y: tensor([1, 1, 0, 0]) }
 * const cv = crossValidate(zeros, data, kFold({ k: 2 }), [recall])
 * print('classes from all of y:', cv.classes, cv.scores.recall)
 * print('classes per fold:', crossValidate(zeros, data, kFold({ k: 2 }), [recall], { classes: 'fold' }).scores.recall)
 *
 * @example Held-out squared errors per row; the root mean squared error is not a mean over rows
 * const meanModel = { name: 'mean', fit: (d) => ({ decide: (x) => full([x.shape[0]], mean(d.y)) }) }
 * const mse = Object.assign((y, p) => mean(square(sub(y, p))), {
 *   info: { key: 'mse', capability: 'decide', direction: 'lower' },
 * })
 * const rmse = Object.assign((y, p) => Math.sqrt(mean(square(sub(y, p)))), {
 *   info: { key: 'rmse', capability: 'decide', direction: 'lower' },
 * })
 * const data = { x: tensor([[0], [1], [2], [3], [4], [5]]), y: tensor([1, 2, 3, 4, 5, 7]) }
 * const cv = crossValidate(meanModel, data, kFold({ k: 3 }), [mse, rmse], { pointwise: true })
 * print('squared error of each row:', cv.pointwise.mse)
 * print('pointwise metrics:', Object.keys(cv.pointwise))
 */
export function crossValidate<
  X extends Features,
  M extends Requirement<X, CapabilityOf<Ms[number]>>,
  const Ms extends readonly ServedMetric[],
>(
  estimator: Fittable<X, M>,
  data: CrossValidationData<X>,
  splitter: Splitter,
  metrics: Ms,
  options: CrossValidateOptions = {},
): CrossValidation<M> {
  const { stream, trainMetrics = false, pointwise = false } = options
  const n = rowCount(data.x)
  const classes = options.classes === 'fold' ? undefined : classOptions(data.y, metrics, options.positive)
  const splits = splitter.split({ n, y: data.y, groups: data.groups }, stream && child(stream, 'split'))
  const rowValues: Record<string, Float64Array>[] = []
  const folds: Fold<M>[] = splits.map((split, f) => {
    const train = takeData(data, split.train.data)
    const test = takeData(data, split.test.data)
    const start = now()
    const model = estimator.fit(train, { stream: stream && child(stream, 'fold', f) })
    const fitMs = now() - start
    const { predictions, values } = evaluateOn(model, test.x, test.y, metrics, classes)
    const fold: Fold<M> = { index: f, train: split.train, test: split.test, model, predictions, metrics: values, fitMs }
    if (trainMetrics) fold.trainMetrics = evaluateOn(model, train.x, train.y, metrics, classes).values
    if (hasTraining(model)) fold.training = model.training
    if (pointwise) rowValues.push(perRow(model, test.x, test.y, predictions, metrics, classes))
    return fold
  })
  const scores: Record<string, Tensor> = {}
  const mean: Record<string, number> = {}
  const std: Record<string, number> = {}
  const directions: Record<string, 'higher' | 'lower'> = {}
  for (const { info } of metrics) {
    const v = Float64Array.from(folds, (fold) => fold.metrics[info.key])
    const mu = v.reduce((a, b) => a + b, 0) / v.length
    scores[info.key] = fromData(v, [v.length])
    mean[info.key] = mu
    std[info.key] = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / (v.length - 1)) : NaN
    directions[info.key] = info.direction
  }
  const outOfFold: Partial<Record<MetricCapability, Tensor>> = {}
  for (const need of new Set(metrics.map((m) => m.info.capability))) {
    if (need === 'predictive') continue
    const out = new Float64Array(n).fill(NaN)
    let ok = true
    for (const fold of folds) {
      const p = fold.predictions[need]
      if (!p || !isTensor(p) || p.shape.length !== 1 || p.shape[0] !== fold.test.shape[0]) {
        ok = false
        break
      }
      fold.test.data.forEach((i, r) => (out[i] = p.data[p.offset + r * (p.strides[0] ?? 1)]))
    }
    if (ok) outOfFold[need] = fromData(out, [n])
  }
  const result: CrossValidation<M> = {
    splitter: splitter.name,
    splits,
    assignment: assignment(splits, n),
    folds,
    scores,
    mean,
    std,
    directions,
    outOfFold,
  }
  if (classes) result.classes = classes
  if (pointwise) {
    const kept = metrics
      .map((m) => m.info.key)
      .filter((key) => folds.every((fold, f) => averagesTo(rowValues[f][key], fold.metrics[key])))
    result.pointwise = {}
    for (const fold of folds) fold.pointwise = {}
    for (const key of kept) {
      const out = new Float64Array(n).fill(NaN)
      folds.forEach((fold, f) => {
        const v = rowValues[f][key]
        fold.pointwise![key] = fromData(v, [v.length])
        fold.test.data.forEach((i, r) => (out[i] = v[r]))
      })
      result.pointwise[key] = fromData(out, [n])
    }
  }
  return result
}
