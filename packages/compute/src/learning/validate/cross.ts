/**
 * Cross-validation (plan §5.5; Stone, 1974, "Cross-validatory choice and assessment of statistical predictions",
 * JRSS B 36; Hastie, Tibshirani and Friedman, 2009, "The Elements of Statistical Learning", §7.10): fit on each
 * split's training rows, evaluate on its test rows, keep everything. Metrics are registered metrics (those of
 * `aifn-compute/learning/metrics`, or any function carrying a `MetricInfo`), read by their `info.capability` and keyed
 * by `info.key`. Randomness comes from one stream, split into a child for the splitter and one per fold's fit, so a
 * cross-validation is reproducible from its seed.
 */

import type { MetricCapability } from 'aifn-compute/foundation/contracts'
import {
  hasTraining,
  outputs,
  rowCount,
  score,
  takeData,
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
import { fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
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
}

/** Options of `crossValidate`. */
export interface CrossValidateOptions {
  /** Randomness: the splitter gets `child(stream, 'split')`, fold f's fit `child(stream, 'fold', f)`. */
  stream?: Stream
  /** Also evaluate each fold's metrics on its training rows (default false). */
  trainMetrics?: boolean
}

/**
 * The outputs a model gives on some rows for the metrics, and the metrics' values on them.
 *
 * @param model The fitted model; it must have every capability the metrics read.
 * @param x The inputs of the rows.
 * @param y The targets of the rows.
 * @param metrics The metrics to compute.
 * @returns The outputs by capability (`predictions`), and each metric's value by `info.key` (`values`).
 */
function evaluateOn(
  model: unknown,
  x: unknown,
  y: Tensor,
  metrics: readonly ServedMetric[],
): { predictions: Outputs; values: Record<string, number> } {
  const predictions = outputs(model, metrics, x)
  return { predictions, values: score(metrics, y, predictions) }
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
 * @param options The stream, and whether to score the training rows too.
 * @returns Every fold, the splits and assignment, each metric's scores per fold with their mean, standard deviation
 *   and direction, and the out-of-fold predictions.
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
  const { stream, trainMetrics = false } = options
  const n = rowCount(data.x)
  const splits = splitter.split({ n, y: data.y, groups: data.groups }, stream && child(stream, 'split'))
  const folds: Fold<M>[] = splits.map((split, f) => {
    const train = takeData(data, split.train.data)
    const test = takeData(data, split.test.data)
    const start = now()
    const model = estimator.fit(train, { stream: stream && child(stream, 'fold', f) })
    const fitMs = now() - start
    const { predictions, values } = evaluateOn(model, test.x, test.y, metrics)
    const fold: Fold<M> = { index: f, train: split.train, test: split.test, model, predictions, metrics: values, fitMs }
    if (trainMetrics) fold.trainMetrics = evaluateOn(model, train.x, train.y, metrics).values
    if (hasTraining(model)) fold.training = model.training
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
  return {
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
}
