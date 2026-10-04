/**
 * Cross-validation (plan §5.5; Stone, 1974, "Cross-validatory choice and assessment of statistical predictions",
 * JRSS B 36; Hastie, Tibshirani and Friedman, 2009, "The Elements of Statistical Learning", §7.10): fit on each
 * split's training rows, evaluate on its test rows, keep everything. Metrics are registered metrics, read by their
 * `info.capability` and keyed by `info.key`.
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

/** A dataset for cross-validation: inputs, targets and optional group labels. */
export type CrossValidationData<X extends Features> = Supervised<X, Tensor> & { readonly groups?: Column }

/** Something with `fit` on supervised data. */
export type Fittable<X extends Features, M> = {
  readonly name: string
  fit(data: Supervised<X, Tensor>, options?: FitOptions): M
}

/** One fold of a cross-validation. */
export interface Fold<M> {
  index: number
  /** Sorted row indices of the training and test sets. */
  train: Tensor
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
  splitter: string
  splits: Split[]
  /** Fold assignment [folds, n]: 1 test, 0 train, −1 unused (see `assignment`). */
  assignment: Tensor
  folds: Fold<M>[]
  /** Each metric over folds, [folds]. */
  scores: Record<string, Tensor>
  mean: Record<string, number>
  /** Sample standard deviation over folds (÷ (folds − 1)). */
  std: Record<string, number>
  /** Each metric's direction, by `info.key`. */
  directions: Record<string, 'higher' | 'lower'>
  /**
   * Out-of-fold predictions [n] per tensor-valued capability ('decide', 'expect'; scores of shape [n]): each row's
   * prediction from the model that did not train on it; NaN for rows never tested. With repeated splits the last wins.
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
