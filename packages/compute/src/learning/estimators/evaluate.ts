/**
 * Metrics meet estimators (plan §5.3): `evaluate(model, data, metrics)` reads each metric's registry `MetricInfo` (its
 * `capability` and `inputs`), asks the model for that capability once, and feeds the metric what it reads. The types
 * reject a metric the model cannot serve (a log loss on a model that only decides). Metrics are the registered
 * functions of `aifn-compute/learning/metrics` (or any function carrying a `MetricInfo` with a capability); this module
 * does not import them, since metrics sit above estimators.
 *
 * The capability split follows scikit-learn's scorers (`predict`, `decision_function`, `predict_proba`; Buitinck et
 * al., 2013, "API design for machine learning software: experiences from the scikit-learn project").
 */

import type {
  Decides,
  Distribution,
  MetricCapability,
  MetricInfo,
  Predicts,
  Scalar,
  Scores,
} from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { classProbabilities } from './distribution'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A metric `evaluate` can serve: a function of the targets and one model output, carrying registry metadata that
 * declares the capability it reads.
 */
export type ServedMetric<C extends MetricCapability = MetricCapability> = ((
  yTrue: never,
  prediction: never,
) => Scalar) & {
  readonly info: MetricInfo & { readonly capability: C }
}

/** The capability a metric (or a list of metrics) reads. */
export type CapabilityOf<M> = M extends { readonly info: { readonly capability: infer C } } ? C : never

/** What a model must have to serve metrics that read capabilities `C` on inputs `X`. */
export type Requirement<X, C> = ('decide' extends C ? Decides<X, Tensor> : unknown) &
  ('score' extends C ? Scores<X> : unknown) &
  ('predictive' extends C ? Predicts<X, Distribution> : unknown)

/** A model output by capability: decisions and scores are tensors, a predictive is a distribution. */
export type Outputs = Partial<Record<MetricCapability, Tensor | Distribution>>

/**
 * The output of `model` for `capability` on inputs `x`: the model's method of that name, called on `x`. Throws
 * `DomainError` when the model has no such method.
 *
 * @param model The fitted model.
 * @param capability The capability a metric reads: `'decide'`, `'score'` or `'predictive'`.
 * @param x The inputs.
 * @returns A tensor (decisions, scores) or a distribution (a predictive).
 *
 * @example Decisions, and a capability the model lacks
 * // A model whose inputs are already probabilities of class 1, deciding at 0.5.
 * const model = withDecision({ predictive: (x) => bernoulliPredictive(x) }, { threshold: 0.5 })
 * print('decisions:', outputFor(model, 'decide', tensor([0.9, 0.2, 0.6])))
 * try {
 *   outputFor(model, 'score', tensor([0.9]))
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
export function outputFor(model: unknown, capability: MetricCapability, x: unknown): Tensor | Distribution {
  const f = (model as Record<string, ((x: unknown) => Tensor | Distribution) | undefined>)[capability]
  if (typeof f !== 'function') throw new DomainError('evaluate', `evaluate: the model has no ${capability}`)
  return f.call(model, x)
}

/**
 * The outputs of `model` on `x` for every capability `metrics` read, each computed once however many metrics read it.
 *
 * @param model The fitted model; it must have every capability the metrics read.
 * @param metrics The metrics, each carrying `info.capability`.
 * @param x The inputs.
 * @returns The outputs, keyed by capability.
 *
 * @example Two metrics, two capabilities
 * // A model whose inputs are already probabilities of class 1, deciding at 0.5.
 * const model = withDecision({ predictive: (x) => bernoulliPredictive(x) }, { threshold: 0.5 })
 * const errorRate = Object.assign((y, d) => mean(abs(sub(y, d))), {
 *   info: { key: 'error-rate', capability: 'decide', direction: 'lower' },
 * })
 * const logLoss = Object.assign((y, p) => -mean(add(mul(y, log(p)), mul(sub(1, y), log(sub(1, p))))), {
 *   info: { key: 'log-loss', capability: 'predictive', inputs: 'probabilities', direction: 'lower' },
 * })
 * const out = outputs(model, [errorRate, logLoss], tensor([0.9, 0.2, 0.6]))
 * print('capabilities:', Object.keys(out))
 * print('decisions:', out.decide)
 * print('predictive mean:', out.predictive.mean())
 */
export function outputs(model: unknown, metrics: readonly ServedMetric[], x: unknown): Outputs {
  const out: Outputs = {}
  for (const m of metrics) out[m.info.capability] ??= outputFor(model, m.info.capability, x)
  return out
}

/**
 * What a metric receives from an output, by its `info.inputs`: for `probabilities`, the class probabilities of the
 * predictive ($N$ values $\Pr(y = 1)$ for a Bernoulli, an $N \times K$ matrix otherwise); every other input is the
 * output itself.
 *
 * @param metric The metric, whose `info.inputs` says what it reads.
 * @param output The model's output for the metric's capability.
 * @returns The class probabilities for a probability metric given a distribution, otherwise `output` unchanged.
 *
 * @example A Bernoulli predictive as the probabilities of class 1
 * const logLoss = Object.assign((y, p) => -mean(add(mul(y, log(p)), mul(sub(1, y), log(sub(1, p))))), {
 *   info: { key: 'log-loss', capability: 'predictive', inputs: 'probabilities', direction: 'lower' },
 * })
 * const predictive = bernoulliPredictive(tensor([0.9, 0.2]))
 * print('log loss reads:', metricInput(logLoss, predictive))
 * print('categorical:', metricInput(logLoss, categoricalPredictive(tensor([[0.2, 0.5, 0.3]]))))
 */
export function metricInput(metric: ServedMetric, output: Tensor | Distribution): Tensor | Distribution {
  if (metric.info.inputs !== 'probabilities' || !('kind' in output) || output.kind !== 'distribution') return output
  const p = classProbabilities(output)
  if (output.name !== 'Bernoulli') return p
  const probs = dense.data(p)
  return fromData(
    Float64Array.from({ length: p.shape[0] }, (_, i) => probs[2 * i + 1]),
    [p.shape[0]],
  )
}

/**
 * Each metric's value on targets `y` given precomputed outputs, keyed by `info.key`. Throws `DomainError` when an
 * output a metric reads is missing.
 *
 * @param metrics The metrics to compute.
 * @param y The targets, as the metrics take them.
 * @param out The model's outputs by capability, as `outputs` returns them.
 * @returns Each metric's value, by `info.key`.
 *
 * @example Score outputs computed once
 * // A model whose inputs are already probabilities of class 1, deciding at 0.5.
 * const model = withDecision({ predictive: (x) => bernoulliPredictive(x) }, { threshold: 0.5 })
 * const errorRate = Object.assign((y, d) => mean(abs(sub(y, d))), {
 *   info: { key: 'error-rate', capability: 'decide', direction: 'lower' },
 * })
 * const out = outputs(model, [errorRate], tensor([0.9, 0.2, 0.6, 0.4]))
 * print(score([errorRate], tensor([1, 0, 0, 0]), out))
 */
export function score(metrics: readonly ServedMetric[], y: unknown, out: Outputs): Record<string, number> {
  const result: Record<string, number> = {}
  for (const m of metrics) {
    const output = out[m.info.capability]
    if (output === undefined)
      throw new DomainError('evaluate', `evaluate: no ${m.info.capability} output for ${m.info.key}`)
    result[m.info.key] = (m as unknown as (y: unknown, p: unknown) => number)(y, metricInput(m, output))
  }
  return result
}

/**
 * Evaluate a fitted model on data: each metric's value, keyed by its `info.key`. The model must have every capability
 * the metrics read (checked by the compiler from the metrics' `info.capability`); each capability is called once.
 * Throws `DomainError` when the model lacks a capability at run time.
 *
 * @param model The fitted model.
 * @param data The inputs `x` and targets `y` to evaluate on (a dataset, or any object with both).
 * @param metrics The metrics: functions of the targets and one model output carrying a `MetricInfo` (registered
 *   metrics of `aifn-compute/learning/metrics`, or hand-made ones as below).
 * @returns Each metric's value, by `info.key`.
 *
 * @example An error rate and a log loss, checked by hand
 * // A model whose inputs are already probabilities of class 1, deciding at 0.5. One decision of four is wrong, and
 * // the log loss is -(log 0.9 + log 0.8 + log 0.4 + log 0.6) / 4.
 * const model = withDecision({ predictive: (x) => bernoulliPredictive(x) }, { threshold: 0.5 })
 * const errorRate = Object.assign((y, d) => mean(abs(sub(y, d))), {
 *   info: { key: 'error-rate', capability: 'decide', direction: 'lower' },
 * })
 * const logLoss = Object.assign((y, p) => -mean(add(mul(y, log(p)), mul(sub(1, y), log(sub(1, p))))), {
 *   info: { key: 'log-loss', capability: 'predictive', inputs: 'probabilities', direction: 'lower' },
 * })
 * const data = dataset(tensor([0.9, 0.2, 0.6, 0.4]), tensor([1, 0, 0, 0]))
 * print(evaluate(model, data, [errorRate, logLoss]))
 * print('by hand:', -(Math.log(0.9) + Math.log(0.8) + Math.log(0.4) + Math.log(0.6)) / 4)
 */
export function evaluate<X, const Ms extends readonly ServedMetric[]>(
  model: Requirement<X, CapabilityOf<Ms[number]>>,
  data: { readonly x: X; readonly y: unknown },
  metrics: Ms,
): Record<string, number> {
  return score(metrics, data.y, outputs(model, metrics, data.x))
}
