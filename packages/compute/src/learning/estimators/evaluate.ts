/**
 * Metrics meet estimators (plan §5.3): `evaluate(model, data, metrics)` reads each metric's registry `MetricInfo` (its
 * `capability` and `inputs`), asks the model for that capability once, and feeds the metric what it reads. The types
 * reject a metric the model cannot serve (a log loss on a model that only decides). Metrics are the registered
 * functions of `aifn-compute/learning/metrics` (or any function carrying a `MetricInfo` with a capability); this module does
 * not import them, since metrics sit above estimators.
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

/** The output of `model` for `capability` on inputs x. */
export function outputFor(model: unknown, capability: MetricCapability, x: unknown): Tensor | Distribution {
  const f = (model as Record<string, ((x: unknown) => Tensor | Distribution) | undefined>)[capability]
  if (typeof f !== 'function') throw new DomainError('evaluate', `evaluate: the model has no ${capability}`)
  return f.call(model, x)
}

/** The outputs of `model` on x for every capability `metrics` read, each computed once. */
export function outputs(model: unknown, metrics: readonly ServedMetric[], x: unknown): Outputs {
  const out: Outputs = {}
  for (const m of metrics) out[m.info.capability] ??= outputFor(model, m.info.capability, x)
  return out
}

/**
 * What a metric receives from an output, by its `info.inputs`: for `probabilities`, the class probabilities of the
 * predictive ([N] P(y = 1) for a Bernoulli, [N, K] otherwise); every other input is the output itself.
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

/** Each metric's value on targets y given precomputed outputs, keyed by `info.key`. */
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
 *
 * @example evaluate(model, test, [accuracy, logLoss]) // { accuracy: 0.93, 'log-loss': 0.21 }
 */
export function evaluate<X, const Ms extends readonly ServedMetric[]>(
  model: Requirement<X, CapabilityOf<Ms[number]>>,
  data: { readonly x: X; readonly y: unknown },
  metrics: Ms,
): Record<string, number> {
  return score(metrics, data.y, outputs(model, metrics, data.x))
}
