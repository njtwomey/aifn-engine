/**
 * The registry of every metric in the module, keyed by `info.key`, so that the lab and `estimators.evaluate` can list
 * metrics, filter them by input kind, note or capability, and read their direction and range.
 */

import * as agreement from './agreement'
import * as classification from './classification'
import * as clustering from './clustering'
import { entries } from 'aifn-compute/foundation/registry'
import { type Capability, type InputKind, type Metric } from './core'
import * as curves from './curves'
import * as distances from './distances'
import * as ordinal from './ordinal'
import * as probabilistic from './probabilistic'
import * as ranking from './ranking'
import * as regression from './regression'
import * as representation from './representation'
import { DomainError } from 'aifn-compute/foundation/errors'

const modules = [
  classification,
  ordinal,
  curves,
  probabilistic,
  regression,
  ranking,
  clustering,
  agreement,
  distances,
  representation,
]

/**
 * Every metric, keyed by its `info.key`: those of classification, ordinal, curves, probabilistic, regression, ranking,
 * clustering, agreement, distances and representation.
 *
 * @example Look a metric up by key
 * print('metrics:', Object.keys(metricRegistry).length)
 * print('auroc:', metricRegistry.auroc.info.name, metricRegistry.auroc.info.range)
 */
export const metricRegistry: Readonly<Record<string, Metric>> = entries('metric', ...modules) as unknown as Readonly<
  Record<string, Metric>
>

/**
 * The metrics matching every given filter, in registry order (by module, then definition order). A filter left out
 * matches every metric, so `listMetrics()` lists them all.
 *
 * @param filter What to keep; a metric is kept when it matches every field given. `inputs` keeps the metrics that
 *   read this kind of input (`labels`, `scores`, `probabilities`, ...), `note` those linked to this site note (a slug
 *   such as `precision-recall-and-f-score`), and `capability` those that need this of a model (`decide`, `score` or
 *   `predictive`).
 * @returns The matching metrics.
 *
 * @example The metrics of predicted probabilities
 * print(listMetrics({ inputs: 'probabilities' }).map((m) => m.info.key))
 *
 * @example The metrics of one note
 * print(listMetrics({ note: 'precision-recall-and-f-score' }).map((m) => m.info.key))
 */
export function listMetrics(filter: { inputs?: InputKind; note?: string; capability?: Capability } = {}): Metric[] {
  return Object.values(metricRegistry).filter(
    (m) =>
      (filter.inputs === undefined || m.info.inputs === filter.inputs) &&
      (filter.note === undefined || (m.info.notes ?? []).includes(filter.note)) &&
      (filter.capability === undefined || m.info.capability === filter.capability),
  )
}

/**
 * The metric with this key; throws `DomainError` for an unknown key.
 *
 * @param key The metric's `info.key`, which is its export name (`accuracy`, `logLoss`).
 * @returns The metric, a callable function with its `info`.
 *
 * @example Fetch a metric by name and call it
 * const f1Score = getMetric('f1')
 * print(f1Score.info.name, '=', f1Score([0, 1, 1, 0], [0, 1, 0, 0]))
 *
 * @example An unknown key
 * try {
 *   getMetric('acuracy')
 * } catch (e) {
 *   print(e.name, ':', e.message)
 * }
 */
export function getMetric(key: string): Metric {
  const m = metricRegistry[key]
  if (!m) throw new DomainError('metrics', `metrics: no metric '${key}'`)
  return m
}

/**
 * True when `a` is better than `b` under the metric's direction (NaN is never better). A generic comparison for model
 * selection and for charts that mark the best value.
 *
 * @param metric The metric whose `info.direction` (`higher` or `lower` is better) decides.
 * @param a The candidate value.
 * @param b The value to beat.
 * @returns Whether `a` is strictly better than `b`. A number is better than NaN, and equal values are not better.
 *
 * @example Higher accuracy is better, lower log loss is better
 * print('accuracy 0.9 over 0.8:', isBetter(accuracy, 0.9, 0.8))
 * print('log loss 0.9 over 0.8:', isBetter(logLoss, 0.9, 0.8))
 * print('anything over NaN:', isBetter(logLoss, 5, NaN))
 */
export function isBetter(metric: Metric, a: number, b: number): boolean {
  if (Number.isNaN(a)) return false
  if (Number.isNaN(b)) return true
  return metric.info.direction === 'higher' ? a > b : a < b
}
