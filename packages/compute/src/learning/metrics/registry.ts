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

/** Every metric, keyed by its `info.key`. */
export const metricRegistry: Readonly<Record<string, Metric>> = entries('metric', ...modules) as unknown as Readonly<
  Record<string, Metric>
>

/** The metrics matching every given filter, in registry order (by module, then definition order). */
export function listMetrics(filter: { inputs?: InputKind; note?: string; capability?: Capability } = {}): Metric[] {
  return Object.values(metricRegistry).filter(
    (m) =>
      (filter.inputs === undefined || m.info.inputs === filter.inputs) &&
      (filter.note === undefined || (m.info.notes ?? []).includes(filter.note)) &&
      (filter.capability === undefined || m.info.capability === filter.capability),
  )
}

/** The metric with this key; throws for an unknown key. */
export function getMetric(key: string): Metric {
  const m = metricRegistry[key]
  if (!m) throw new DomainError('metrics', `metrics: no metric '${key}'`)
  return m
}

/**
 * True when `a` is better than `b` under the metric's direction (NaN is never better). A generic comparison for model
 * selection and for charts that mark the best value.
 */
export function isBetter(metric: Metric, a: number, b: number): boolean {
  if (Number.isNaN(a)) return false
  if (Number.isNaN(b)) return true
  return metric.info.direction === 'higher' ? a > b : a < b
}
