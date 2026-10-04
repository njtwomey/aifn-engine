/**
 * Registering estimators (design S §2.7, §3.1): `defineModel(spec, factory)` attaches a `ModelInfo` to an estimator
 * factory (a function returning an `Estimator`), so a registry can list every model by task or capability and the
 * catalog can say "every model that gives a predictive distribution" without fitting anything. The declared
 * capabilities are checked by a protocol test that fits each registered estimator on tiny data.
 */

import { define, isEntry } from 'aifn-compute/foundation/registry'
import type { Entry, ModelInfo, Stability } from 'aifn-compute/foundation/contracts'
import type { Estimator } from './data'

/** An estimator factory: hyperparameters (and possibly a stream or sub-estimators) in, an `Estimator` out. */
export type EstimatorFactory = (...args: never[]) => Estimator<never, unknown>

/** A registered estimator factory. */
export type ModelEntry<F extends EstimatorFactory = EstimatorFactory> = Entry<F, ModelInfo>

/**
 * What a model definition states: its `ModelInfo` without the fields `defineModel` fills (`kind`; `stability`,
 * default `experimental`).
 */
export type ModelSpec = Omit<ModelInfo, 'kind' | 'stability'> & { readonly stability?: Stability }

/** Attach model metadata to an estimator factory (the factory itself is returned, with `info` added). */
export function defineModel<F extends EstimatorFactory>(spec: ModelSpec, factory: F): Entry<F, ModelInfo> {
  return define<F, ModelInfo>({ stability: 'experimental', ...spec, kind: 'model' }, factory)
}

/** True when `x` is an estimator factory registered with `defineModel`. */
export function isModelEntry(x: unknown): x is ModelEntry {
  return typeof x === 'function' && isEntry<ModelInfo>(x, 'model')
}
