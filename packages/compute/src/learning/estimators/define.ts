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

/**
 * Attach model metadata to an estimator factory (the factory itself is returned, with a frozen `info` added, its
 * `kind` `'model'` and its `stability` `'experimental'` unless the spec says otherwise). Throws `DomainError` when the
 * factory already has an `info`, so a factory is registered once.
 *
 * @param spec The model's metadata: `key`, `module`, `name`, `task`, `capabilities`, `hyper` and the optional `Info`
 *   fields (`summary`, `notes`, `cite`, ...).
 * @param factory The estimator factory; it is given the `info` property.
 * @returns `factory`, typed as a registry entry.
 *
 * @example Register a mean regressor
 * const meanRegressor = defineModel(
 *   {
 *     key: 'meanRegressor', module: 'example', name: 'Mean regressor',
 *     task: 'regression', capabilities: ['decide'], hyper: { dims: {} },
 *   },
 *   () => ({
 *     name: 'mean',
 *     fit: (d) => {
 *       const m = mean(d.y)
 *       return { decide: (x) => full([x.shape[0]], m) }
 *     },
 *   }),
 * )
 * print('info:', meanRegressor.info)
 * print('registered:', isModelEntry(meanRegressor))
 * print('prediction:', meanRegressor().fit(dataset(tensor([[0], [1]]), tensor([1, 3]))).decide(tensor([[5]])))
 */
export function defineModel<F extends EstimatorFactory>(spec: ModelSpec, factory: F): Entry<F, ModelInfo> {
  return define<F, ModelInfo>({ stability: 'experimental', ...spec, kind: 'model' }, factory)
}

/**
 * True when `x` is an estimator factory registered with `defineModel`: a function whose `info` has `kind: 'model'`.
 *
 * @param x Anything.
 * @returns Whether `x` is a registered model factory.
 *
 * @example A plain factory is not registered
 * const factory = () => ({ name: 'none', fit: () => ({}) })
 * print('before:', isModelEntry(factory))
 * const spec = {
 *   key: 'none', module: 'example', name: 'None', task: 'regression', capabilities: [], hyper: { dims: {} },
 * }
 * print('after:', isModelEntry(defineModel(spec, factory)))
 */
export function isModelEntry(x: unknown): x is ModelEntry {
  return typeof x === 'function' && isEntry<ModelInfo>(x, 'model')
}
