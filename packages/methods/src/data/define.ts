/**
 * The registered things of `aifn-methods/data` (design S §2.8, §3.1): dataset generators, dataset modifiers, test
 * objectives and test log densities, each defined where it is written (with `aifn-compute/foundation/registry`'s
 * `definer`) and collected by `data/index.ts`. `generate` and `modify` call an entry with or without a stream, as its
 * `info.random` says, so a caller (the recipe interpreter, a dataset picker) needs no per-entry code.
 */

import type {
  DatasetInfo,
  LogDensityInfo,
  ModifierInfo,
  ObjectiveInfo,
  Stream,
} from 'aifn-compute/foundation/contracts'
import type { Entry } from 'aifn-compute/foundation/registry'
import type { Dataset } from './types'

/** Any function: registered values are called through `generate` and `modify`, which read `info.random`. */
type Callable = (...args: never[]) => unknown

/** A registered dataset generator: `(s, knobs)` when `info.random`, else `(knobs)`. */
export type DatasetEntry = Entry<Callable, DatasetInfo>

/** A registered modifier: `(s, dataset, params)` when `info.random`, else `(dataset, params)`. */
export type ModifierEntry = Entry<Callable, ModifierInfo>

/** A registered test objective factory, called `(params)`. */
export type ObjectiveEntry = Entry<Callable, ObjectiveInfo>

/** A registered test log-density factory, called `(params)`. */
export type LogDensityEntry = Entry<Callable, LogDensityInfo>

/**
 * Call a generator with its knobs, and with the stream first when it draws (`entry.info.random`). The knobs are passed
 * as given: neither filled with defaults nor clamped into `entry.info.knobs` (the recipe interpreter does that).
 *
 * @param entry The registered generator.
 * @param s The stream to draw from; not passed to a deterministic generator.
 * @param knobs The generator's options, by name.
 * @returns What the generator returns: a `Dataset`, or another output such as a table (see `entry.info.output`).
 *
 * @example Six points of the two moons, through the registry
 * const d = generate(datasetRegistry.moons, stream(0), { n: 6 })
 * print(d.meta.name, ' x:', d.x.shape, ' labels:', d.y)
 */
export function generate(entry: DatasetEntry, s: Stream, knobs: Readonly<Record<string, unknown>>): unknown {
  const f = entry as unknown as (...args: unknown[]) => unknown
  return entry.info.random ? f(s, knobs) : f(knobs)
}

/**
 * Apply a modifier with its parameters, and with the stream first when it draws (`entry.info.random`). The parameters
 * are passed as given, and a modifier that needs class labels is not skipped here (the recipe interpreter does that).
 *
 * @param entry The registered modifier.
 * @param s The stream to draw from; not passed to a deterministic modifier.
 * @param d The dataset the modifier is applied to.
 * @param params The modifier's parameters, by name.
 * @returns The modified dataset.
 *
 * @example Label noise at rate 0.2, through the registry
 * const d = generate(datasetRegistry.moons, stream(0), { n: 200 })
 * const noisy = modify(modifierRegistry.withLabelNoise, stream(1), d, { rate: 0.2 })
 * const y = toArray(d.y)
 * print('labels flipped:', toArray(noisy.y).filter((v, i) => v !== y[i]).length, 'of', y.length)
 */
export function modify(
  entry: ModifierEntry,
  s: Stream,
  d: Dataset,
  params: Readonly<Record<string, unknown>>,
): Dataset {
  const f = entry as unknown as (...args: unknown[]) => Dataset
  return entry.info.random ? f(s, d, params) : f(d, params)
}
