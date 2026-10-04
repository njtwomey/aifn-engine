/**
 * The registered things of `aifn-methods/data` (design S §2.8, §3.1): dataset generators, dataset modifiers,
 * test objectives and test log densities, each defined where it is written (with
 * `aifn-compute/foundation/registry`'s `definer`) and collected by `data/index.ts`. `generate` and `modify` call an entry with
 * or without a stream, as its `info.random` says.
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

/** Call a generator with its knobs (and the stream, when it draws). */
export function generate(entry: DatasetEntry, s: Stream, knobs: Readonly<Record<string, unknown>>): unknown {
  const f = entry as unknown as (...args: unknown[]) => unknown
  return entry.info.random ? f(s, knobs) : f(knobs)
}

/** Apply a modifier with its parameters (and the stream, when it draws). */
export function modify(
  entry: ModifierEntry,
  s: Stream,
  d: Dataset,
  params: Readonly<Record<string, unknown>>,
): Dataset {
  const f = entry as unknown as (...args: unknown[]) => Dataset
  return entry.info.random ? f(s, d, params) : f(d, params)
}
