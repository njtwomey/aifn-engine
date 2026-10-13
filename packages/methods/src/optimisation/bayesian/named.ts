/**
 * Objectives named by their registry key, so that a whole Bayesian optimisation run is plain data: a name such as
 * `'forrester'`, `'branin'` or `'svm-tuning-surface'`, or an object with the name and the objective's parameters
 * (`{ name: 'svm-tuning-surface', noise: 0.02 }`), resolves to the registered test function of
 * `aifn-methods/data/objectives`, whose box and known minimum then come with it. Kebab-case names are read as the
 * camel-case keys (`svm-tuning-surface` is `svmTuningSurface`).
 */

import { entries, type Entry } from 'aifn-compute/foundation/registry'
import type { ObjectiveInfo } from 'aifn-compute/foundation/contracts'
import type { Vector } from 'aifn-compute/foundation/contracts'
import type { Stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'
import * as objectives from 'aifn-methods/data/objectives'
import type { TestFunction } from 'aifn-methods/data/objectives'

/**
 * A registered objective by name: its registry key (`'forrester'`, `'branin'`, `'svmTuningSurface'`) or the key in
 * kebab case (`'svm-tuning-surface'`).
 */
export type ObjectiveName = 'forrester' | 'branin' | 'svmTuningSurface' | 'svm-tuning-surface' | (string & {})

/**
 * A registered objective with its parameters: a name, or an object holding the `name` and the parameters its factory
 * takes (`noise` for the SVM tuning surface, `fidelity` for Forrester).
 */
export type NamedObjective = ObjectiveName | { readonly name: ObjectiveName; readonly [option: string]: unknown }

/** A test function made from a named objective, with its value given a stream and its noise-free value. */
export type ResolvedObjective = {
  /** The registry key it resolved to. */
  key: string
  /** The test function: its box (`lo`, `hi`), minimisers and minimum value. */
  fn: TestFunction
  /** Its value at a point, with a stream for any evaluation noise. */
  value: (x: Vector, s: Stream) => number
  /** Its value without evaluation noise (the same as `value` for a deterministic function). */
  clean: (x: Vector) => number
}

/** A registered objective factory, called with its parameters. */
type Factory = Entry<(params?: Record<string, unknown>) => TestFunction, ObjectiveInfo>

/**
 * True when `x` names a registered objective (a string, or an object with a string `name`) rather than being a
 * function.
 *
 * @param x The objective as given.
 * @returns Whether it is a `NamedObjective`.
 */
export function isNamedObjective(x: unknown): x is NamedObjective {
  return (
    typeof x === 'string' || (typeof x === 'object' && x !== null && typeof (x as { name?: unknown }).name === 'string')
  )
}

/**
 * Resolve a named objective to its registered test function, made with the parameters given. Throws `DomainError`
 * for a name that is not registered, listing those that are.
 *
 * @param spec The name, or the name with the factory's parameters.
 * @param where The caller's name, for error messages.
 * @returns The key, the test function, its value with a stream and its noise-free value.
 */
export function resolveObjective(spec: NamedObjective, where: string): ResolvedObjective {
  const table = entries<ObjectiveInfo>('objective', objectives) as Readonly<Record<string, Factory>>
  const { name, ...params } = typeof spec === 'string' ? { name: spec } : spec
  const key = name in table ? name : name.replace(/-([a-z0-9])/g, (_, c: string) => c.toUpperCase())
  const factory = table[key]
  if (!factory)
    throw new DomainError(
      where,
      `${where}: no registered objective '${name}' (known: ${Object.keys(table).sort().join(', ')})`,
    )
  const fn = factory(params)
  const withStream = fn.value as (x: Vector, s?: Stream) => number
  const mean = (fn as { mean?: (x: Vector) => number }).mean
  return { key, fn, value: (x, s) => withStream(x, s), clean: mean ?? ((x) => fn.value(x)) }
}
