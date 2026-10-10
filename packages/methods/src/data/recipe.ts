/**
 * The recipe interpreter (design S §2.8): a recipe is plain data, `{ base, seed, knobs, modifiers: [{ op, params }] }`,
 * and `recipeBook(datasets, modifiers)` replays it through the dataset and modifier registries. Nothing here names a
 * generator: the bases are the registered generators that return a `Dataset`, the knobs and parameters are clamped
 * into each entry's `Space`, and the space of all recipes is derived from the registry, so a dataset picker, URL state
 * and validation need no per-base code.
 *
 * - Knobs and parameters missing from a recipe take their defaults; unknown or inactive ones are dropped and listed in
 *   `meta.ignored` (as `knob`, or `op.param` for a modifier), never used silently. A modifier that needs class labels
 *   is skipped on data without them and listed as `op`.
 * - The base draws from `child(stream(seed), 'base')` and modifier $i$ from `child(stream(seed), 'modifiers', i)`,
 *   so editing one step leaves the draws of the others alone.
 * - `meta.recipe` holds the normalised recipe (every knob and parameter explicit), so `make(d.meta.recipe)` rebuilds
 *   `d`.
 */

import type { Recipe } from 'aifn-compute/foundation/contracts'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  clampReport,
  defaults,
  int,
  space,
  variants,
  type Space,
  type SpaceValue,
  type SpaceValues,
} from 'aifn-compute/foundation/space'
import { generate, modify, type DatasetEntry, type ModifierEntry } from './define'
import type { Dataset } from './types'

/** One modifier step as written: the registered modifier's key and its parameters (defaults when omitted). */
export interface RecipeStepInput {
  /** The registry key of the modifier. */
  readonly op: string
  /** Its parameters, by name; missing ones take their defaults. */
  readonly params?: Readonly<Record<string, unknown>>
}

/** A recipe as written: a base generator's key, and optionally a seed (default 0), knobs and modifier steps. */
export interface RecipeInput {
  /** The registry key of the base generator, one that returns a `Dataset`. */
  readonly base: string
  /** The seed of the root stream the base and the modifiers draw from (default 0). */
  readonly seed?: number | string
  /** The base generator's knobs, by name; missing ones take their defaults. */
  readonly knobs?: Readonly<Record<string, unknown>>
  /** The modifier steps, applied in order (default none). */
  readonly modifiers?: readonly RecipeStepInput[]
}

/** A recipe with every knob and parameter explicit, and what normalising it dropped. */
export interface NormalisedRecipe {
  /** The recipe with the seed, every knob and every modifier parameter explicit. */
  readonly recipe: Recipe
  /** Dropped knobs (`knob`) and modifier parameters (`op.param`): unknown, or inactive under their conditions. */
  readonly ignored: readonly string[]
}

/** The recipe operations over one pair of registries. */
export interface RecipeBook {
  /** The keys of the generators a recipe can start from (those returning a `Dataset`), in registry order. */
  readonly bases: readonly string[]
  /** The keys of the modifiers, in registry order. */
  readonly ops: readonly string[]
  /** The space of a recipe's base and knobs: `base` is a `variants` dimension with each base's knob space. */
  readonly space: Space
  /** Fill defaults, clamp into each space and report what was dropped. Throws on an unknown base or op. */
  normalise(r: RecipeInput): NormalisedRecipe
  /** Build the dataset a recipe describes (the same recipe, after a JSON round trip too, gives the same dataset). */
  make(r: RecipeInput): Dataset
  /** One line: the base with its non-default knobs, each modifier with its parameters, and the seed. */
  describe(r: RecipeInput): string
  /** A compact string for a URL query: URI-encoded JSON with the base's default knobs left out. */
  encode(r: RecipeInput): string
  /** The normalised recipe in a string from `encode`. Throws on anything that is not a recipe. */
  decode(text: string): Recipe
  /** Check that a parsed JSON value is a recipe: known keys, a known base and ops, objects where objects go. */
  parse(value: unknown): RecipeInput
}

/**
 * Whether a value is a plain object (not null and not an array), as recipe fields that hold knobs or parameters must
 * be.
 *
 * @param v Any value, typically parsed from JSON.
 * @returns True for an object that is neither null nor an array.
 */
const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

/**
 * A knob's value as short text for `describe`: numbers to three significant digits, objects as JSON, anything else as
 * a string.
 *
 * @param v The value.
 * @returns Its text.
 */
function fmt(v: SpaceValue): string {
  if (typeof v === 'number') return String(+v.toPrecision(3))
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

/**
 * The recipe operations over a dataset registry and a modifier registry: the bases (the generators whose
 * `info.output` is `'dataset'`), the ops, the space of recipes, and `normalise`, `make`, `describe`, `encode`,
 * `decode` and `parse` (see `RecipeBook`). The recipe space's `base` defaults to `moons` when it is a base. The
 * registries are read when the book is made and when it is used, not copied.
 *
 * @param datasets The dataset generators by key, as `datasetRegistry` holds them.
 * @param modifiers The dataset modifiers by key, as `modifierRegistry` holds them.
 * @returns The recipe operations.
 */
export function recipeBook(
  datasets: Readonly<Record<string, DatasetEntry>>,
  modifiers: Readonly<Record<string, ModifierEntry>>,
): RecipeBook {
  const bases = Object.keys(datasets).filter((k) => datasets[k].info.output === 'dataset')
  const ops = Object.keys(modifiers)

  const baseOf = (key: unknown): DatasetEntry => {
    const entry = typeof key === 'string' ? datasets[key] : undefined
    if (!entry || entry.info.output !== 'dataset') throw new TypeError(`recipe: unknown base ${String(key)}`)
    return entry
  }
  const opOf = (key: unknown): ModifierEntry => {
    const entry = typeof key === 'string' ? modifiers[key] : undefined
    if (!entry) throw new TypeError(`recipe: unknown modifier ${String(key)}`)
    return entry
  }

  function normalise(r: RecipeInput): NormalisedRecipe {
    const entry = baseOf(r.base)
    const knobs = clampReport(entry.info.knobs, r.knobs ?? {})
    const ignored = [...knobs.dropped]
    const steps = (r.modifiers ?? []).map(({ op, params }) => {
      const m = opOf(op)
      const p = clampReport(m.info.params, params ?? {})
      ignored.push(...p.dropped.map((k) => `${op}.${k}`))
      return { op, params: p.values }
    })
    return { recipe: { base: r.base, seed: r.seed ?? 0, knobs: knobs.values, modifiers: steps }, ignored }
  }

  function make(r: RecipeInput): Dataset {
    const { recipe, ignored } = normalise(r)
    const root = stream(recipe.seed)
    const skipped: string[] = []
    let d = generate(datasets[recipe.base], child(root, 'base'), recipe.knobs) as Dataset
    recipe.modifiers.forEach(({ op, params }, i) => {
      const m = modifiers[op]
      if (m.info.needs === 'labels' && d.y?.dtype !== 'int32') skipped.push(op)
      else d = modify(m, child(root, 'modifiers', i), d, params)
    })
    const all = [...ignored, ...skipped]
    return { ...d, meta: { ...d.meta, recipe, ...(all.length ? { ignored: all } : {}) } }
  }

  function describe(r: RecipeInput): string {
    const { recipe } = normalise(r)
    const entry = datasets[recipe.base]
    const initial = defaults(entry.info.knobs)
    const knobs = Object.entries(recipe.knobs as SpaceValues)
      .filter(([k, v]) => JSON.stringify(v) !== JSON.stringify(initial[k]))
      .map(([k, v]) => `${k} ${fmt(v)}`)
    const parts = [`${entry.info.name}${knobs.length ? ` (${knobs.join(', ')})` : ''}`]
    for (const { op, params } of recipe.modifiers) {
      const shown = Object.entries(params as SpaceValues).map(([k, v]) => `${k} ${fmt(v)}`)
      parts.push(`${modifiers[op].info.name.toLowerCase()}${shown.length ? ` (${shown.join(', ')})` : ''}`)
    }
    parts.push(`seed ${recipe.seed}`)
    return parts.join(', ')
  }

  function encode(r: RecipeInput): string {
    const { recipe } = normalise(r)
    const initial = defaults(datasets[recipe.base].info.knobs)
    const knobs = Object.fromEntries(
      Object.entries(recipe.knobs as SpaceValues).filter(([k, v]) => JSON.stringify(v) !== JSON.stringify(initial[k])),
    )
    return encodeURIComponent(JSON.stringify({ ...recipe, knobs }))
  }

  function parse(value: unknown): RecipeInput {
    if (!isRecord(value)) throw new TypeError('recipe: not an object')
    for (const key of Object.keys(value))
      if (!['base', 'seed', 'knobs', 'modifiers'].includes(key)) throw new TypeError(`recipe: unknown key ${key}`)
    baseOf(value.base)
    if (value.seed !== undefined && typeof value.seed !== 'number' && typeof value.seed !== 'string')
      throw new TypeError('recipe: seed must be a number or a string')
    if (value.knobs !== undefined && !isRecord(value.knobs)) throw new TypeError('recipe: knobs must be an object')
    if (value.modifiers !== undefined) {
      if (!Array.isArray(value.modifiers)) throw new TypeError('recipe: modifiers must be a list')
      for (const step of value.modifiers as unknown[]) {
        if (!isRecord(step)) throw new TypeError('recipe: a modifier step must be an object')
        opOf(step.op)
        if (step.params !== undefined && !isRecord(step.params))
          throw new TypeError(`recipe: the params of ${String(step.op)} must be an object`)
      }
    }
    return value as unknown as RecipeInput
  }

  const cases = Object.fromEntries(bases.map((k) => [k, datasets[k].info.knobs]))
  const recipeSpace = space({
    base: variants(cases, bases.includes('moons') ? { default: 'moons' } : {}),
    seed: int(0, 9999, { default: 0 }),
  })

  return {
    bases,
    ops,
    space: recipeSpace,
    normalise,
    make,
    describe,
    encode,
    decode: (text) => normalise(parse(JSON.parse(decodeURIComponent(text)))).recipe,
    parse,
  }
}
