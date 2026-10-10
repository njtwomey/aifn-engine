/**
 * The description language of subgroup discovery over a table: a selector tests one attribute (nominal `=` or `≠` a
 * level, numeric `≥` or `≤` a cut point) and a description is a conjunction of selectors. The cover of a description
 * is the set of rows satisfying every selector, a bitset, so a refinement's cover is its parent's cover AND one
 * selector's cover.
 *
 * Numeric attributes are discretised by cut points: `equal-frequency` (cuts at the $1/b, \dots, (b - 1)/b$ quantiles
 * of the column's values, as data values), `equal-width` ($b$ equal intervals of its range), or `on-the-fly`, where
 * the cuts are the equal-frequency cuts of the values inside the description being refined, so a deeper description
 * gets cuts adapted to its own rows (Grosskreutz and Rüping, 2009, "On subgroup discovery in numerical domains").
 *
 * Refinement is canonical: selectors are ordered by attribute and operator, and a description is extended only by
 * selectors after its last one, so every conjunction is generated once. A description holds at most one selector per
 * attribute and operator, and an attribute tested with `=` is not tested again; `≥` then `≤` on one numeric attribute
 * gives an interval.
 */

import type { Column, Table } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { isTensor, toFlat } from 'aifn-compute/foundation/tensor'
import { bitset, bitsetAnd, bitsetFull, bitsetIndices, type Bitset } from './cover'

/** A selector's test: nominal `=` and `≠`, numeric `≥` and `≤`. */
export type SelectorOp = '=' | '≠' | '≥' | '≤'

/** One test on one attribute. Plain data. */
export interface Selector {
  /** The column tested. */
  readonly attribute: string
  /** The test: `=` or `≠` a level of a nominal attribute, `≥` or `≤` a cut point of a numeric one. */
  readonly op: SelectorOp
  /** The level or cut point compared with. */
  readonly value: string | number
}

/** A conjunction of selectors in canonical order; the empty description covers every row. */
export type Description = readonly Selector[]

/** How numeric attributes get their cut points. */
export type Discretisation = 'equal-frequency' | 'equal-width' | 'on-the-fly'

/** Options of `selectorLanguage`. */
export interface LanguageOptions {
  /** Columns that are not attributes (the targets). */
  exclude?: readonly string[]
  /** Numeric columns to treat as nominal (each distinct value a level). */
  nominal?: readonly string[]
  /** Default `equal-frequency`. */
  discretisation?: Discretisation
  /** Intervals per numeric attribute, $b$: at most $b - 1$ cut points (default 4). */
  bins?: number
  /** Also generate `≠` selectors for nominal attributes (default false). */
  negations?: boolean
}

/** An attribute of the language. */
export interface LanguageAttribute {
  /** The column's name in the table. */
  readonly name: string
  /** Nominal (tested with `=` and `≠`) or numeric (tested with `≥` and `≤`). */
  readonly kind: 'nominal' | 'numeric'
  /** Nominal: the levels, sorted. */
  readonly levels: readonly (string | number)[]
  /** Numeric: the fixed cut points, ascending (empty with `on-the-fly`). */
  readonly cuts: readonly number[]
}

/** A description language over a table, with memoised covers. */
export interface SelectorLanguage {
  /** The number of rows of the table. */
  readonly rows: number
  /** The attributes, in the order of the table's columns: the canonical order of selectors. */
  readonly attributes: readonly LanguageAttribute[]
  /** How numeric attributes get their cut points. */
  readonly discretisation: Discretisation
  /** The number of intervals per numeric attribute. */
  readonly bins: number
  /** Whether nominal attributes also get `≠` selectors. */
  readonly negations: boolean
  /** The single-selector descriptions' selectors (the root's refinements), in canonical order. */
  readonly selectors: readonly Selector[]
  /** The rows a selector covers. */
  selectorCover(s: Selector): Bitset
  /** The rows a description covers. */
  cover(d: Description): Bitset
  /** The selectors that may extend `d`, in canonical order. */
  extensions(d: Description): Selector[]
  /** The refinements of `d`: `d` extended by each of its extensions. */
  refinements(d: Description): Description[]
  /** `d`'s selectors in canonical order (a description built by hand, e.g. a selector added interactively). */
  canonical(d: Description): Description
}

const OPS: readonly SelectorOp[] = ['=', '≠', '≥', '≤']

/**
 * The canonical key of a selector, with its value written exactly: two selectors are the same test when their keys
 * are equal.
 *
 * @param s The selector.
 * @returns The attribute, the operator and the value, separated by spaces.
 *
 * @example A nominal and a numeric selector
 * print(selectorKey({ attribute: 'colour', op: '=', value: 'red' }))
 * print(selectorKey({ attribute: 'size', op: '≥', value: 3.14159 }))
 */
export function selectorKey(s: Selector): string {
  return `${s.attribute} ${s.op} ${s.value}`
}

/**
 * The canonical key of a description: its selectors' keys joined by `∧` (`∅` for the empty description). Keys are
 * equal only for selectors in the same order, so a description built by hand is put in canonical order first
 * (`language.canonical`).
 *
 * @param d The description.
 * @returns The key, as used to memoise covers and to tell search nodes apart.
 *
 * @example A conjunction and the empty description
 * const d = [
 *   { attribute: 'colour', op: '=', value: 'red' },
 *   { attribute: 'size', op: '≥', value: 3.14159 },
 * ]
 * print(descriptionKey(d))
 * print(descriptionKey([]))
 */
export function descriptionKey(d: Description): string {
  return d.length ? d.map(selectorKey).join(' ∧ ') : '∅'
}

/**
 * A description for display: numeric cut points to `digits` significant digits; `everything` for the empty one.
 *
 * @param d The description.
 * @param digits The significant digits of numeric values.
 * @returns The selectors joined by `∧`.
 *
 * @example Cut points rounded for display
 * const d = [
 *   { attribute: 'colour', op: '=', value: 'red' },
 *   { attribute: 'size', op: '≥', value: 3.14159 },
 * ]
 * print(formatDescription(d))
 * print(formatDescription(d, 1))
 * print(formatDescription([]))
 */
export function formatDescription(d: Description, digits = 3): string {
  if (!d.length) return 'everything'
  return d
    .map((s) => `${s.attribute} ${s.op} ${typeof s.value === 'number' ? Number(s.value.toPrecision(digits)) : s.value}`)
    .join(' ∧ ')
}

/**
 * True when `s` may join `d` (at most one selector per attribute and operator; nothing beside an `=`). It does not
 * check that a `≥` and a `≤` on one attribute leave a non-empty interval.
 *
 * @param d The description to extend.
 * @param s The selector to add.
 * @returns Whether the conjunction is in the language.
 *
 * @example An interval is allowed, a second lower bound or a test beside an equality is not
 * const d = [{ attribute: 'size', op: '≥', value: 3 }]
 * print('size <= 5:', compatibleSelector(d, { attribute: 'size', op: '≤', value: 5 }))
 * print('size >= 4:', compatibleSelector(d, { attribute: 'size', op: '≥', value: 4 }))
 * const red = [{ attribute: 'colour', op: '=', value: 'red' }]
 * print('colour != blue:', compatibleSelector(red, { attribute: 'colour', op: '≠', value: 'blue' }))
 */
export function compatibleSelector(d: Description, s: Selector): boolean {
  for (const p of d) if (p.attribute === s.attribute && (p.op === s.op || p.op === '=' || s.op === '=')) return false
  return true
}

/**
 * A column's values as an array: a tensor flattened, an array as it is.
 *
 * @param c The column.
 * @returns Its values in row order.
 */
const values = (c: Column): readonly (string | number)[] => (isTensor(c) ? toFlat(c) : c)

/**
 * Equal-frequency cut points of finite values (data values; neither the minimum nor the maximum): the value at
 * position $\lfloor jm/b \rfloor$ for $j = 1, \dots, b - 1$, with repeats dropped.
 *
 * @param sortedValues The $m$ values, finite and in ascending order.
 * @param bins The number of intervals $b$.
 * @returns The distinct cut points, ascending (none for no values).
 */
function equalFrequency(sortedValues: ArrayLike<number>, bins: number): number[] {
  const m = sortedValues.length
  if (m === 0) return []
  const out: number[] = []
  for (let j = 1; j < bins; j++) {
    const c = sortedValues[Math.min(m - 1, Math.floor((j * m) / bins))]
    if (c > sortedValues[0] && c < sortedValues[m - 1] && c !== out[out.length - 1]) out.push(c)
  }
  return out
}

/**
 * Equal-width cut points: $b - 1$ points splitting the range of the values into $b$ equal intervals.
 *
 * @param sortedValues The values, finite and in ascending order.
 * @param bins The number of intervals $b$.
 * @returns The cut points, ascending (none when there are no values or they are all equal).
 */
function equalWidth(sortedValues: ArrayLike<number>, bins: number): number[] {
  const m = sortedValues.length
  if (m === 0) return []
  const lo = sortedValues[0]
  const hi = sortedValues[m - 1]
  if (!(hi > lo)) return []
  return Array.from({ length: bins - 1 }, (_, j) => lo + ((j + 1) * (hi - lo)) / bins)
}

/**
 * Cut points of `xs` by a fixed method (values that are not finite are ignored). Throws `DomainError` unless `bins` is
 * an integer of at least 1.
 *
 * @param xs The values of a numeric column, in any order.
 * @param bins The number of intervals $b$: at most $b - 1$ cut points.
 * @param method `equal-frequency` (data values at the $j/b$ quantiles, neither the minimum nor the maximum, repeats
 *   dropped) or `equal-width` (equal intervals of the range).
 * @returns The cut points, ascending.
 *
 * @example Four intervals of 1 to 8, by either method
 * const xs = [8, 1, 7, 2, 6, 3, 5, 4]
 * print('equal-frequency:', cutPoints(xs, 4, 'equal-frequency'))
 * print('equal-width:', cutPoints(xs, 4, 'equal-width'))
 */
export function cutPoints(xs: ArrayLike<number>, bins: number, method: 'equal-frequency' | 'equal-width'): number[] {
  if (!(Number.isInteger(bins) && bins >= 1))
    throw new DomainError('cutPoints', 'cutPoints: bins must be an integer ≥ 1')
  const sorted = Float64Array.from(Array.from(xs).filter((v) => Number.isFinite(v))).sort()
  return method === 'equal-width' ? equalWidth(sorted, bins) : equalFrequency(sorted, bins)
}

/**
 * The description language of a table: every column not excluded is an attribute (a tensor or a numeric array is
 * numeric unless listed in `nominal`; any other array is nominal, its distinct values the levels). Throws `DomainError`
 * when no column is left or the columns differ in length; asking for the cover of a selector on an unknown attribute
 * throws too.
 *
 * @param table The table, a column per name; the target columns go in `options.exclude`.
 * @param options The columns to exclude or treat as nominal, the discretisation and number of intervals of numeric
 *   attributes, and whether to generate `≠` selectors.
 * @returns The language: its attributes and single selectors, covers (memoised), and the canonical refinement operator.
 *
 * @example A colour and a size: the selectors, and a description's cover
 * const table = {
 *   colour: ['red', 'red', 'blue', 'blue', 'green', 'green'],
 *   size: [1, 2, 3, 4, 5, 6],
 *   bought: [1, 1, 0, 0, 1, 0],
 * }
 * const language = selectorLanguage(table, { exclude: ['bought'], bins: 3 })
 * print('selectors:', language.selectors.map(selectorKey))
 * const d = [{ attribute: 'size', op: '≥', value: 3 }]
 * print('rows of size >= 3:', bitsetIndices(language.cover(d)))
 * print('its refinements:', language.refinements(d).map(descriptionKey))
 * print('refinements of colour = blue:', language.refinements([language.selectors[0]]).map(descriptionKey))
 */
export function selectorLanguage(table: Table, options: LanguageOptions = {}): SelectorLanguage {
  const discretisation = options.discretisation ?? 'equal-frequency'
  const bins = options.bins ?? 4
  const negations = options.negations ?? false
  const exclude = new Set(options.exclude ?? [])
  const nominal = new Set(options.nominal ?? [])
  const names = Object.keys(table).filter((k) => !exclude.has(k))
  if (!names.length) throw new DomainError('selectorLanguage', 'selectorLanguage: the table has no attributes')
  const columns = new Map<string, readonly (string | number)[]>()
  let rows = -1
  for (const k of names) {
    const v = values(table[k])
    if (rows >= 0 && v.length !== rows)
      throw new DomainError('selectorLanguage', `selectorLanguage: column '${k}' has ${v.length} rows, not ${rows}`)
    rows = v.length
    columns.set(k, v)
  }
  const attributes: LanguageAttribute[] = names.map((name) => {
    const v = columns.get(name)!
    const isNumeric = v.every((x) => typeof x === 'number') && !nominal.has(name)
    if (!isNumeric) {
      const levels = [...new Set(v)].sort((a, b) =>
        typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b)),
      )
      return { name, kind: 'nominal', levels, cuts: [] }
    }
    const cuts = discretisation === 'on-the-fly' ? [] : cutPoints(v as number[], bins, discretisation)
    return { name, kind: 'numeric', levels: [], cuts }
  })
  const index = new Map(attributes.map((a, i) => [a.name, i]))
  const ordinal = (s: Selector) => index.get(s.attribute)! * 4 + OPS.indexOf(s.op)

  const selectorCovers = new Map<string, Bitset>()
  const selectorCover = (s: Selector): Bitset => {
    const key = selectorKey(s)
    let c = selectorCovers.get(key)
    if (c) return c
    const col = columns.get(s.attribute)
    if (!col) throw new DomainError('selectorLanguage', `selectorLanguage: unknown attribute '${s.attribute}'`)
    const v = s.value
    const test: (x: string | number) => boolean =
      s.op === '='
        ? (x) => x === v
        : s.op === '≠'
          ? (x) => x !== v
          : s.op === '≥'
            ? (x) => (x as number) >= (v as number)
            : (x) => (x as number) <= (v as number)
    c = bitset(rows, (i) => test(col[i]))
    selectorCovers.set(key, c)
    return c
  }
  const all = bitsetFull(rows)
  const covers = new Map<string, Bitset>()
  const cover = (d: Description): Bitset => {
    if (!d.length) return all
    const key = descriptionKey(d)
    let c = covers.get(key)
    if (c) return c
    c = bitsetAnd(cover(d.slice(0, -1)), selectorCover(d[d.length - 1]))
    if (covers.size > 200_000) covers.clear()
    covers.set(key, c)
    return c
  }
  /** Numeric cuts for refining d: fixed, or the equal-frequency cuts of the values d covers. */
  const cutsFor = (a: LanguageAttribute, d: Description): readonly number[] => {
    if (discretisation !== 'on-the-fly') return a.cuts
    const col = columns.get(a.name)! as readonly number[]
    const inside = Array.from(bitsetIndices(cover(d)), (i) => col[i])
    return cutPoints(inside, bins, 'equal-frequency')
  }
  const extensions = (d: Description): Selector[] => {
    const last = d.length ? d[d.length - 1] : null
    const after = last ? ordinal(last) : -1
    const out: Selector[] = []
    for (let ai = last ? index.get(last.attribute)! : 0; ai < attributes.length; ai++) {
      const a = attributes[ai]
      if (last && last.attribute === a.name && last.op === '=') continue
      if (a.kind === 'nominal') {
        if (ai * 4 > after) for (const v of a.levels) out.push({ attribute: a.name, op: '=', value: v })
        if (negations && ai * 4 + 1 > after)
          for (const v of a.levels) out.push({ attribute: a.name, op: '≠', value: v })
      } else {
        const cuts = cutsFor(a, d)
        if (ai * 4 + 2 > after) for (const c of cuts) out.push({ attribute: a.name, op: '≥', value: c })
        if (ai * 4 + 3 > after) {
          const lower = last && last.attribute === a.name && last.op === '≥' ? (last.value as number) : -Infinity
          for (const c of cuts) if (c > lower) out.push({ attribute: a.name, op: '≤', value: c })
        }
      }
    }
    return out
  }
  return {
    rows,
    attributes,
    discretisation,
    bins,
    negations,
    selectors: extensions([]),
    selectorCover,
    cover,
    extensions,
    refinements: (d) => extensions(d).map((s) => [...d, s]),
    canonical: (d) => [...d].sort((a, b) => ordinal(a) - ordinal(b)),
  }
}
