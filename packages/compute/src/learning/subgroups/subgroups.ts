/**
 * Subgroup discovery (Klösgen, 1996; Wrobel, 1997): find the descriptions of a table whose covers are largest and most
 * unusual in a target, by a quality measure. The description language and its canonical refinement operator define a
 * search space for `aifn-compute/optim/search`, which runs it by beam, best-first, depth-first or breadth-first
 * search, with branch and bound when the measure has an optimistic estimate; a minimum support is an anti-monotone
 * constraint (a description below it is neither a result nor refined).
 *
 * Redundancy: two subgroups are redundant when their covers overlap (Jaccard index at or above a threshold), or when
 * one's description contains the other's; the top $k$ keep only the better of a redundant pair (van Leeuwen and
 * Knobbe, 2012, "Diverse subgroup set discovery").
 */

import type { Algorithm } from 'aifn-compute/foundation/contracts'
import {
  refinementSearch,
  refinementSearchSteps,
  type RedundancyTest,
  type SearchOptions,
  type SearchSpace,
  type SearchState,
  type SearchStrategy,
} from 'aifn-compute/optim/search'
import { bitsetCount, bitsetJaccard, type Bitset } from './cover'
import { descriptionKey, selectorKey, type Description, type SelectorLanguage } from './language'
import type { QualityMeasure } from './quality'

/**
 * How the results are filtered for redundancy: `none`; `cover`, two subgroups whose covers overlap with a Jaccard
 * index at or above `threshold`; or `description`, two subgroups one of whose selectors all appear in the other's.
 */
export type Redundancy =
  | { kind: 'none' }
  | {
      kind: 'cover'
      /** Jaccard index of the covers at or above which two subgroups are redundant (default 0.8). */
      threshold?: number
    }
  | { kind: 'description' }

/** Options of subgroup discovery. */
export interface SubgroupOptions {
  /** The search strategy of `refinementSearch` (default `beam`). */
  strategy?: SearchStrategy
  /** Descriptions kept per level by beam search (default 10). */
  beamWidth?: number
  /** Selectors per description at most (default 3). */
  maxDepth?: number
  /** Subgroups returned (default 10). */
  k?: number
  /** Rows a subgroup must cover (default 1). */
  minSupport?: number
  /** Branch and bound with the measure's optimistic estimate (default true when it has one). */
  prune?: boolean
  /** The redundancy filter of the results (default `none`). */
  redundancy?: Redundancy
  /** Results only above this quality (default $-\infty$). */
  minQuality?: number
  /** Stop after evaluating this many descriptions (default unlimited). */
  maxNodes?: number
}

/** A subgroup found: its description, cover, size and quality. */
export interface Subgroup {
  /** The selectors, in canonical order. */
  readonly description: Description
  /** The description's canonical key (`descriptionKey`). */
  readonly key: string
  /** The measure's quality of the cover. */
  readonly quality: number
  /** The number of rows covered. */
  readonly size: number
  /** The rows covered. */
  readonly cover: Bitset
}

/**
 * The redundancy test of `redundancy` over a language's covers (undefined for `none`): covers whose Jaccard index is
 * at or above the threshold (default 0.8), or descriptions one of which holds every selector of the other.
 *
 * @param language The language whose covers are compared.
 * @param redundancy The kind of redundancy, and its threshold for `cover`.
 * @returns A test of two descriptions, true when they are redundant, as `refinementSearch` takes it.
 *
 * @example Two descriptions with the same rows
 * const table = {
 *   colour: ['red', 'red', 'red', 'blue', 'blue', 'green', 'green', 'green'],
 *   size: [1, 2, 3, 4, 5, 6, 7, 8],
 *   bought: [1, 1, 1, 0, 0, 0, 0, 1],
 * }
 * const language = selectorLanguage(table, { exclude: ['bought'], bins: 4 })
 * const red = [{ attribute: 'colour', op: '=', value: 'red' }]
 * const small = [{ attribute: 'size', op: '≤', value: 3 }]
 * print('cover:', subgroupRedundancy(language, { kind: 'cover' })(red, small))
 * print('description:', subgroupRedundancy(language, { kind: 'description' })(red, small))
 */
export function subgroupRedundancy(
  language: SelectorLanguage,
  redundancy: Redundancy = { kind: 'none' },
): RedundancyTest<Description> | undefined {
  if (redundancy.kind === 'none') return undefined
  if (redundancy.kind === 'cover') {
    const t = redundancy.threshold ?? 0.8
    return (a, b) => bitsetJaccard(language.cover(a), language.cover(b)) >= t
  }
  return (a, b) => {
    const [small, large] = a.length <= b.length ? [a, b] : [b, a]
    const keys = new Set(large.map(selectorKey))
    return small.every((s) => keys.has(selectorKey(s)))
  }
}

/**
 * The search space of subgroup discovery: descriptions refined canonically from the empty one, scored by `measure`,
 * with its optimistic estimate as the bound when it has one. A description covering fewer than `minSupport` rows has
 * quality and bound $-\infty$ and is not refined.
 *
 * @param language The description language.
 * @param measure The quality measure of covers.
 * @param options `minSupport`, the rows a description must cover (default 1).
 * @returns The space for `refinementSearch` and `refinementSearchSteps`.
 *
 * @example The root's refinements and their qualities
 * const table = {
 *   colour: ['red', 'red', 'red', 'blue', 'blue', 'green', 'green', 'green'],
 *   size: [1, 2, 3, 4, 5, 6, 7, 8],
 *   bought: [1, 1, 1, 0, 0, 0, 0, 1],
 * }
 * const language = selectorLanguage(table, { exclude: ['bought'], bins: 4 })
 * const space = subgroupSpace(language, wraccQuality(table.bought), { minSupport: 3 })
 * for (const d of space.refine(space.root)) print(descriptionKey(d), space.quality(d))
 */
export function subgroupSpace(
  language: SelectorLanguage,
  measure: QualityMeasure,
  options: { minSupport?: number } = {},
): SearchSpace<Description> {
  const minSupport = Math.max(1, options.minSupport ?? measure.minSupport ?? 1)
  const m =
    'withMinSupport' in measure && typeof (measure as any).withMinSupport === 'function'
      ? (measure as any).withMinSupport(minSupport)
      : measure
  const big = (d: Description) => bitsetCount(language.cover(d)) >= minSupport
  return {
    root: [],
    refine: (d) => language.refinements(d),
    quality: (d) => (big(d) ? m.quality(language.cover(d)) : -Infinity),
    ...(m.bound ? { bound: (d: Description) => (big(d) ? m.bound!(language.cover(d)) : -Infinity) } : {}),
    key: descriptionKey,
    expandable: big,
  }
}

/**
 * The options of `refinementSearch` for subgroup discovery: the defaults of `SubgroupOptions` filled in, and the
 * redundancy filter built.
 *
 * @param language The language, whose covers the redundancy filter compares.
 * @param o The subgroup discovery options.
 * @returns The search options.
 */
function searchOptions(language: SelectorLanguage, o: SubgroupOptions): SearchOptions<Description> {
  return {
    strategy: o.strategy ?? 'beam',
    beamWidth: o.beamWidth ?? 10,
    maxDepth: o.maxDepth ?? 3,
    k: o.k ?? 10,
    ...(o.prune !== undefined ? { prune: o.prune } : {}),
    ...(o.minQuality !== undefined ? { minQuality: o.minQuality } : {}),
    ...(o.maxNodes !== undefined ? { maxNodes: o.maxNodes } : {}),
    redundant: subgroupRedundancy(language, o.redundancy),
  }
}

/**
 * Subgroup discovery step by step: the states of `refinementSearchSteps` over `subgroupSpace`.
 *
 * @param language The description language.
 * @param measure The quality measure of covers.
 * @param options The search and its limits, as `subgroupDiscovery` takes them.
 * @returns The algorithm: run it from `undefined` and read `results` from a state.
 *
 * @example Beam search, one level per step
 * const table = {
 *   colour: ['red', 'red', 'red', 'blue', 'blue', 'green', 'green', 'green'],
 *   size: [1, 2, 3, 4, 5, 6, 7, 8],
 *   bought: [1, 1, 1, 0, 0, 0, 0, 1],
 * }
 * const language = selectorLanguage(table, { exclude: ['bought'], bins: 4 })
 * const alg = subgroupDiscoverySteps(language, wraccQuality(table.bought), { k: 3, beamWidth: 2 })
 * for (let k = 1; k <= 3; k++) {
 *   const s = run(alg, undefined, k)
 *   print(`step ${k}: level`, s.level, 'terminated =', s.terminated, 'results:', s.results.map((v) => v.key))
 * }
 */
export function subgroupDiscoverySteps(
  language: SelectorLanguage,
  measure: QualityMeasure,
  options: SubgroupOptions = {},
): Algorithm<undefined, SearchState<Description>> {
  const minSupport = Math.max(1, options.minSupport ?? measure.minSupport ?? 1)
  const m =
    'withMinSupport' in measure && typeof (measure as any).withMinSupport === 'function'
      ? (measure as any).withMinSupport(minSupport)
      : measure
  return refinementSearchSteps(
    subgroupSpace(language, m, { minSupport }),
    searchOptions(language, { ...options, minSupport }),
  )
}

/**
 * The subgroup of a description: cover, size and quality, as `subgroupDiscovery` reports them.
 *
 * @param language The language that covers the description.
 * @param measure The quality measure.
 * @param description The description, in canonical order (`language.canonical` puts one in order).
 * @returns The subgroup.
 *
 * @example A description chosen by hand
 * const table = {
 *   colour: ['red', 'red', 'red', 'blue', 'blue', 'green', 'green', 'green'],
 *   size: [1, 2, 3, 4, 5, 6, 7, 8],
 *   bought: [1, 1, 1, 0, 0, 0, 0, 1],
 * }
 * const language = selectorLanguage(table, { exclude: ['bought'], bins: 4 })
 * const g = subgroupOf(language, liftQuality(table.bought), [{ attribute: 'colour', op: '=', value: 'red' }])
 * print(g.key, 'size =', g.size, 'lift =', g.quality, 'rows =', bitsetIndices(g.cover))
 */
export function subgroupOf(language: SelectorLanguage, measure: QualityMeasure, description: Description): Subgroup {
  const cover = language.cover(description)
  return {
    description,
    key: descriptionKey(description),
    quality: measure.quality(cover),
    size: bitsetCount(cover),
    cover,
  }
}

/**
 * The top $k$ subgroups, best first, by `refinementSearch` over `subgroupSpace` with the options' strategy, depth,
 * support and redundancy filter.
 *
 * @param language The description language.
 * @param measure The quality measure of covers.
 * @param options The search strategy and its limits, the number of results, and the redundancy filter.
 * @returns The subgroups found, at most `k`.
 *
 * @example The top three by WRAcc, with and without the redundancy filter
 * const table = {
 *   colour: ['red', 'red', 'red', 'blue', 'blue', 'green', 'green', 'green'],
 *   size: [1, 2, 3, 4, 5, 6, 7, 8],
 *   bought: [1, 1, 1, 0, 0, 0, 0, 1],
 * }
 * const language = selectorLanguage(table, { exclude: ['bought'], bins: 4 })
 * const wracc = wraccQuality(table.bought)
 * for (const g of subgroupDiscovery(language, wracc, { k: 3 })) print(formatDescription(g.description), g.quality)
 * print('with redundant covers removed:')
 * for (const g of subgroupDiscovery(language, wracc, { k: 3, redundancy: { kind: 'cover' } }))
 *   print(formatDescription(g.description), g.quality)
 */
export function subgroupDiscovery(
  language: SelectorLanguage,
  measure: QualityMeasure,
  options: SubgroupOptions = {},
): Subgroup[] {
  const minSupport = Math.max(1, options.minSupport ?? measure.minSupport ?? 1)
  const m =
    'withMinSupport' in measure && typeof (measure as any).withMinSupport === 'function'
      ? (measure as any).withMinSupport(minSupport)
      : measure
  const final = refinementSearch(
    subgroupSpace(language, m, { minSupport }),
    searchOptions(language, { ...options, minSupport }),
  )
  return final.results.map((v) => ({ ...subgroupOf(language, m, v.node), quality: v.quality }))
}
