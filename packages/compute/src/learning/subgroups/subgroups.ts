/**
 * Subgroup discovery (Klösgen, 1996; Wrobel, 1997): find the descriptions of a table whose covers are largest and most
 * unusual in a target, by a quality measure. The description language and its canonical refinement operator define a
 * search space for `aifn-compute/optim/search`, which runs it by beam, best-first, depth-first or breadth-first search, with
 * branch and bound when the measure has an optimistic estimate; a minimum support is an anti-monotone constraint (a
 * description below it is neither a result nor refined).
 *
 * Redundancy: two subgroups are redundant when their covers overlap (Jaccard index at or above a threshold), or when
 * one's description contains the other's; the top k keep only the better of a redundant pair (van Leeuwen and
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

/** How the results are filtered for redundancy. */
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
  /** Default `beam`. */
  strategy?: SearchStrategy
  /** Default 10. */
  beamWidth?: number
  /** Selectors per description at most (default 3). */
  maxDepth?: number
  /** Subgroups returned (default 10). */
  k?: number
  /** Rows a subgroup must cover (default 1). */
  minSupport?: number
  /** Branch and bound with the measure's optimistic estimate (default true when it has one). */
  prune?: boolean
  /** Default `none`. */
  redundancy?: Redundancy
  /** Results only above this quality (default −∞). */
  minQuality?: number
  /** Stop after evaluating this many descriptions (default unlimited). */
  maxNodes?: number
}

/** A subgroup found: its description, cover, size and quality. */
export interface Subgroup {
  readonly description: Description
  readonly key: string
  readonly quality: number
  readonly size: number
  readonly cover: Bitset
}

/** The redundancy test of `redundancy` over a language's covers (undefined for `none`). */
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

/** The search space of subgroup discovery: descriptions refined canonically, scored by `measure`. */
export function subgroupSpace(
  language: SelectorLanguage,
  measure: QualityMeasure,
  options: { minSupport?: number } = {},
): SearchSpace<Description> {
  const minSupport = Math.max(1, options.minSupport ?? 1)
  const big = (d: Description) => bitsetCount(language.cover(d)) >= minSupport
  return {
    root: [],
    refine: (d) => language.refinements(d),
    quality: (d) => (big(d) ? measure.quality(language.cover(d)) : -Infinity),
    ...(measure.bound ? { bound: (d: Description) => (big(d) ? measure.bound!(language.cover(d)) : -Infinity) } : {}),
    key: descriptionKey,
    expandable: big,
  }
}

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

/** Subgroup discovery step by step: the states of `refinementSearchSteps` over `subgroupSpace`. */
export function subgroupDiscoverySteps(
  language: SelectorLanguage,
  measure: QualityMeasure,
  options: SubgroupOptions = {},
): Algorithm<undefined, SearchState<Description>> {
  return refinementSearchSteps(
    subgroupSpace(language, measure, { minSupport: options.minSupport }),
    searchOptions(language, options),
  )
}

/** The subgroup of a description: cover, size and quality. */
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

/** The top k subgroups, best first. */
export function subgroupDiscovery(
  language: SelectorLanguage,
  measure: QualityMeasure,
  options: SubgroupOptions = {},
): Subgroup[] {
  const final = refinementSearch(
    subgroupSpace(language, measure, { minSupport: options.minSupport }),
    searchOptions(language, options),
  )
  return final.results.map((v) => ({ ...subgroupOf(language, measure, v.node), quality: v.quality }))
}
