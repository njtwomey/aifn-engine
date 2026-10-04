/**
 * SD-Map (Atzmüller and Puppe, 2006, "SD-Map: a fast algorithm for exhaustive subgroup discovery"): exhaustive
 * subgroup discovery for a binary target by FP-growth (Han, Pei and Yin, 2000). Selectors are items; one pass builds a
 * frequent-pattern tree whose nodes count both the rows and the positive rows on their path, so every conjunction's
 * (n, tp), and with it any count-based quality, is read from the tree without touching the data again. Conditional
 * trees are built per item, least frequent first, down to `maxDepth` selectors. With `prune`, a conditional tree is not
 * built when the measure's optimistic estimate of its pattern cannot beat the current k-th quality.
 *
 * It finds the same subgroups as exhaustive search over the same language (the same valid conjunctions; a minimum
 * support of at least 1), but with a fixed discretisation only.
 */

import { DomainError } from 'aifn-compute/foundation/errors'
import { offerResult, type SearchVisit } from 'aifn-compute/optim/search'
import { bitsetCount, bitsetHas } from './cover'
import { compatibleSelector, descriptionKey, type Description, type Selector, type SelectorLanguage } from './language'
import type { CountMeasure } from './quality'
import type { Subgroup } from './subgroups'

/** Options of `sdMap`. */
export interface SdMapOptions {
  /** Rows a subgroup must cover (default 1). */
  minSupport?: number
  /** Selectors per description at most (default 3). */
  maxDepth?: number
  /** Subgroups returned (default 10). */
  k?: number
  /** Skip conditional trees whose optimistic estimate cannot enter the top k (default true when the measure has one). */
  prune?: boolean
  /** Results only above this quality (default −∞). */
  minQuality?: number
}

/** A frequent-pattern tree with row and positive counts per node. */
interface FpTree {
  item: number[]
  count: number[]
  pos: number[]
  parent: number[]
  children: Map<number, number>[]
  /** Per item, its nodes. */
  header: Map<number, number[]>
}

function newTree(): FpTree {
  return { item: [-1], count: [0], pos: [0], parent: [-1], children: [new Map()], header: new Map() }
}

/** Insert a path of items (already in tree order) with its counts. */
function insert(tree: FpTree, items: readonly number[], count: number, pos: number) {
  let v = 0
  for (const it of items) {
    let c = tree.children[v].get(it)
    if (c === undefined) {
      c = tree.item.length
      tree.item.push(it)
      tree.count.push(0)
      tree.pos.push(0)
      tree.parent.push(v)
      tree.children.push(new Map())
      tree.children[v].set(it, c)
      const h = tree.header.get(it)
      if (h) h.push(c)
      else tree.header.set(it, [c])
    }
    tree.count[c] += count
    tree.pos[c] += pos
    v = c
  }
}

/** True when selector `s` may join the selectors of `pattern` (the language's rule, and a non-empty interval). */
function joins(pattern: readonly Selector[], s: Selector): boolean {
  if (!compatibleSelector(pattern, s)) return false
  for (const p of pattern)
    if (p.attribute === s.attribute) {
      const ge = p.op === '≥' ? p : s
      const le = p.op === '≤' ? p : s
      if (!((le.value as number) > (ge.value as number))) return false
    }
  return true
}

/** The top k subgroups of a binary target by a count-based measure, found exhaustively by FP-growth. */
export function sdMap(language: SelectorLanguage, measure: CountMeasure, options: SdMapOptions = {}): Subgroup[] {
  if (language.discretisation === 'on-the-fly')
    throw new DomainError('sdMap', 'sdMap: needs a fixed discretisation (equal-frequency or equal-width)')
  const minSupport = Math.max(1, options.minSupport ?? 1)
  const maxDepth = options.maxDepth ?? 3
  const k = options.k ?? 10
  const prune = options.prune ?? measure.boundFromCounts !== undefined
  const minQuality = options.minQuality ?? -Infinity
  const n = language.rows
  // Frequent items, most frequent first (ties by language order): the tree's item order.
  const all = language.selectors.map((s, i) => ({ s, i, support: bitsetCount(language.selectorCover(s)) }))
  const items = all.filter((x) => x.support >= minSupport).sort((a, b) => b.support - a.support || a.i - b.i)
  const selector = items.map((x) => x.s)
  const covers = selector.map((s) => language.selectorCover(s))
  const tree = newTree()
  for (let r = 0; r < n; r++) {
    const path: number[] = []
    for (let j = 0; j < covers.length; j++) if (bitsetHas(covers[j], r)) path.push(j)
    if (path.length) insert(tree, path, 1, bitsetHas(measure.target, r) ? 1 : 0)
  }
  let results: readonly SearchVisit<Description>[] = []
  let id = 0
  const threshold = () => (results.length >= k ? Math.max(minQuality, results[k - 1].quality) : minQuality)

  const mine = (t: FpTree, suffix: readonly number[]) => {
    // Least frequent item first, as FP-growth does; item ids are positions in the global order.
    const order = [...t.header.keys()].sort((a, b) => b - a)
    for (const it of order) {
      const nodes = t.header.get(it)!
      let cnt = 0
      let pos = 0
      for (const v of nodes) {
        cnt += t.count[v]
        pos += t.pos[v]
      }
      if (cnt < minSupport) continue
      const sels = suffix.map((j) => selector[j])
      if (!joins(sels, selector[it])) continue
      const pattern = [...suffix, it]
      const description = language.canonical(pattern.map((j) => selector[j]))
      const q = measure.fromCounts(cnt, pos)
      const visit: SearchVisit<Description> = {
        id: id++,
        parent: -1,
        depth: pattern.length,
        node: description,
        key: descriptionKey(description),
        quality: q,
        bound: measure.boundFromCounts?.(cnt, pos) ?? Infinity,
        fate: 'leaf',
      }
      results = offerResult(results, visit, k, { minQuality })
      if (pattern.length >= maxDepth) continue
      if (prune && !(visit.bound > threshold())) continue
      // The conditional pattern base of `it`: each node's prefix path, with the node's counts.
      const base: { path: number[]; count: number; pos: number }[] = []
      const support = new Map<number, number>()
      for (const v of nodes) {
        const path: number[] = []
        for (let u = t.parent[v]; u > 0; u = t.parent[u]) path.push(t.item[u])
        path.reverse()
        base.push({ path, count: t.count[v], pos: t.pos[v] })
        for (const j of path) support.set(j, (support.get(j) ?? 0) + t.count[v])
      }
      const patternSels = pattern.map((j) => selector[j])
      const keep = (j: number) => (support.get(j) ?? 0) >= minSupport && joins(patternSels, selector[j])
      const conditional = newTree()
      for (const b of base) {
        const path = b.path.filter(keep)
        if (path.length) insert(conditional, path, b.count, b.pos)
      }
      if (conditional.header.size) mine(conditional, pattern)
    }
  }
  mine(tree, [])
  return results.map((v) => {
    const cover = language.cover(v.node)
    return { description: v.node, key: v.key, quality: v.quality, size: bitsetCount(cover), cover }
  })
}
