/**
 * Search over a space defined by a refinement operator: a root (the most general node), `refine` (a node's immediate
 * specialisations), a quality to maximise and, optionally, an optimistic estimate that bounds the quality of every
 * refinement of a node. One step-through algorithm runs four strategies over it:
 *
 * - **beam**: level-wise; each step refines every node of the beam and keeps the `beamWidth` best refinements as the
 *   next beam (Clark and Niblett, 1989, CN2; Lavrač, Kavšek, Flach and Todorovski, 2004, CN2-SD);
 * - **best-first**: each step expands the open node with the highest optimistic estimate (its quality without one);
 * - **depth-first** and **breadth-first**: each step expands the top of a stack or the front of a queue.
 *
 * With `prune` (the default when the space has a `bound`) every strategy is branch and bound: a node whose optimistic
 * estimate is no better than the current k-th best quality is not expanded, because no refinement of it can enter the
 * top k (Webb, 1995, OPUS; Grosskreutz, Rüping and Wrobel, 2008). Depth-first or breadth-first without pruning is
 * exhaustive search to `maxDepth`. Every node evaluated is offered to a top-k result set, optionally filtered for
 * redundancy by a caller's test (two nodes describing the same pattern keep only the better one).
 *
 * States are plain data: the frontier, the results, and what the step did (the nodes expanded, the refinements
 * generated with their quality, bound and fate, the open nodes discarded when the bound fell below a risen threshold),
 * so a figure can play the search one step at a time.
 */

import type { Algorithm, Size, Status } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { run } from 'aifn-compute/foundation/trace'

/** A search space given by a refinement operator. Nodes must be plain data (they are kept in states). */
export interface SearchSpace<N> {
  /** The most general node; depth 0. */
  readonly root: N
  /** The immediate refinements of a node. A canonical operator reaches each node once; otherwise give `key`. */
  refine(node: N): readonly N[]
  /** The quality of a node, higher is better; −∞ (or NaN) for a node that may not be a result. */
  quality(node: N): number
  /**
   * An optimistic estimate: an upper bound on the quality of every refinement of `node`, at any depth. Branch and
   * bound is exact only when this holds.
   */
  bound?(node: N): number
  /** A canonical key, so a node reached twice is evaluated once. Default `JSON.stringify`. */
  key?(node: N): string
  /** False when no refinement of `node` can be a result (an anti-monotone constraint such as minimum support). */
  expandable?(node: N): boolean
}

/** How the frontier is ordered: see the module comment. */
export type SearchStrategy = 'beam' | 'best-first' | 'depth-first' | 'breadth-first'

/** True when two nodes describe the same pattern, so only the better is kept (e.g. their covers overlap enough). */
export type RedundancyTest<N> = (a: N, b: N) => boolean

/** Options of `refinementSearchSteps`. */
export interface SearchOptions<N> {
  /** Default `beam`. */
  strategy?: SearchStrategy
  /** Nodes kept per level by beam search (default 10). */
  beamWidth?: Size
  /** The deepest refinement level searched; the root is depth 0 (default 3). */
  maxDepth?: Size
  /** Results kept (default 10). */
  k?: Size
  /** Branch and bound: skip a node whose bound is ≤ the k-th best quality. Default true when the space has `bound`. */
  prune?: boolean
  /** Only nodes of quality above this are results; also the pruning threshold until k results are held. Default −∞. */
  minQuality?: number
  /** A redundancy filter for the results; with one, pruning is exact for the filtered set only approximately. */
  redundant?: RedundancyTest<N>
  /** Offer the root as a result too (default false). */
  includeRoot?: boolean
  /** Stop (terminated) after evaluating this many nodes (default unlimited). */
  maxNodes?: Size
}

/**
 * What became of a node: `root`; `queued` (in the frontier or the next beam); `dropped` (beam search: not among the
 * best `beamWidth`); `pruned` (its bound cannot beat the k-th best); `duplicate` (reached before; not re-evaluated);
 * `leaf` (at `maxDepth`, or not `expandable`); `expanded` (only in `searchHistory`).
 */
export type VisitFate = 'root' | 'queued' | 'dropped' | 'pruned' | 'duplicate' | 'leaf' | 'expanded'

/** One evaluated node of the search. */
export interface SearchVisit<N> {
  /** Order of evaluation, from 0 (the root). */
  readonly id: number
  /** The id of the node it refines; −1 for the root. */
  readonly parent: number
  readonly depth: number
  readonly node: N
  readonly key: string
  /** NaN for a duplicate (not evaluated). */
  readonly quality: number
  /** The optimistic estimate; +∞ without one. */
  readonly bound: number
  readonly fate: VisitFate
}

/** A state of `refinementSearchSteps`. */
export interface SearchState<N> extends Status {
  /** Open nodes after the step: the beam; best-first by priority (highest first); depth-first top last; queue front first. */
  readonly frontier: readonly SearchVisit<N>[]
  /** The top k, best first (ties by evaluation order). */
  readonly results: readonly SearchVisit<N>[]
  /** The nodes this step expanded (beam search: the whole beam). */
  readonly expanded: readonly SearchVisit<N>[]
  /** The refinements this step generated, each with its fate. */
  readonly generated: readonly SearchVisit<N>[]
  /** Open nodes this step removed unexpanded because the threshold rose above their bound. */
  readonly discarded: readonly SearchVisit<N>[]
  /** The quality a bound must exceed to be explored: the k-th best once k results are held, else `minQuality`. */
  readonly threshold: number
  /** The best quality so far (NaN before any result). */
  readonly best: number
  /** Beam search: the depth of the current beam. Other strategies: the depth of the node last expanded. */
  readonly level: number
  /** Nodes evaluated so far (the root included). */
  readonly evaluated: number
  /** Nodes expanded so far. */
  readonly expansions: number
  /** Nodes pruned so far (when generated, or when discarded from the frontier). */
  readonly pruned: number
  /** Keys of every node evaluated, for duplicate detection. */
  readonly seen: readonly string[]
  readonly nextId: number
  readonly terminated: boolean
}

interface Resolved<N> {
  strategy: SearchStrategy
  beamWidth: number
  maxDepth: number
  k: number
  prune: boolean
  minQuality: number
  redundant?: RedundancyTest<N>
  includeRoot: boolean
  maxNodes: number
}

function resolve<N>(space: SearchSpace<N>, o: SearchOptions<N>): Resolved<N> {
  const r = {
    strategy: o.strategy ?? 'beam',
    beamWidth: o.beamWidth ?? 10,
    maxDepth: o.maxDepth ?? 3,
    k: o.k ?? 10,
    prune: o.prune ?? space.bound !== undefined,
    minQuality: o.minQuality ?? -Infinity,
    redundant: o.redundant,
    includeRoot: o.includeRoot ?? false,
    maxNodes: o.maxNodes ?? Infinity,
  }
  const where = 'refinementSearchSteps'
  if (!['beam', 'best-first', 'depth-first', 'breadth-first'].includes(r.strategy))
    throw new DomainError(where, `${where}: unknown strategy '${String(r.strategy)}'`)
  if (!(Number.isInteger(r.beamWidth) && r.beamWidth >= 1))
    throw new DomainError(where, `${where}: beamWidth must be a positive integer`)
  if (!(Number.isInteger(r.k) && r.k >= 1)) throw new DomainError(where, `${where}: k must be a positive integer`)
  if (!(r.maxDepth >= 0)) throw new DomainError(where, `${where}: maxDepth must be ≥ 0`)
  return r
}

/** Results order: higher quality first, then earlier evaluation. */
const byQuality = <N>(a: SearchVisit<N>, b: SearchVisit<N>) => b.quality - a.quality || a.id - b.id

/**
 * `results` with `visit` offered: kept when its quality exceeds `minQuality` and it ranks in the top `k`. With a
 * redundancy test, a visit redundant with a better-or-equal result is refused, and results redundant with it and worse
 * are removed.
 */
export function offerResult<N>(
  results: readonly SearchVisit<N>[],
  visit: SearchVisit<N>,
  k: Size,
  options: { minQuality?: number; redundant?: RedundancyTest<N> } = {},
): readonly SearchVisit<N>[] {
  const q = visit.quality
  if (!(q > (options.minQuality ?? -Infinity))) return results
  if (results.length >= k && byQuality(visit, results[k - 1]) >= 0) return results
  let kept = results
  const redundant = options.redundant
  if (redundant) {
    for (const r of results) if (r.quality >= q && redundant(visit.node, r.node)) return results
    kept = results.filter((r) => !redundant(visit.node, r.node))
  }
  const out = [...kept, visit].sort(byQuality)
  return out.length > k ? out.slice(0, k) : out
}

/** The top `k` of `visits` with the redundancy filter applied greedily from the best down. */
export function filterRedundant<N>(
  visits: readonly SearchVisit<N>[],
  redundant: RedundancyTest<N>,
  k: Size = Infinity,
): SearchVisit<N>[] {
  const out: SearchVisit<N>[] = []
  for (const v of [...visits].sort(byQuality)) {
    if (out.length >= k) break
    if (!out.some((r) => redundant(v.node, r.node))) out.push(v)
  }
  return out
}

const thresholdOf = <N>(results: readonly SearchVisit<N>[], o: Resolved<N>) =>
  results.length >= o.k ? Math.max(o.minQuality, results[o.k - 1].quality) : o.minQuality

/**
 * Search a refinement space step by step (see the module comment). Each step of beam search refines a whole level;
 * each step of the other strategies takes one node from the frontier and expands it, or discards it when its bound no
 * longer beats the threshold. The state's `results` are the top k.
 */
export function refinementSearchSteps<N>(
  space: SearchSpace<N>,
  options: SearchOptions<N> = {},
): Algorithm<undefined, SearchState<N>> {
  const o = resolve(space, options)
  const keyOf = space.key ?? ((n: N) => JSON.stringify(n))
  const evaluate = (node: N, parent: number, depth: number, id: number, fate: VisitFate): SearchVisit<N> => {
    const q = space.quality(node)
    return {
      id,
      parent,
      depth,
      node,
      key: keyOf(node),
      quality: Number.isNaN(q) ? -Infinity : q,
      bound: space.bound ? space.bound(node) : Infinity,
      fate,
    }
  }
  const canExpand = (v: SearchVisit<N>) => v.depth < o.maxDepth && (space.expandable?.(v.node) ?? true)
  const prunable = (v: SearchVisit<N>, threshold: number) => o.prune && !(v.bound > threshold)
  const offer = (results: readonly SearchVisit<N>[], v: SearchVisit<N>) =>
    offerResult(results, v, o.k, { minQuality: o.minQuality, redundant: o.redundant })
  const priority = (v: SearchVisit<N>) => (space.bound ? v.bound : v.quality)
  const byPriority = (a: SearchVisit<N>, b: SearchVisit<N>) => priority(b) - priority(a) || a.id - b.id

  /** Evaluate the refinements of `parents`; duplicates are recorded but not evaluated. */
  const refineAll = (s: SearchState<N>, parents: readonly SearchVisit<N>[]) => {
    const seen = new Set(s.seen)
    const keys = [...s.seen]
    let results = s.results
    let nextId = s.nextId
    let evaluated = s.evaluated
    const fresh: SearchVisit<N>[] = []
    const duplicates: SearchVisit<N>[] = []
    for (const p of parents) {
      for (const child of space.refine(p.node)) {
        if (evaluated >= o.maxNodes) break
        const key = keyOf(child)
        if (seen.has(key)) {
          duplicates.push({
            id: -1,
            parent: p.id,
            depth: p.depth + 1,
            node: child,
            key,
            quality: NaN,
            bound: NaN,
            fate: 'duplicate',
          })
          continue
        }
        seen.add(key)
        keys.push(key)
        const v = evaluate(child, p.id, p.depth + 1, nextId++, 'queued')
        evaluated++
        results = offer(results, v)
        fresh.push(v)
      }
    }
    return { fresh, duplicates, results, nextId, evaluated, keys }
  }

  const finish = (
    s: SearchState<N>,
    patch: Omit<SearchState<N>, 't' | 'best' | 'threshold' | 'terminated' | 'seen' | 'nextId' | 'evaluated'> &
      Partial<Pick<SearchState<N>, 'seen' | 'nextId' | 'evaluated'>>,
  ): SearchState<N> => {
    const evaluated = patch.evaluated ?? s.evaluated
    return {
      ...s,
      ...patch,
      t: s.t + 1,
      evaluated,
      best: patch.results.length ? patch.results[0].quality : NaN,
      threshold: thresholdOf(patch.results, o),
      terminated: patch.frontier.length === 0 || evaluated >= o.maxNodes,
    }
  }

  return {
    name: 'refinementSearch',
    init(): SearchState<N> {
      const root = evaluate(space.root, -1, 0, 0, 'root')
      const results = o.includeRoot ? offer([], root) : []
      const frontier = canExpand(root) ? [root] : []
      return {
        t: 0,
        frontier,
        results,
        expanded: [],
        generated: [root],
        discarded: [],
        threshold: thresholdOf(results, o),
        best: results.length ? results[0].quality : NaN,
        level: 0,
        evaluated: 1,
        expansions: 0,
        pruned: 0,
        seen: [root.key],
        nextId: 1,
        terminated: frontier.length === 0,
      }
    },
    step(s): SearchState<N> {
      if (s.terminated) return { ...s, t: s.t + 1 }
      if (o.strategy === 'beam') {
        const beam = s.frontier
        const r = refineAll(s, beam)
        const threshold = thresholdOf(r.results, o)
        let pruned = 0
        const fated = r.fresh.map((v): SearchVisit<N> => {
          if (!canExpand(v)) return { ...v, fate: 'leaf' }
          if (prunable(v, threshold)) {
            pruned++
            return { ...v, fate: 'pruned' }
          }
          return v
        })
        const candidates = fated.filter((v) => v.fate === 'queued').sort((a, b) => byQuality(a, b))
        const kept = new Set(candidates.slice(0, o.beamWidth).map((v) => v.id))
        const generated = [
          ...fated.map((v): SearchVisit<N> => (v.fate === 'queued' && !kept.has(v.id) ? { ...v, fate: 'dropped' } : v)),
          ...r.duplicates,
        ]
        return finish(s, {
          frontier: candidates.filter((v) => kept.has(v.id)),
          results: r.results,
          expanded: beam,
          generated,
          discarded: [],
          level: s.level + 1,
          expansions: s.expansions + beam.length,
          pruned: s.pruned + pruned,
          seen: r.keys,
          nextId: r.nextId,
          evaluated: r.evaluated,
        })
      }
      // One node from the frontier.
      const frontier = [...s.frontier]
      const node = o.strategy === 'depth-first' ? frontier.pop()! : frontier.shift()!
      if (prunable(node, s.threshold)) {
        return finish(s, {
          frontier,
          results: s.results,
          expanded: [],
          generated: [],
          discarded: [{ ...node, fate: 'pruned' }],
          level: node.depth,
          expansions: s.expansions,
          pruned: s.pruned + 1,
        })
      }
      const r = refineAll(s, [node])
      const threshold = thresholdOf(r.results, o)
      let pruned = 0
      const fated = r.fresh.map((v): SearchVisit<N> => {
        if (!canExpand(v)) return { ...v, fate: 'leaf' }
        if (prunable(v, threshold)) {
          pruned++
          return { ...v, fate: 'pruned' }
        }
        return v
      })
      const open = fated.filter((v) => v.fate === 'queued')
      let next: SearchVisit<N>[]
      if (o.strategy === 'depth-first') next = [...frontier, ...open.reverse()]
      else if (o.strategy === 'breadth-first') next = [...frontier, ...open]
      else next = [...frontier, ...open].sort(byPriority)
      return finish(s, {
        frontier: next,
        results: r.results,
        expanded: [node],
        generated: [...fated, ...r.duplicates],
        discarded: [],
        level: node.depth,
        expansions: s.expansions + 1,
        pruned: s.pruned + pruned,
        seen: r.keys,
        nextId: r.nextId,
        evaluated: r.evaluated,
      })
    },
    done: (s) => s.terminated,
  }
}

/** The final state of `refinementSearchSteps`: its `results` are the top k. */
export function refinementSearch<N>(space: SearchSpace<N>, options: SearchOptions<N> = {}): SearchState<N> {
  return run(refinementSearchSteps(space, options), undefined, Infinity)
}

/** Every node a search evaluated, by id, with its final fate, and the step at which each was generated and expanded. */
export interface SearchHistory<N> {
  readonly visits: readonly SearchVisit<N>[]
  /** Step that generated each visit (0 for the root). */
  readonly generatedAt: Int32Array
  /** Step that expanded each visit, −1 if never. */
  readonly expandedAt: Int32Array
  /** Step that discarded each visit from the frontier (its bound fell to the threshold), −1 if never. */
  readonly discardedAt: Int32Array
}

/** The search tree recorded by a sequence of states (`trace(...).steps`, every step kept). */
export function searchHistory<N>(states: readonly SearchState<N>[]): SearchHistory<N> {
  const visits: SearchVisit<N>[] = []
  const generated: number[] = []
  const expanded: number[] = []
  const discarded: number[] = []
  for (const s of states) {
    for (const v of s.generated) {
      if (v.id < 0) continue
      visits[v.id] = v
      generated[v.id] = s.t
      expanded[v.id] ??= -1
      discarded[v.id] ??= -1
    }
    for (const v of s.expanded) {
      expanded[v.id] = s.t
      visits[v.id] = { ...visits[v.id], fate: 'expanded' }
    }
    for (const v of s.discarded) {
      discarded[v.id] = s.t
      visits[v.id] = { ...visits[v.id], fate: 'pruned' }
    }
  }
  return {
    visits,
    generatedAt: Int32Array.from(generated),
    expandedAt: Int32Array.from(expanded),
    discardedAt: Int32Array.from(discarded),
  }
}
