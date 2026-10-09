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
  /** The quality of a node, higher is better; $-\infty$ (or NaN, read as such) for a node that may not be a result. */
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

/** How the frontier is ordered and taken: see the file comment. */
export type SearchStrategy = 'beam' | 'best-first' | 'depth-first' | 'breadth-first'

/** True when two nodes describe the same pattern, so only the better is kept (e.g. their covers overlap enough). */
export type RedundancyTest<N> = (a: N, b: N) => boolean

/** Options of `refinementSearchSteps`. */
export interface SearchOptions<N> {
  /** How the frontier is ordered and taken (see the file comment). Default `beam`. */
  strategy?: SearchStrategy
  /** Nodes kept per level by beam search (default 10). */
  beamWidth?: Size
  /** The deepest refinement level searched; the root is depth 0 (default 3). */
  maxDepth?: Size
  /** Results kept (default 10). */
  k?: Size
  /**
   * Branch and bound: skip a node whose bound is at most the threshold, the $k$-th best quality (or NaN). Default true
   * when the space has `bound`.
   */
  prune?: boolean
  /**
   * Only nodes of quality above this are results; also the pruning threshold until $k$ results are held. Default
   * $-\infty$.
   */
  minQuality?: number
  /** A redundancy filter for the results; with one, pruning is exact for the filtered set only approximately. */
  redundant?: RedundancyTest<N>
  /** Offer the root as a result too (default false). */
  includeRoot?: boolean
  /** Stop (terminated) after evaluating this many nodes, the root included (default unlimited). */
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
  /** The id of the node it refines; $-1$ for the root. */
  readonly parent: number
  /** The number of refinements from the root (0 for the root). */
  readonly depth: number
  /** The node itself. */
  readonly node: N
  /** The node's canonical key, from the space's `key` (or `JSON.stringify`). */
  readonly key: string
  /** The node's quality: $-\infty$ when `quality` gave NaN, and NaN for a duplicate (not evaluated). */
  readonly quality: number
  /** The optimistic estimate; $+\infty$ without one, and NaN for a duplicate. */
  readonly bound: number
  /** What became of the node on the step that generated it (or, in `searchHistory`, in the end). */
  readonly fate: VisitFate
}

/** A state of `refinementSearchSteps`. */
export interface SearchState<N> extends Status {
  /**
   * Open nodes after the step: the beam (best first); for best-first, by priority (highest first); for depth-first, a
   * stack with its top last; for breadth-first, a queue with its front first.
   */
  readonly frontier: readonly SearchVisit<N>[]
  /** The top k, best first (ties by evaluation order). */
  readonly results: readonly SearchVisit<N>[]
  /** The nodes this step expanded (beam search: the whole beam). */
  readonly expanded: readonly SearchVisit<N>[]
  /** The refinements this step generated, each with its fate. */
  readonly generated: readonly SearchVisit<N>[]
  /** Open nodes this step removed unexpanded because the threshold rose to their bound (not in beam search). */
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
  /** The id the next node evaluated will get. */
  readonly nextId: number
  /** The frontier is empty or `maxNodes` nodes have been evaluated: the search is over. */
  readonly terminated: boolean
}

/** The options of `refinementSearchSteps` with their defaults filled in. */
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

/**
 * The options with their defaults filled in, checked: throws `DomainError` for an unknown strategy, a `beamWidth` or
 * `k` that is not a positive integer, or a negative `maxDepth`.
 *
 * @param space The search space, read only for whether it has a `bound` (the default of `prune`).
 * @param o The caller's options.
 * @returns Every option, with its default where it was left out.
 */
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

/**
 * Results order, as a comparator for `sort`: higher quality first, then earlier evaluation.
 *
 * @param a A visit.
 * @param b Another visit.
 * @returns Negative when `a` ranks before `b`, positive when after, 0 for the same visit.
 */
const byQuality = <N>(a: SearchVisit<N>, b: SearchVisit<N>) => b.quality - a.quality || a.id - b.id

/**
 * `results` with `visit` offered: kept when its quality exceeds `minQuality` and it ranks in the top `k`. With a
 * redundancy test, a visit redundant with a better-or-equal result is refused, and results redundant with it and worse
 * are removed. Neither `results` nor `visit` is modified.
 *
 * @param results The current result set, best first (as this function returns it); not modified.
 * @param visit The evaluated node to offer.
 * @param k The most results to keep.
 * @param options `minQuality`, the quality a result must exceed (default $-\infty$), and `redundant`, the redundancy
 *   test, called as `redundant(visit.node, result.node)` (default none).
 * @returns The new result set, best first (ties by evaluation order), at most `k` long; `results` itself when the
 *   visit is refused.
 *
 * @example A result set of two that keeps no overlapping intervals
 * const visit = (id, node, quality) =>
 *   ({ id, parent: 0, depth: 1, node, key: String(node), quality, bound: Infinity, fate: 'queued' })
 * const overlap = (a, b) => a[0] < b[1] && b[0] < a[1]
 * const r1 = offerResult([], visit(1, [0, 4], 0.9), 2, { redundant: overlap })
 * const r2 = offerResult(r1, visit(2, [6, 9], 0.5), 2, { redundant: overlap })
 * print('two disjoint intervals:', r2.map((v) => v.node))
 * const r3 = offerResult(r2, visit(3, [2, 5], 0.7), 2, { redundant: overlap })
 * print('[2, 5] overlaps the better [0, 4]:', r3.map((v) => v.node))
 * const r4 = offerResult(r3, visit(4, [5, 8], 0.8), 2, { redundant: overlap })
 * print('[5, 8] overlaps the worse [6, 9]:', r4.map((v) => v.node))
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

/**
 * The top `k` of `visits` with the redundancy filter applied greedily from the best down: a visit is kept unless it is
 * redundant with one already kept.
 *
 * @param visits The visits to filter, in any order; not modified.
 * @param redundant The redundancy test, called as `redundant(candidate.node, kept.node)`.
 * @param k The most visits to keep (default all).
 * @returns The visits kept, best first (ties by evaluation order).
 *
 * @example Keep the best of each group of overlapping intervals
 * const visit = (id, node, quality) =>
 *   ({ id, parent: 0, depth: 1, node, key: String(node), quality, bound: Infinity, fate: 'queued' })
 * const overlap = (a, b) => a[0] < b[1] && b[0] < a[1]
 * const visits = [
 *   visit(1, [0, 4], 0.9),
 *   visit(2, [2, 5], 0.7),
 *   visit(3, [5, 8], 0.8),
 *   visit(4, [6, 9], 0.5),
 *   visit(5, [9, 12], 0.4),
 * ]
 * print('filtered:', filterRedundant(visits, overlap).map((v) => v.node))
 * print('filtered, top 2:', filterRedundant(visits, overlap, 2).map((v) => v.node))
 */
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

/**
 * The quality a node's bound must exceed to be explored: the $k$-th best quality once $k$ results are held (and at
 * least `minQuality`), else `minQuality`.
 *
 * @param results The current result set, best first.
 * @param o The resolved options, for `k` and `minQuality`.
 * @returns The threshold.
 */
const thresholdOf = <N>(results: readonly SearchVisit<N>[], o: Resolved<N>) =>
  results.length >= o.k ? Math.max(o.minQuality, results[o.k - 1].quality) : o.minQuality

/**
 * Search a refinement space step by step (see the file comment). Each step of beam search refines a whole level;
 * each step of the other strategies takes one node from the frontier and expands it, or discards it when its bound no
 * longer beats the threshold. The state's `results` are the top $k$. A node reached a second time (by its key) is
 * recorded as a duplicate and not evaluated again. The run ends (`terminated`) when the frontier is empty or
 * `maxNodes` nodes have been evaluated. Throws `DomainError` for invalid options, when the algorithm is built.
 *
 * @param space The search space: its root, refinement operator, quality and, optionally, an optimistic estimate,
 *   canonical key and expandability test. Its functions are called as the search runs.
 * @param options The strategy, the beam width, the depth and size limits, the number of results, pruning and the
 *   redundancy filter.
 * @returns The algorithm, started with `undefined`, to step with `run` or `trace`.
 *
 * @example Beam search keeps the two best subsets of each size
 * // Subsets of five weighted items, each grown by adding a later item; a subset's quality is its total weight.
 * const w = [5, -2, 4, 3, -1]
 * const total = (s) => s.reduce((a, i) => a + w[i], 0)
 * const after = (s) => (s.length ? s[s.length - 1] + 1 : 0)
 * const space = { root: [], refine: (s) => w.slice(after(s)).map((_, j) => [...s, after(s) + j]), quality: total }
 * const tr = trace(refinementSearchSteps(space, { strategy: 'beam', beamWidth: 2, k: 1 }), undefined, 10)
 * for (const s of tr.steps) print(`step ${s.t}: beam =`, s.frontier.map((v) => v.node), 'best =', s.best)
 *
 * @example Best-first branch and bound discards what cannot beat the best
 * // Subsets of five weighted items, each grown by adding a later item; a subset's quality is its total weight.
 * const w = [5, -2, 4, 3, -1]
 * const total = (s) => s.reduce((a, i) => a + w[i], 0)
 * const after = (s) => (s.length ? s[s.length - 1] + 1 : 0)
 * // No refinement can gain more than the positive weights still to come.
 * const bound = (s) => total(s) + w.slice(after(s)).reduce((a, x) => a + Math.max(x, 0), 0)
 * const refine = (s) => w.slice(after(s)).map((_, j) => [...s, after(s) + j])
 * const space = { root: [], refine, quality: total, bound }
 * const tr = trace(refinementSearchSteps(space, { strategy: 'best-first', k: 1 }), undefined, 100)
 * for (const s of tr.steps.slice(1)) {
 *   print(`step ${s.t}: expanded`, s.expanded.map((v) => v.node), 'discarded', s.discarded.map((v) => v.node))
 *   print('  best =', s.best, 'threshold =', s.threshold)
 * }
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

/**
 * The final state of `refinementSearchSteps`, run until it terminates: its `results` are the top $k$, with counts of
 * the nodes evaluated, expanded and pruned. Throws `DomainError` for invalid options.
 *
 * @param space The search space, as `refinementSearchSteps` takes it.
 * @param options The search options, as `refinementSearchSteps` takes them.
 * @returns The state the search ended in.
 *
 * @example The best subsets, with and without branch and bound
 * // Subsets of five weighted items, each grown by adding a later item; a subset's quality is its total weight.
 * const w = [5, -2, 4, 3, -1]
 * const total = (s) => s.reduce((a, i) => a + w[i], 0)
 * const after = (s) => (s.length ? s[s.length - 1] + 1 : 0)
 * const space = {
 *   root: [],
 *   refine: (s) => w.slice(after(s)).map((_, j) => [...s, after(s) + j]),
 *   quality: total,
 *   // No refinement can gain more than the positive weights still to come.
 *   bound: (s) => total(s) + w.slice(after(s)).reduce((a, x) => a + Math.max(x, 0), 0),
 * }
 * for (const prune of [false, true]) {
 *   const s = refinementSearch(space, { strategy: 'depth-first', k: 3, prune })
 *   print(`prune ${prune}: top 3 =`, s.results.map((v) => v.node), 'qualities', s.results.map((v) => v.quality))
 *   print('  evaluated =', s.evaluated, 'pruned =', s.pruned)
 * }
 */
export function refinementSearch<N>(space: SearchSpace<N>, options: SearchOptions<N> = {}): SearchState<N> {
  return run(refinementSearchSteps(space, options), undefined, Infinity)
}

/** Every node a search evaluated, by id, with its final fate, and the step at which each was generated and expanded. */
export interface SearchHistory<N> {
  /**
   * The visits indexed by id, each with its final fate: `expanded` if it was, `pruned` if it was discarded from the
   * frontier, else the fate it was generated with. Duplicates are left out.
   */
  readonly visits: readonly SearchVisit<N>[]
  /** Step that generated each visit (0 for the root). */
  readonly generatedAt: Int32Array
  /** Step that expanded each visit, $-1$ if never. */
  readonly expandedAt: Int32Array
  /** Step that discarded each visit from the frontier (its bound fell to the threshold), $-1$ if never. */
  readonly discardedAt: Int32Array
}

/**
 * The search tree recorded by a sequence of states (`trace(...).steps`, every step kept).
 *
 * @param states The states of a run of `refinementSearchSteps` in order, from the initial state on; a step left out
 *   loses the nodes it generated.
 * @returns The visits by id with their final fates, and the step at which each was generated, expanded and discarded.
 *
 * @example The tree a narrow beam search grew
 * const w = [5, -2, 4, 3, -1]
 * const total = (s) => s.reduce((a, i) => a + w[i], 0)
 * const after = (s) => (s.length ? s[s.length - 1] + 1 : 0)
 * const space = { root: [], refine: (s) => w.slice(after(s)).map((_, j) => [...s, after(s) + j]), quality: total }
 * const tr = trace(refinementSearchSteps(space, { strategy: 'beam', beamWidth: 1, maxDepth: 2, k: 1 }), undefined, 10)
 * const h = searchHistory(tr.steps)
 * print('nodes =', h.visits.map((v) => v.node))
 * print('fates =', h.visits.map((v) => v.fate))
 * print('generated at step', h.generatedAt)
 * print('expanded at step', h.expandedAt)
 */
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
