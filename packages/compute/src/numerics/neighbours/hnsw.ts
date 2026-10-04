/**
 * Hierarchical navigable small-world graphs (Malkov and Yashunin 2018, "Efficient and robust approximate nearest
 * neighbor search using hierarchical navigable small world graphs", IEEE TPAMI 42(4)). Every point gets a top layer
 * ℓ = ⌊−ln U · m_L⌋ with U ~ U(0, 1) and m_L = 1/ln M, so layer ℓ holds about a fraction M^{−ℓ} of the points. Each layer
 * is a proximity graph on its points: a point inserted at layer ℓ links to up to M of its nearest (2M on layer 0),
 * chosen by the neighbour-selection heuristic that keeps a candidate only when it is closer to the new point than to
 * every neighbour already kept, so links spread in direction (Algorithm 4).
 *
 * A search enters at the top layer's entry point and walks greedily to the nearest point it can reach on each layer
 * (beam width 1), then drops a layer and starts from there; on layer 0 it runs a beam search of width ef ≥ k
 * (Algorithms 2 and 5). The long links of the sparse upper layers cross the space in a few hops; the dense bottom layer
 * refines. `trace` records each layer's expanded path and every point whose distance was computed.
 */

import type { MatrixLike, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, uniform, type Stream } from 'aifn-compute/foundation/random'
import {
  checkK,
  distanceOf,
  queryOf,
  rowsOf,
  stackResults,
  type Neighbours,
  type NeighbourMetric,
  type QueryResult,
} from './search'

/** A built HNSW index. */
export interface HnswIndex {
  readonly kind: 'hnsw-index'
  readonly n: Size
  readonly d: Size
  readonly data: Float64Array
  readonly metric: NeighbourMetric
  /** Links per point on layers ≥ 1 (M) and on layer 0 (2M). */
  readonly M: Size
  /** The top layer of every point. */
  readonly levels: Int32Array
  /** links[ℓ][i]: the neighbours of point i on layer ℓ (empty when i is not on layer ℓ). */
  readonly links: readonly (readonly (readonly number[])[])[]
  /** The entry point: a point on the top layer. */
  readonly entry: number
  readonly topLayer: number
  /** Distances computed during construction. */
  readonly buildDistanceEvaluations: number
}

/** Options of {@link hnswIndex}. */
export interface HnswOptions {
  stream: Stream
  /** Links per point per layer (default 16; layer 0 allows 2M). */
  M?: Size
  /** Beam width while inserting (default 200, as hnswlib). */
  efConstruction?: Size
  /** The level multiplier m_L (default 1/ln M). */
  levelMultiplier?: number
  /** Select neighbours by the heuristic (default true) or simply the M nearest. */
  heuristic?: boolean
  /** Default `euclidean`. */
  metric?: NeighbourMetric
}

type Scored = { i: number; d: number }

/** Insert into an array sorted by distance (then index), keeping it sorted. */
function insertSorted(list: Scored[], e: Scored): void {
  let lo = 0
  let hi = list.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (list[mid].d < e.d || (list[mid].d === e.d && list[mid].i < e.i)) lo = mid + 1
    else hi = mid
  }
  list.splice(lo, 0, e)
}

interface Searcher {
  dist: (a: ArrayLike<number>, j: number) => number
  links: number[][][]
}

/**
 * Beam search on one layer from the entry points (Algorithm 2): returns the ef nearest found, nearest first, the points
 * expanded in order and every point whose distance was computed.
 */
function searchLayer(
  s: Searcher,
  q: ArrayLike<number>,
  entries: readonly Scored[],
  ef: number,
  layer: number,
): { found: Scored[]; expanded: number[]; visited: number[] } {
  const visited = new Set<number>(entries.map((e) => e.i))
  const candidates: Scored[] = []
  const found: Scored[] = []
  for (const e of entries) {
    insertSorted(candidates, e)
    insertSorted(found, e)
  }
  while (found.length > ef) found.pop()
  const expanded: number[] = []
  while (candidates.length) {
    const c = candidates.shift()!
    if (c.d > found[found.length - 1].d) break
    expanded.push(c.i)
    for (const e of s.links[layer][c.i]) {
      if (visited.has(e)) continue
      visited.add(e)
      const de = s.dist(q, e)
      if (found.length < ef || de < found[found.length - 1].d) {
        insertSorted(candidates, { i: e, d: de })
        insertSorted(found, { i: e, d: de })
        if (found.length > ef) found.pop()
      }
    }
  }
  return { found, expanded, visited: [...visited] }
}

/** Choose up to m neighbours of a point from candidates sorted nearest first (Algorithm 4, or the m nearest). */
function selectNeighbours(
  candidates: readonly Scored[],
  m: number,
  heuristic: boolean,
  between: (a: number, b: number) => number,
): number[] {
  if (!heuristic) return candidates.slice(0, m).map((c) => c.i)
  const kept: Scored[] = []
  for (const c of candidates) {
    if (kept.length >= m) break
    if (kept.every((r) => c.d < between(c.i, r.i))) kept.push(c)
  }
  return kept.map((c) => c.i)
}

/** Build an HNSW index by inserting the rows of x in order (Algorithm 1). */
export function hnswIndex(x: MatrixLike, options: HnswOptions): HnswIndex {
  const { n, d, data } = rowsOf(x, 'hnswIndex')
  const { M = 16, efConstruction = 200, heuristic = true, metric = 'euclidean' } = options
  if (!(Number.isInteger(M) && M >= 2)) throw new DomainError('hnswIndex', 'hnswIndex: M must be an integer ≥ 2')
  const mL = options.levelMultiplier ?? 1 / Math.log(M)
  let evaluations = 0
  const dist = (a: ArrayLike<number>, j: number) => {
    evaluations++
    return distanceOf(metric, a, 0, data, j, d)
  }
  const between = (a: number, b: number) => {
    evaluations++
    return distanceOf(metric, data, a, data, b, d)
  }
  const levels = new Int32Array(n)
  for (let i = 0; i < n; i++) levels[i] = Math.floor(-Math.log(1 - uniform(child(options.stream, 'level', i))) * mL)
  // A loop, not Math.max(...levels): spreading a large array overflows the call stack.
  let top = 0
  for (const l of levels) top = Math.max(top, l)
  const links: number[][][] = Array.from({ length: top + 1 }, () => Array.from({ length: n }, () => []))
  const s: Searcher = { dist, links }
  let entry = -1
  let topLayer = -1
  for (let i = 0; i < n; i++) {
    const q = data.subarray(i * d, (i + 1) * d)
    const level = levels[i]
    if (entry < 0) {
      entry = i
      topLayer = level
      continue
    }
    let eps: Scored[] = [{ i: entry, d: dist(q, entry) }]
    for (let layer = topLayer; layer > level; layer--) eps = searchLayer(s, q, eps, 1, layer).found.slice(0, 1)
    for (let layer = Math.min(topLayer, level); layer >= 0; layer--) {
      const { found } = searchLayer(s, q, eps, efConstruction, layer)
      const mMax = layer === 0 ? 2 * M : M
      const chosen = selectNeighbours(found, M, heuristic, between)
      links[layer][i] = chosen
      for (const e of chosen) {
        const list = links[layer][e]
        list.push(i)
        if (list.length > mMax) {
          const scored = list.map((j) => ({ i: j, d: between(e, j) })).sort((a, b) => a.d - b.d || a.i - b.i)
          links[layer][e] = selectNeighbours(scored, mMax, heuristic, between)
        }
      }
      eps = found
    }
    if (level > topLayer) {
      entry = i
      topLayer = level
    }
  }
  return {
    kind: 'hnsw-index',
    n,
    d,
    data: Float64Array.from(data),
    metric,
    M,
    levels,
    links,
    entry,
    topLayer: Math.max(topLayer, 0),
    buildDistanceEvaluations: evaluations,
  }
}

/** One layer of an HNSW search: the points expanded in order (the greedy path on upper layers) and those evaluated. */
export interface HnswLayerTrace {
  readonly layer: number
  readonly expanded: readonly number[]
  readonly visited: readonly number[]
  /** The nearest point found on the layer: the entry point of the next. */
  readonly nearest: number
}

/** The answer to one HNSW query with its descent through the layers, top first. */
export interface HnswQueryResult extends QueryResult {
  readonly layers: readonly HnswLayerTrace[]
}

/** The k nearest points found by the layered search with beam width ef on layer 0 (default max(k, 50)). */
export function hnswQuery(index: HnswIndex, query: VectorLike, k: Size, options: { ef?: Size } = {}): HnswQueryResult {
  const q = queryOf(query, index.d, 'hnswQuery')
  checkK(k, index.n, 'hnswQuery')
  let evaluations = 0
  const s: Searcher = {
    dist: (a, j) => {
      evaluations++
      return distanceOf(index.metric, a, 0, index.data, j, index.d)
    },
    links: index.links as number[][][],
  }
  const ef = Math.max(k, options.ef ?? 50)
  let eps: Scored[] = [{ i: index.entry, d: s.dist(q, index.entry) }]
  const layers: HnswLayerTrace[] = []
  for (let layer = index.topLayer; layer >= 0; layer--) {
    const r = searchLayer(s, q, eps, layer === 0 ? ef : 1, layer)
    layers.push({ layer, expanded: r.expanded, visited: r.visited, nearest: r.found[0].i })
    eps = layer === 0 ? r.found : r.found.slice(0, 1)
  }
  const top = eps.slice(0, k)
  return { indices: top.map((e) => e.i), distances: top.map((e) => e.d), distanceEvaluations: evaluations, layers }
}

/** {@link hnswQuery} for every row of `queries`. */
export function hnswSearch(index: HnswIndex, queries: MatrixLike, k: Size, options: { ef?: Size } = {}): Neighbours {
  const Q = rowsOf(queries, 'hnswSearch')
  return stackResults(
    Array.from({ length: Q.n }, (_, i) => hnswQuery(index, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k, options)),
    k,
  )
}
