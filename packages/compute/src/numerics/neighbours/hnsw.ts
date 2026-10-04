/**
 * Hierarchical navigable small-world graphs (Malkov and Yashunin 2018, "Efficient and robust approximate nearest
 * neighbor search using hierarchical navigable small world graphs", IEEE TPAMI 42(4)). Every point gets a top layer
 * $\ell = \lfloor -\ln U \cdot m_L \rfloor$ with $U \sim \mathcal{U}(0, 1)$ and $m_L = 1/\ln M$, so layer $\ell$ holds about a fraction $M^{-\ell}$ of the points. Each layer
 * is a proximity graph on its points: a point inserted at layer $\ell$ links to up to $M$ of its nearest ($2M$ on layer $0$),
 * chosen by the neighbour-selection heuristic that keeps a candidate only when it is closer to the new point than to
 * every neighbour already kept, so links spread in direction (Algorithm 4).
 *
 * A search enters at the top layer's entry point and walks greedily to the nearest point it can reach on each layer
 * (beam width 1), then drops a layer and starts from there; on layer 0 it runs a beam search of width $ef \ge k$
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
  /** Discriminator kind. */
  readonly kind: 'hnsw-index'
  /** Total number of indexed points $n$. */
  readonly n: Size
  /** Dimensionality $d$. */
  readonly d: Size
  /** Flattened point data array ($n \times d$). */
  readonly data: Float64Array
  /** Distance metric used. */
  readonly metric: NeighbourMetric
  /** Links per point on layers $\ge 1$ ($M$) and on layer $0$ ($2M$). */
  readonly M: Size
  /** The top layer of every point. */
  readonly levels: Int32Array
  /** `links[layer][i]`: the neighbours of point $i$ on layer `layer` (empty when $i$ is not on layer `layer`). */
  readonly links: readonly (readonly (readonly number[])[])[]
  /** The entry point index on the top layer. */
  readonly entry: number
  /** Topmost layer index. */
  readonly topLayer: number
  /** Distances computed during construction. */
  readonly buildDistanceEvaluations: number
}

/** Options of {@link hnswIndex}. */
export interface HnswOptions {
  /** Random stream. */
  stream: Stream
  /** Links per point per layer (default 16; layer $0$ allows $2M$). */
  M?: Size
  /** Beam width while inserting (default 200, as hnswlib). */
  efConstruction?: Size
  /** The level multiplier $m_L$ (default $1/\ln M$). */
  levelMultiplier?: number
  /** Select neighbours by the heuristic (default `true`) or simply the $M$ nearest. */
  heuristic?: boolean
  /** Metric to evaluate distances with (default `'euclidean'`). */
  metric?: NeighbourMetric
}

type Scored = { i: number; d: number }

/**
 * Insert into an array sorted by distance (then index), keeping it sorted.
 *
 * @param list - Sorted candidate array to insert into.
 * @param e - Candidate point to insert.
 */
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
 * Beam search on one layer from the entry points (Algorithm 2): returns the $ef$ nearest found, nearest first, the points
 * expanded in order and every point whose distance was computed.
 *
 * @param s - Graph searcher providing distance and links.
 * @param q - Query coordinates array.
 * @param entries - Entry points to begin the search from.
 * @param ef - Beam search capacity.
 * @param layer - Layer index to search.
 * @returns Object with found candidates, expanded nodes, and visited set.
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

/**
 * Choose up to $m$ neighbours of a point from candidates sorted nearest first (Algorithm 4, or the $m$ nearest).
 *
 * @param candidates - Sorted candidates array.
 * @param m - Target number of neighbours.
 * @param heuristic - Whether to apply the diversity heuristic.
 * @param between - Function evaluating distance between two stored point indices.
 * @returns Selected neighbour indices.
 */
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

/**
 * Build an HNSW index by inserting the rows of $x$ in order (Algorithm 1).
 *
 * @param x - Input data matrix ($n \times d$).
 * @param options - Construction options including random stream.
 * @param options.stream - Random stream for layer assignment.
 * @param options.M - Number of bi-directional links per node (default 16).
 * @param options.efConstruction - Size of dynamic candidate list during construction (default 200).
 * @param options.levelMultiplier - Level multiplier $m_L$ (default $1/\ln M$).
 * @param options.heuristic - Whether to use the neighbour diversity heuristic (default `true`).
 * @param options.metric - Distance metric (default `'euclidean'`).
 * @returns Built HNSW index structure.
 *
 * @example Build an HNSW index
 * const s = stream(42)
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const index = hnswIndex(data, { stream: s, M: 4 })
 * print('Index points:', index.n)
 */
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
  /** Layer index. */
  readonly layer: number
  /** Point indices expanded in order on this layer. */
  readonly expanded: readonly number[]
  /** Point indices visited on this layer. */
  readonly visited: readonly number[]
  /** The nearest point found on the layer: the entry point of the next. */
  readonly nearest: number
}

/** The answer to one HNSW query with its descent through the layers, top first. */
export interface HnswQueryResult extends QueryResult {
  /** Trace of search descent across graph layers. */
  readonly layers: readonly HnswLayerTrace[]
}

/**
 * The $k$ nearest points found by the layered search with beam width $ef$ on layer $0$ (default $\max(k, 50)$).
 *
 * @param index - HNSW index to search.
 * @param query - Query vector of length $d$.
 * @param k - Number of nearest neighbours $k$ to return.
 * @param options - Search options.
 * @param options.ef - Beam width on layer 0 (default $\max(k, 50)$).
 * @returns Query result containing $k$ nearest indices, distances, and layer traces.
 *
 * @example Query nearest neighbours using HNSW
 * const s = stream(42)
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const index = hnswIndex(data, { stream: s, M: 4 })
 * const res = hnswQuery(index, [1.1, 0.9], 2)
 * print('Nearest index:', res.indices[0])
 */
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

/**
 * Search nearest neighbours across multiple queries using an HNSW index.
 *
 * @param index - HNSW index to search.
 * @param queries - Query points matrix ($m \times d$).
 * @param k - Number of nearest neighbours $k$ to return per query.
 * @param options - Search options.
 * @param options.ef - Beam width on layer 0 (default $\max(k, 50)$).
 * @returns Stacked `Neighbours` object.
 *
 * @example Search nearest neighbours with HNSW for multiple queries
 * const s = stream(42)
 * const data = [[0, 0], [1, 1], [2, 2], [3, 3]]
 * const queries = [[0.1, 0.1], [2.1, 2.1]]
 * const index = hnswIndex(data, { stream: s, M: 4 })
 * const res = hnswSearch(index, queries, 2)
 * print('Nearest indices:\n' + res.indices)
 */
export function hnswSearch(index: HnswIndex, queries: MatrixLike, k: Size, options: { ef?: Size } = {}): Neighbours {
  const Q = rowsOf(queries, 'hnswSearch')
  return stackResults(
    Array.from({ length: Q.n }, (_, i) => hnswQuery(index, Q.data.subarray(i * Q.d, (i + 1) * Q.d), k, options)),
    k,
  )
}
