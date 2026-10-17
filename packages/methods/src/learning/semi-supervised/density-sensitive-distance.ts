/**
 * The density-sensitive $\rho$-path distance of Chapelle and Zien (2005, §2.1.2 and Algorithm 1).
 *
 * The points are the nodes of a neighbour graph whose edges $(i, j)$ have Euclidean lengths $d_{ij}$. An edge costs
 * $e^{\rho d_{ij}} - 1$, a path costs the sum of its edge costs, and the $\rho$-path distance between two nodes is
 *
 * $$
 * D_\rho(i, j) = \frac{1}{\rho}\log\Bigl(1 + \min_{\text{paths } i \to j}
 * \sum_{\text{edges}} (e^{\rho d} - 1)\Bigr).
 * $$
 *
 * As $\rho \to 0$ each edge cost is $\rho d$ to first order and $D_\rho$ tends to the shortest-path length (the
 * Euclidean distance on a full graph). As $\rho \to \infty$ the sum is dominated by its longest edge and $D_\rho$
 * tends to the minimax path distance $\min_{\text{paths}}\max_{\text{edges}} d$ of the connectivity kernel. Large
 * $\rho$ therefore shrinks distances within a dense cluster, where every path can use short edges, and keeps distances
 * across a gap, which every path must cross with a long edge. Shortest paths come from Dijkstra's algorithm from each
 * source. `rhoPathExtend` gives the distance from new query points by one extra edge from the query to each of its
 * graph neighbours. $\rho = 0$ is computed as the plain shortest-path length. In float64 an edge cost overflows once
 * $\rho d$ exceeds about 709; a reachable point whose path cost overflows is an error, never a silent Infinity.
 *
 * Chapelle and Zien square $D_\rho$ and use it in a kernel or an MDS embedding; this file returns $D_\rho$ itself.
 * Pure functions of plain arrays, so a worker can run them.
 */

import { DomainError, NumericalError } from 'aifn-compute/foundation/errors'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { fromEdges, unionFind, unionFindRoot, unite, type Graph } from 'aifn-compute/graph'
import { dijkstra } from 'aifn-compute/graph/shortest-paths'
import { kNearestNeighbourGraph } from 'aifn-compute/graph/structures'
import { bruteForceNeighbours } from 'aifn-compute/numerics/neighbours'

const OP = 'rhoPathDistances'

/** Rows of points: an array of rows, each of length $d$. */
export type PointRows = readonly (readonly number[])[]

/**
 * The cost $e^{\rho d} - 1$ of an edge of Euclidean length $d$ (just $d$ when $\rho = 0$).
 *
 * @param d The edge's length.
 * @param rho The softening $\rho \ge 0$.
 * @returns The cost; Infinity once $\rho d$ exceeds about 709.
 *
 * @example Costs of a short and a long edge
 * print([0, 1, 4].map((rho) => [rhoEdgeCost(0.4, rho), rhoEdgeCost(1, rho)]))
 */
export function rhoEdgeCost(d: number, rho: number): number {
  return rho > 0 ? Math.expm1(rho * d) : d
}

/**
 * The $\rho$-path distance $\frac{1}{\rho}\log(1 + S)$ from a path cost $S$ (just $S$ when $\rho = 0$).
 *
 * @param cost The path cost $S$, a sum of edge costs.
 * @param rho The softening $\rho \ge 0$.
 * @returns $D_\rho$.
 *
 * @example One edge's cost maps back to its length
 * print(rhoDistance(rhoEdgeCost(0.4, 3), 3))
 */
export function rhoDistance(cost: number, rho: number): number {
  return rho > 0 ? Math.log1p(cost) / rho : cost
}

/** Options of the $\rho$-path distances. */
export interface RhoPathOptions {
  /** The softening $\rho \ge 0$. */
  rho: number
  /** Neighbours per point in the graph (default every other point, the full graph of Chapelle and Zien). */
  neighbours?: number
  /**
   * The neighbour graph with Euclidean edge lengths, from an earlier result on the same points and `neighbours`, so
   * that a change of $\rho$ does not rebuild it.
   */
  graph?: Graph
}

/** $\rho$-path distances from a few source points to every point. */
export interface RhoPathDistances {
  /** The softening $\rho$. */
  rho: number
  /** The source nodes, in the order given. */
  sources: number[]
  /** For each source, the $\rho$-path distance to every point ($n$ values; Infinity when unreachable). */
  distances: Float64Array[]
  /** For each source, the minimal path cost $S$ to every point. */
  costs: Float64Array[]
  /** For each source, each point's predecessor on its cheapest path ($-1$ for the source and unreachable points). */
  predecessors: Int32Array[]
  /** The neighbour graph with the Euclidean edge lengths. */
  graph: Graph
  /** Neighbours per point used. */
  neighbours: number
}

/**
 * $\rho$-path distances from each source point to every point, by Dijkstra's algorithm (`dijkstra` of
 * `aifn-compute/graph/shortest-paths`) on the symmetric $k$-nearest-neighbour graph (`kNearestNeighbourGraph` of
 * `aifn-compute/graph/structures`, Euclidean edge lengths) with edge costs $e^{\rho d} - 1$. A point outside a
 * source's connected component is at distance Infinity. Throws `DomainError` for fewer than two points, a negative
 * $\rho$ or a source that is not a point, and `NumericalError` ('not-finite') when a reachable point's path cost
 * overflows (an edge with $\rho d$ above about 709): lower $\rho$ or rescale the points. About 110 ms per $\rho$ on
 * 200 points with the full graph; pass `graph` from an earlier result to reuse the graph.
 *
 * @param points The $n$ points, one row each.
 * @param sources The source nodes (indices into `points`), such as the labelled points.
 * @param options $\rho$, the number of neighbours and an optional graph to reuse.
 * @returns Distances, path costs and predecessors from each source, and the graph.
 *
 * @example Ten points on a line, labels at 0 and 4.2: distances to 2.4 as rho grows
 * const line = [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 3.4, 3.8, 4.2].map((v) => [v, 0])
 * for (const rho of [0, 1, 2, 60]) {
 *   const r = rhoPathDistances(line, [0, 9], { rho })
 *   print('rho', rho, ' from 0:', r.distances[0][6], ' from 4.2:', r.distances[1][6])
 * }
 */
export function rhoPathDistances(
  points: PointRows,
  sources: readonly number[],
  options: RhoPathOptions,
): RhoPathDistances {
  const n = points.length
  if (n < 2) throw new DomainError(OP, `${OP}: need at least two points`)
  if (!(options.rho >= 0)) throw new DomainError(OP, `${OP}: rho must be ≥ 0`)
  for (const s of sources)
    if (!Number.isInteger(s) || s < 0 || s >= n) throw new DomainError(OP, `${OP}: source ${s} is not a point`)
  const k = Math.min(options.neighbours ?? n - 1, n - 1)
  const graph = options.graph ?? kNearestNeighbourGraph(points, k)
  const weighted = fromEdges(
    n,
    graph.edges.map((e) => [e.from, e.to, rhoEdgeCost(e.weight ?? 1, options.rho)] as const),
    { directed: false },
  )
  const costs: Float64Array[] = []
  const predecessors: Int32Array[] = []
  // Components of the graph, to tell a point out of reach from a path cost that overflowed.
  const uf = unionFind(n)
  for (const e of graph.edges) unite(uf, e.from, e.to)
  for (const s of sources) {
    const paths = dijkstra(weighted, s)
    const cost = Float64Array.from(toFlat(paths.distance))
    for (let v = 0; v < n; v++)
      if (cost[v] === Infinity && unionFindRoot(uf, v) === unionFindRoot(uf, s))
        throw new NumericalError(
          OP,
          `${OP}: the path cost from ${s} to ${v} overflows at rho = ${options.rho}; lower rho or rescale the points`,
          'not-finite',
        )
    costs.push(cost)
    predecessors.push(Int32Array.from(toFlat(paths.predecessor)))
  }
  return {
    rho: options.rho,
    sources: [...sources],
    distances: costs.map((c) => c.map((v) => rhoDistance(v, options.rho))),
    costs,
    predecessors,
    graph,
    neighbours: k,
  }
}

/** The nearest points to each query: indices and Euclidean distances, nearest first. */
export interface QueryNeighbours {
  /** For each query, the indices of its nearest points, nearest first. */
  indices: number[][]
  /** For each query, the Euclidean distances to those points. */
  distances: number[][]
}

/**
 * The `neighbours` nearest points to each query (all points by default), by brute force (`bruteForceNeighbours` of
 * `aifn-compute/numerics/neighbours`); independent of $\rho$, so a figure computes them once for every $\rho$.
 *
 * @param points The graph's points.
 * @param queries The query points.
 * @param neighbours How many neighbours to keep per query (capped at the number of points).
 * @returns Indices and distances, one row per query.
 *
 * @example The two nearest of three points to a query
 * print(rhoQueryNeighbours([[0, 0], [1, 0], [3, 0]], [[0.8, 0]], 2))
 */
export function rhoQueryNeighbours(points: PointRows, queries: PointRows, neighbours = points.length): QueryNeighbours {
  const near = bruteForceNeighbours(points, queries, Math.min(neighbours, points.length))
  return { indices: toArray(near.indices) as number[][], distances: toArray(near.distances) as number[][] }
}

/**
 * $\rho$-path distances from the sources to new query points: each query joins the graph by an edge to each of its
 * nearest points (as many as each point has in the graph; all of them for the full graph), and its cost is the
 * cheapest of (edge cost + that point's path cost). Throws `NumericalError` ('not-finite') when that cost overflows
 * although an entry point is reachable.
 *
 * @param points The points the distances were computed on.
 * @param queries The query points, one row each.
 * @param result The output of `rhoPathDistances` on `points`.
 * @param near The queries' neighbours among the points (`rhoQueryNeighbours`), when computed once for several $\rho$.
 * @returns For each source, the distance to each query and the graph node through which its cheapest path enters.
 *
 * @example A query between the two labelled ends of a line
 * const line = [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 3.4, 3.8, 4.2].map((v) => [v, 0])
 * const r = rhoPathDistances(line, [0, 9], { rho: 2 })
 * const ext = rhoPathExtend(line, [[2.9, 0]], r)
 * print('distance from each label:', ext.distances.map((d) => d[0]), ' entering at:', ext.entry.map((e) => e[0]))
 */
export function rhoPathExtend(
  points: PointRows,
  queries: PointRows,
  result: RhoPathDistances,
  near: QueryNeighbours = rhoQueryNeighbours(
    points,
    queries,
    result.neighbours >= points.length - 1 ? points.length : result.neighbours,
  ),
): { distances: Float64Array[]; entry: Int32Array[] } {
  const m = queries.length
  const { indices: idx, distances: dist } = near
  const k = idx[0]?.length ?? 0
  const distances = result.costs.map(() => new Float64Array(m))
  const entry = result.costs.map(() => new Int32Array(m))
  result.costs.forEach((cost, s) => {
    for (let q = 0; q < m; q++) {
      let best = Infinity
      let via = -1
      let reachable = false
      for (let t = 0; t < k; t++) {
        if (cost[idx[q][t]] < Infinity) reachable = true
        const c = rhoEdgeCost(dist[q][t], result.rho) + cost[idx[q][t]]
        if (c < best) {
          best = c
          via = idx[q][t]
        }
      }
      if (best === Infinity && reachable)
        throw new NumericalError(
          'rhoPathExtend',
          `rhoPathExtend: the cost of query ${q} overflows at rho = ${result.rho}; lower rho or rescale the points`,
          'not-finite',
        )
      distances[s][q] = rhoDistance(best, result.rho)
      entry[s][q] = via
    }
  })
  return { distances, entry }
}

/**
 * The nodes of the cheapest path from source number `s` to `target`, read back from the predecessors (source
 * first); empty when the target is unreachable.
 *
 * @param result The output of `rhoPathDistances`.
 * @param s The position of the source in `result.sources`.
 * @param target The node to reach.
 * @returns The path's nodes, from the source to the target.
 *
 * @example The cheapest path from 0 to 2.4 walks the run of close points
 * const line = [0, 0.4, 0.8, 1.2, 1.6, 2, 2.4, 3.4, 3.8, 4.2].map((v) => [v, 0])
 * print(rhoPathNodes(rhoPathDistances(line, [0, 9], { rho: 1 }), 0, 6))
 */
export function rhoPathNodes(result: RhoPathDistances, s: number, target: number): number[] {
  const pred = result.predecessors[s]
  const source = result.sources[s]
  const out = [target]
  let v = target
  for (let steps = 0; v !== source; steps++) {
    v = pred[v]
    if (v < 0 || steps > pred.length) return []
    out.push(v)
  }
  return out.reverse()
}
