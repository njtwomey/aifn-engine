/**
 * `aifn-compute/graph/shortest-paths`: weighted shortest paths on the graphs of `aifn-compute/graph`, single-source
 * and all-pairs.
 *
 * - Single source, non-negative weights: `dijkstra` (all distances, or stop at a target) and `aStar` (one path to a
 *   target, guided by a heuristic, with the number of expansions). Both throw `DomainError` for a negative weight.
 * - Negative weights: `bellmanFord` (single source, reports a reachable negative cycle) and `floydWarshall` (all
 *   pairs, $V \times V$ distances and predecessors, flags a negative cycle).
 * - Paths: `shortestPath` reads a path back from predecessors (a row of them for Floyd–Warshall); `directedArcs`
 *   lists the arcs Bellman–Ford and Floyd–Warshall relax.
 * - Step-through algorithms for figures and traces: `dijkstraSteps`, `aStarSteps`, `bellmanFordSteps` and
 *   `floydWarshallSteps`, run with `run(alg, undefined, steps)`; `shortestPathsAlgorithms` and
 *   `shortestPathsFunctions` register them by key.
 *
 * A path's length is the sum of its edge weights (1 for an edge without one), and an undirected edge can be used both
 * ways. Distances are float64 with Infinity where a node is unreachable; predecessors are int32 with $-1$ for none,
 * as in `scipy.sparse.csgraph`. Unweighted (hop-count) distances are `unweightedShortestPaths` of
 * `aifn-compute/graph/traversal`.
 */

export {
  aStar,
  aStarSteps,
  bellmanFord,
  bellmanFordSteps,
  dijkstra,
  dijkstraSteps,
  directedArcs,
  floydWarshall,
  floydWarshallSteps,
  shortestPath,
  type BellmanFordState,
  type DijkstraState,
  type DirectedArc,
  type FloydWarshallState,
  type ShortestPathOptions,
  type ShortestPaths,
} from './paths'
export { shortestPathsAlgorithms, shortestPathsFunctions } from './registry'
