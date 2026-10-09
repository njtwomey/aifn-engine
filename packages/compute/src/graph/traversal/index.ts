/**
 * `aifn-compute/graph/traversal`: traversals, orderings and components of the graphs of `aifn-compute/graph`.
 *
 * - Searches: `breadthFirstSearch` (visit order, depths, layers and the breadth-first forest), `depthFirstSearch`
 *   (discovery and finish times, edge classes and the depth-first forest), `iterativeDeepening` (a shortest path by
 *   depth-limited passes) and `unweightedShortestPaths` (hop counts and predecessors).
 * - Order and cycles: `topologicalSort` (Kahn's algorithm or reverse depth-first postorder), `isDag` and `findCycle`,
 *   which returns a witness cycle.
 * - Components: `connectedComponents` (by union–find, directions ignored), `stronglyConnectedComponents` (Tarjan or
 *   Kosaraju), `condensation` (the DAG of the strongly connected components) and `bipartite` (a two-colouring or an
 *   odd cycle).
 * - Step-through algorithms for figures and traces: `breadthFirstSteps`, `depthFirstSteps`, `iterativeDeepeningSteps`,
 *   `kahnSteps`, `tarjanSteps` and `kosarajuSteps`, each run with `run(alg, undefined, steps)`; one step examines one
 *   edge or finishes one node. `traversalAlgorithms` and `traversalFunctions` register them by key.
 *
 * Neighbours are visited in `adjacency` order, fixed by the edge list, and roots are tried in index order, so every
 * result is reproducible. Per-node results are int32 tensors of length $V$ with $-1$ for "none" (no parent, not
 * reached). Topological order and Kahn's algorithm need a directed graph and throw `DomainError` otherwise.
 */

export {
  breadthFirstSearch,
  breadthFirstSteps,
  depthFirstSearch,
  depthFirstSteps,
  iterativeDeepening,
  iterativeDeepeningSteps,
  unweightedShortestPaths,
  type BreadthFirstEvent,
  type BreadthFirstResult,
  type BreadthFirstState,
  type DepthFirstEvent,
  type DepthFirstResult,
  type DepthFirstState,
  type EdgeClass,
  type IterativeDeepeningEvent,
  type IterativeDeepeningOptions,
  type IterativeDeepeningResult,
  type IterativeDeepeningState,
  type TraversalOptions,
} from './traversal'
export { findCycle, isDag, kahnSteps, topologicalSort, type KahnState, type TopologicalOrder } from './order'
export {
  bipartite,
  condensation,
  connectedComponents,
  kosarajuSteps,
  stronglyConnectedComponents,
  tarjanSteps,
  type BipartiteResult,
  type Components,
  type KosarajuState,
  type TarjanEvent,
  type TarjanState,
} from './components'
export { traversalAlgorithms, traversalFunctions } from './registry'
