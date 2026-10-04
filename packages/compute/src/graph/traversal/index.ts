/**
 * `aifn-compute/graph/traversal`: traversals and orderings of the graphs of `aifn-compute/graph`: breadth-first and depth-first search
 * (with steps), iterative deepening, unweighted shortest paths, cycles, topological order (Kahn), connected and
 * strongly connected components (Kosaraju, Tarjan), condensation and bipartiteness.
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
