/**
 * `aifn-compute/graph/spanning-trees`: minimum spanning trees by Kruskal (union–find) and Prim (heap), with their
 * steps.
 *
 * - The answer: `minimumSpanningTree`, Kruskal's algorithm by default or Prim's with `method: 'prim'`; both give the
 *   same total weight.
 * - Step through: `kruskalSteps` (edges in order of weight, each accepted or rejected as closing a cycle) and
 *   `primSteps` (one tree grown from node 0 by its lightest leaving edge), for `run` and `trace`; `sameSet` asks a
 *   Kruskal state's forest whether two nodes are already joined.
 *
 * Edge directions are ignored and an unset weight counts as 1. On a disconnected graph the result is a minimum
 * spanning forest, one tree per connected component. Trees are edge indices into `graph.edges` (int32 tensors).
 */

export {
  kruskalSteps,
  minimumSpanningTree,
  primSteps,
  sameSet,
  type KruskalState,
  type PrimCandidate,
  type PrimState,
  type SpanningEvent,
  type SpanningTree,
} from './trees'
export { spanningTreesAlgorithms, spanningTreesFunctions } from './registry'
