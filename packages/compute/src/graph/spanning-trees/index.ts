/**
 * `aifn-compute/graph/spanning-trees`: minimum spanning trees by Kruskal (union–find) and Prim (heap), with their steps.
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
