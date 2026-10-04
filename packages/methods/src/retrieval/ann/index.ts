/**
 * `aifn-methods/retrieval/ann`: a recall-against-speed benchmark of the nearest-neighbour indexes of
 * `aifn-compute/numerics/neighbours` (k-d tree, LSH, IVF, PQ, HNSW), streamed method by method.
 */

export {
  ANN_METHODS,
  annBenchmark,
  type AnnBenchmarkOptions,
  type AnnBenchmarkSnapshot,
  type AnnCurve,
  type AnnMethod,
  type AnnPoint,
} from './benchmark'
export { annFunctions } from './registry'
