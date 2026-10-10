/**
 * `aifn-methods/retrieval/ann`: a recall-against-speed benchmark of the nearest-neighbour indexes of
 * `aifn-compute/numerics/neighbours` (k-d tree, LSH, IVF, PQ, HNSW), streamed method by method.
 *
 * - Benchmarking: `annBenchmark`, a generator that runs each index of `ANN_METHODS` at several settings of its speed
 *   knob and yields an `AnnBenchmarkSnapshot` after each, holding one `AnnCurve` of `AnnPoint`s (recall@$k$, queries
 *   per second, distance evaluations per query) per index, in the manner of ann-benchmarks. IVF, PQ and HNSW are the
 *   index families FAISS offers.
 * - Registry: `annFunctions` registers the benchmark.
 *
 * Recall and distance counts are reproducible from the seed; the timings are the machine's.
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
