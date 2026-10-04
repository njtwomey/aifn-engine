/**
 * `aifn-compute/numerics/neighbours`: nearest-neighbour search, exact and approximate, as scikit-learn's `neighbors` and faiss.
 *
 * - Exact: `bruteForceNeighbours` (the reference), the $k$-d, ball and vantage-point trees (`kdTree`, `ballTree`, `vpTree`, `treeQuery` with
 *   its visit order, `treeSearch`).
 * - Hashing: random-hyperplane and $p$-stable families (`hyperplaneFamily`, `pStableFamily`), `lshIndex`, `lshQuery`, and
 *   the banding shared with MinHash (`lshBands`, `lshCandidates`, `lshProbability`, `lshThreshold`).
 * - Graphs: `nearestNeighbourDescent` ($k$-NN graphs), HNSW (`hnswIndex`, `hnswQuery` with its layer-by-layer trace).
 * - Quantisation: codebooks (`kmeansPlusPlus`, `assignNearest`, `lloydUpdate`, `trainCodebook`), the inverted file
 *   (`ivfIndex`, `ivfQuery`), product quantisation and OPQ (`productQuantiser`, `optimisedProductQuantiser`,
 *   `pqEncode`, `pqDecode`, `pqDistanceTable`, `pqQuery`).
 * - Measurement: `searchRecall`, `benchmarkSearch`.
 */

export {
  bruteForceNeighbours,
  bruteForceQuery,
  kBest,
  type KBest,
  type BruteForceOptions,
  type NeighbourMetric,
  type Neighbours,
  type QueryResult,
} from './search'
export {
  ballTree,
  kdTree,
  vpTree,
  nodeLowerBound,
  treeQuery,
  treeSearch,
  type SpaceTree,
  type SpaceTreeNode,
  type SpaceTreeOptions,
  type TreeMetric,
  type TreeQueryResult,
  type TreeVisit,
} from './trees'
export {
  hyperplaneFamily,
  lshBands,
  lshCandidates,
  lshCollisionProbability,
  lshIndex,
  lshProbability,
  lshQuery,
  lshSearch,
  lshSignatures,
  lshThreshold,
  pStableFamily,
  type Banding,
  type LshFamily,
  type LshFamilyOptions,
  type LshIndex,
  type LshQueryResult,
} from './lsh'
export { nearestNeighbourDescent, type NearestNeighbourDescentOptions, type NeighbourLists } from './nn-descent'
export {
  assignNearest,
  kmeansPlusPlus,
  lloydUpdate,
  trainCodebook,
  type Assignment,
  type Codebook,
  type CodebookOptions,
  type KMeansPlusPlus,
  type LloydUpdate,
} from './codebook'
export {
  ivfIndex,
  ivfQuery,
  ivfSearch,
  optimisedProductQuantiser,
  pqDecode,
  pqDistanceTable,
  pqEncode,
  pqQuery,
  pqSearch,
  productQuantiser,
  type IvfIndex,
  type IvfOptions,
  type IvfQueryResult,
  type OpqOptions,
  type OpqResult,
  type ProductQuantiser,
  type ProductQuantiserOptions,
} from './quantisation'
export {
  hnswIndex,
  hnswQuery,
  hnswSearch,
  type HnswIndex,
  type HnswLayerTrace,
  type HnswOptions,
  type HnswQueryResult,
} from './hnsw'
export { benchmarkSearch, searchRecall, type SearchBenchmark } from './benchmark'
export { neighbourFunctions } from './registry'
