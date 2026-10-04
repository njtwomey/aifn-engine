/** The functions of `aifn-compute/numerics/neighbours`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as benchmark from './benchmark'
import * as codebook from './codebook'
import * as hnsw from './hnsw'
import * as lsh from './lsh'
import * as nnDescent from './nn-descent'
import * as quantisation from './quantisation'
import * as search from './search'
import * as trees from './trees'

const fn = definer<FunctionInfo>('function', 'numerics/neighbours')
const SEARCH = ['nearest-neighbour-search', 'k-nearest-neighbours']
const ANN = ['approximate-nearest-neighbour-search']

fn(
  {
    key: 'bruteForceNeighbours',
    name: 'Exact k nearest neighbours by brute force',
    summary: 'Scan every stored point for each query: O(nmd), the reference every index is measured against.',
    role: 'solver',
    notes: SEARCH,
  },
  search.bruteForceNeighbours,
)
fn(
  { key: 'bruteForceQuery', name: 'Exact k nearest neighbours of one query', role: 'solver', notes: SEARCH },
  search.bruteForceQuery,
)
fn(
  {
    key: 'kdTree',
    name: 'k-d tree',
    summary: 'Median splits on the coordinate of largest spread, with a bounding box at every node.',
    role: 'construction',
    notes: ['k-d-tree', 'space-partitioning-trees', 'tree-based-approximate-nearest-neighbours'],
    cite: ['bentley1975', 'friedman1977'],
  },
  trees.kdTree,
)
fn(
  {
    key: 'ballTree',
    name: 'Ball tree',
    summary: 'The k-d splits with a centroid and covering radius at every node, for bounds in any metric.',
    role: 'construction',
    notes: ['ball-tree', 'space-partitioning-trees'],
    cite: ['omohundro1989', 'uhlmann1991'],
  },
  trees.ballTree,
)
fn(
  {
    key: 'vpTree',
    name: 'Vantage-point tree',
    summary:
      'Splits each node at the median distance from a vantage point; children keep the shell of distances that holds them.',
    role: 'construction',
    notes: ['vantage-point-tree', 'space-partitioning-trees'],
    cite: ['yianilos1993'],
  },
  trees.vpTree,
)
fn(
  {
    key: 'treeQuery',
    name: 'Branch-and-bound tree search',
    summary:
      'Exact k-NN by depth-first descent, nearer child first, pruning nodes whose lower bound exceeds the k-th best.',
    role: 'solver',
    notes: ['nearest-neighbour-search'],
    cite: ['friedman1977'],
  },
  trees.treeQuery,
)
fn(
  { key: 'treeSearch', name: 'Tree search over many queries', role: 'solver', notes: ['nearest-neighbour-search'] },
  trees.treeSearch,
)
fn(
  { key: 'nodeLowerBound', name: 'Lower bound of a tree node', role: 'property', notes: ['nearest-neighbour-search'] },
  trees.nodeLowerBound,
)
fn(
  {
    key: 'hyperplaneFamily',
    name: 'Random-hyperplane LSH',
    tex: 'h(\\xvec) = [\\mathbf{r}^\\top\\xvec \\ge 0]',
    summary: 'Sign-of-projection hashes for the cosine distance; vectors at angle θ collide with probability 1 − θ/π.',
    role: 'construction',
    random: true,
    notes: ['locality-sensitive-hashing'],
    cite: ['charikar2002'],
  },
  lsh.hyperplaneFamily,
)
fn(
  {
    key: 'pStableFamily',
    name: 'p-stable LSH',
    tex: 'h(\\xvec) = \\lfloor (\\mathbf{a}^\\top\\xvec + b)/w \\rfloor',
    summary: 'Quantised Gaussian projections for the Euclidean distance.',
    role: 'construction',
    random: true,
    notes: ['locality-sensitive-hashing'],
    cite: ['datar2004'],
  },
  lsh.pStableFamily,
)
fn(
  { key: 'lshSignatures', name: 'LSH signatures', role: 'transform', notes: ['locality-sensitive-hashing'] },
  lsh.lshSignatures,
)
fn(
  {
    key: 'lshCollisionProbability',
    name: 'LSH collision probability',
    summary: 'The chance one hash of a family puts two points at a given distance (or angle) in one bucket.',
    role: 'property',
    notes: ['locality-sensitive-hashing'],
    cite: ['charikar2002', 'datar2004'],
  },
  lsh.lshCollisionProbability,
)
fn(
  {
    key: 'lshIndex',
    name: 'LSH index',
    role: 'construction',
    notes: ['locality-sensitive-hashing'],
    cite: ['indyk1998'],
  },
  lsh.lshIndex,
)
fn(
  {
    key: 'lshQuery',
    name: 'LSH query',
    summary: 'Rerank, by exact distance, the points sharing a bucket with the query in any table.',
    role: 'solver',
    notes: ['locality-sensitive-hashing'],
  },
  lsh.lshQuery,
)
fn(
  { key: 'lshSearch', name: 'LSH search over many queries', role: 'solver', notes: ['locality-sensitive-hashing'] },
  lsh.lshSearch,
)
fn(
  {
    key: 'lshBands',
    name: 'LSH bands of a signature',
    summary: 'The bucket key of each band of b·r hashes.',
    role: 'transform',
    notes: ['locality-sensitive-hashing', 'character-n-grams-and-shingles'],
    cite: ['indyk1998'],
  },
  lsh.lshBands,
)
fn(
  {
    key: 'lshCandidates',
    name: 'LSH candidate pairs',
    summary: 'Every pair of rows sharing at least one band bucket.',
    role: 'solver',
    notes: ['locality-sensitive-hashing', 'character-n-grams-and-shingles'],
    cite: ['indyk1998', 'broder1997'],
  },
  lsh.lshCandidates,
)
fn(
  {
    key: 'lshProbability',
    name: 'Banding S-curve',
    tex: '1 - (1 - p^r)^b',
    summary: 'The probability that two items become candidates under b bands of r rows.',
    role: 'property',
    notes: ['locality-sensitive-hashing', 'character-n-grams-and-shingles'],
    cite: ['indyk1998'],
  },
  lsh.lshProbability,
)
fn(
  {
    key: 'lshThreshold',
    name: 'Banding threshold',
    tex: '(1/b)^{1/r}',
    role: 'property',
    notes: ['locality-sensitive-hashing'],
  },
  lsh.lshThreshold,
)
fn(
  {
    key: 'nearestNeighbourDescent',
    name: 'NN-descent',
    summary: 'Approximate k-nearest-neighbour graph by local joins of neighbours of neighbours.',
    role: 'solver',
    random: true,
    notes: ['graph-based-approximate-nearest-neighbours', 'uniform-manifold-approximation-and-projection'],
  },
  nnDescent.nearestNeighbourDescent,
)
fn(
  {
    key: 'kmeansPlusPlus',
    name: 'k-means++ seeding',
    summary: 'Seed k centres, each drawn with probability proportional to the squared distance to the nearest so far.',
    role: 'construction',
    random: true,
    notes: ['k-means'],
    cite: ['arthur2007'],
  },
  codebook.kmeansPlusPlus,
)
fn(
  {
    key: 'assignNearest',
    name: 'Nearest-codeword assignment',
    role: 'transform',
    notes: ['k-means', 'product-quantisation'],
  },
  codebook.assignNearest,
)
fn({ key: 'lloydUpdate', name: "Lloyd's centroid update", role: 'transform', notes: ['k-means'] }, codebook.lloydUpdate)
fn(
  {
    key: 'trainCodebook',
    name: 'Vector-quantisation codebook',
    summary: "k codewords by k-means++ seeding and Lloyd's iterations.",
    role: 'fit',
    random: true,
    notes: ['k-means', 'inverted-file-index', 'product-quantisation'],
  },
  codebook.trainCodebook,
)
fn(
  {
    key: 'ivfIndex',
    name: 'Inverted file index',
    summary: 'k-means cells, each listing its points; a query scans the cells of its nearest centroids.',
    role: 'construction',
    random: true,
    notes: ['inverted-file-index'],
    cite: ['sivic2003', 'jegou2011'],
  },
  quantisation.ivfIndex,
)
fn({ key: 'ivfQuery', name: 'IVF query', role: 'solver', notes: ['inverted-file-index'] }, quantisation.ivfQuery)
fn(
  { key: 'ivfSearch', name: 'IVF search over many queries', role: 'solver', notes: ['inverted-file-index'] },
  quantisation.ivfSearch,
)
fn(
  {
    key: 'productQuantiser',
    name: 'Product quantiser',
    summary: 'One k-means codebook per sub-space: a vector becomes M codeword indices.',
    role: 'fit',
    random: true,
    notes: ['product-quantisation'],
    cite: ['jegou2011'],
  },
  quantisation.productQuantiser,
)
fn(
  {
    key: 'optimisedProductQuantiser',
    name: 'Optimised product quantiser (OPQ)',
    summary: 'Alternate PQ on XR with the orthogonal Procrustes update of the rotation R.',
    role: 'fit',
    random: true,
    notes: ['optimised-product-quantisation', 'product-quantisation'],
    cite: ['ge2014'],
  },
  quantisation.optimisedProductQuantiser,
)
fn({ key: 'pqEncode', name: 'PQ encoding', role: 'transform', notes: ['product-quantisation'] }, quantisation.pqEncode)
fn({ key: 'pqDecode', name: 'PQ decoding', role: 'transform', notes: ['product-quantisation'] }, quantisation.pqDecode)
fn(
  {
    key: 'pqDistanceTable',
    name: 'ADC distance table',
    summary: "A query's squared distances to every sub-codeword, M × K.",
    role: 'transform',
    notes: ['product-quantisation'],
  },
  quantisation.pqDistanceTable,
)
fn(
  {
    key: 'pqQuery',
    name: 'PQ search by asymmetric distance',
    tex: '\\hat d^2 = \\sum_m T[m, c_m]',
    role: 'solver',
    notes: ['product-quantisation'],
    cite: ['jegou2011'],
  },
  quantisation.pqQuery,
)
fn(
  { key: 'pqSearch', name: 'PQ search over many queries', role: 'solver', notes: ['product-quantisation'] },
  quantisation.pqSearch,
)
fn(
  {
    key: 'hnswIndex',
    name: 'HNSW index',
    summary: 'Layered proximity graphs: random top layers, M links per point chosen by the diversity heuristic.',
    role: 'construction',
    random: true,
    notes: ['hnsw', 'graph-based-approximate-nearest-neighbours'],
    cite: ['malkov2018'],
  },
  hnsw.hnswIndex,
)
fn(
  {
    key: 'hnswQuery',
    name: 'HNSW layered greedy search',
    summary: 'Greedy descent through the upper layers, then a beam search of width ef on layer 0.',
    role: 'solver',
    notes: ['hnsw'],
    cite: ['malkov2018'],
  },
  hnsw.hnswQuery,
)
fn({ key: 'hnswSearch', name: 'HNSW search over many queries', role: 'solver', notes: ['hnsw'] }, hnsw.hnswSearch)
fn(
  {
    key: 'searchRecall',
    name: 'Recall@k of an approximate search',
    tex: '\\frac{|A \\cap E|}{k}',
    role: 'property',
    notes: ['approximate-nearest-neighbour-benchmarking', ...ANN],
  },
  benchmark.searchRecall,
)
fn(
  {
    key: 'benchmarkSearch',
    name: 'Search benchmark',
    summary: 'Recall@k, queries per second and distance evaluations per query of a search.',
    role: 'property',
    notes: ['approximate-nearest-neighbour-benchmarking'],
    cite: ['aumuller2020'],
  },
  benchmark.benchmarkSearch,
)

/** The functions of the module, keyed by name. */
export const neighbourFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', search, trees, lsh, nnDescent, codebook, quantisation, hnsw, benchmark) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
