/**
 * `aifn-methods/unsupervised/embedding/graph`: node embeddings, vectors for the nodes of a graph from its structure.
 *
 * - Matrix factorisation: `netmfMatrix`, the matrix DeepWalk implicitly factorises (NetMF), with `nodeSimilarity`
 *   (adjacency, two-hop paths or the NetMF matrix) and `dotProductEmbedding`, the exact rank-$d$ fit
 *   $\Zmat\Zmat^\top \approx \Smat$ by the leading eigenvectors, scored by `reconstructionError`.
 * - Random walks: `node2vecWalks` (DeepWalk at $p = q = 1$), one step's `node2vecTransition`, the window counts
 *   `walkCooccurrence`, and `skipGram`, skip-gram with negative sampling fitted to the walks.
 * - `graphEmbeddingFunctions`: the module's registry entries.
 *
 * Graphs are `Graph`s of `aifn-compute/graph`, undirected for the usual embeddings, with nodes $0, \dots, n - 1$.
 * Everything is deterministic given its stream; the walks and the skip-gram fit take plain data, so a worker can run
 * them. The NetMF embedding itself is `wordVectors(netmfMatrix(g), d)` and the PMI of sampled walks is
 * `ppmi(walkCooccurrence(walks, n, T), { shift: b })`, both of `aifn-compute/text/cooccurrence`.
 */

export { netmfMatrix, type NetmfOptions } from './netmf'
export {
  dotProductEmbedding,
  nodeSimilarity,
  reconstructionError,
  type DotProductEmbedding,
  type SimilarityKind,
} from './shallow-embedding'
export {
  node2vecTransition,
  node2vecWalks,
  skipGram,
  walkCooccurrence,
  type SkipGramOptions,
  type SkipGramResult,
  type WalkOptions,
  type WalkTransition,
} from './random-walk-embedding'
export { graphEmbeddingFunctions } from './registry'
