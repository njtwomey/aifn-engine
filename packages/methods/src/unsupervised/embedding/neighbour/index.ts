/**
 * `aifn-methods/unsupervised/embedding/neighbour`: neighbour embeddings, t-SNE, UMAP and PaCMAP, which keep each point
 * near its neighbours in a low-dimensional layout.
 *
 * - t-SNE (exact, $O(n^2)$ per iteration, as scikit-learn's `TSNE(method='exact')`): `tsne`, stepped through by
 *   `tsneSteps`, on the affinities of `jointProbabilities` and the bandwidths of `perplexityCalibration`.
 * - UMAP: `umap`, stepped through by `umapSteps`, on the fuzzy graph of `fuzzyGraph` (exact neighbours up to
 *   `DESCENT_ABOVE` rows; approximate ones by nearest-neighbour descent from `aifn-compute/numerics/neighbours`
 *   beyond), with `curveParameters` for the output curve and `spectralLayout` for the start (dense up to
 *   `DENSE_SPECTRAL_UP_TO` rows, Lanczos beyond).
 * - PaCMAP: `pacmap`, stepped through by `pacmapSteps`, on the pairs of `pacmapPairs` with the phased weights of
 *   `pacmapWeights`.
 * - The registry tables `neighbourEmbeddingAlgorithms` and `neighbourEmbeddingFunctions`.
 *
 * All three are transductive (they place the training rows; there is no `transform`) and keep their run in
 * `training`. Their randomness comes from the fit options' `stream`; distances between well-separated groups in the
 * result say little, only which points are together.
 */

export {
  jointProbabilities,
  perplexityCalibration,
  tsne,
  tsneSteps,
  type PerplexityCalibration,
  type TsneModel,
  type TsneParams,
  type TsneState,
} from './tsne'
export {
  curveParameters,
  DENSE_SPECTRAL_UP_TO,
  DESCENT_ABOVE,
  fuzzyGraph,
  spectralLayout,
  umap,
  umapSteps,
  type FuzzyGraph,
  type FuzzyGraphOptions,
  type NeighbourSearch,
  type UmapModel,
  type UmapState,
} from './umap'
export {
  pacmap,
  pacmapPairs,
  pacmapSteps,
  pacmapWeights,
  type PacmapModel,
  type PacmapPairs,
  type PacmapState,
} from './pacmap'
export { neighbourEmbeddingAlgorithms, neighbourEmbeddingFunctions } from './registry'
