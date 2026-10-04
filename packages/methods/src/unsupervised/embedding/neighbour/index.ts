/**
 * `aifn-methods/unsupervised/embedding/neighbour`: neighbour embeddings: t-SNE, UMAP and PaCMAP (approximate k-nearest neighbours
 * by nearest-neighbour descent come from `aifn-compute/numerics/neighbours`).
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
