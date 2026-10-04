/**
 * `aifn-methods/unsupervised/embedding/manifold`: manifold learning: Isomap, locally linear embedding, Laplacian
 * eigenmaps, diffusion maps and self-organising maps.
 */

export {
  isomap,
  laplacianEigenmaps,
  locallyLinearEmbedding,
  neighbourGraph,
  type IsomapModel,
  type LleModel,
  type SpectralEmbeddingModel,
} from './manifold'
export { diffusionMap, type DiffusionMapModel } from './diffusion'
export { selfOrganisingMap, selfOrganisingMapSteps, somGrid, type SomModel, type SomParams, type SomState } from './som'
export { manifoldAlgorithms, manifoldFunctions } from './registry'
