/**
 * `aifn-methods/unsupervised/embedding/manifold`: manifold learning, by neighbour graphs, diffusion and
 * self-organising maps.
 *
 * - Neighbour-graph embeddings of the training rows: `isomap` (classical MDS of geodesic distances along the graph),
 *   `laplacianEigenmaps` (bottom eigenvectors of the normalised graph Laplacian) and `locallyLinearEmbedding` (the
 *   weights that rebuild each point from its neighbours), all from the graph `neighbourGraph` builds.
 * - `diffusionMap`: eigenvectors of a random walk on a Gaussian kernel, scaled so that Euclidean distance is diffusion
 *   distance.
 * - Self-organising maps: `selfOrganisingMap` (a grid of prototypes with its U-matrix, mapping new rows to grid
 *   cells), stepped through by `selfOrganisingMapSteps`, with `somGrid` for the units' grid positions.
 * - The registry tables `manifoldAlgorithms` and `manifoldFunctions`.
 *
 * The graph and diffusion embeddings are transductive (no `transform`) and deterministic, with each coordinate's sign
 * fixed by its largest-magnitude entry; a disconnected neighbour graph is reported in `components`, not thrown.
 * Neighbours exclude the point itself, with ties to the lower index.
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
