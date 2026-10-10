/**
 * `aifn-methods/unsupervised/embedding`: embeddings, maps of high-dimensional rows to a few coordinates.
 *
 * - `linear`: projections and linear models, `pca` (exported here), kernel PCA, classical and metric MDS, factor
 *   analysis, probabilistic PCA, FastICA and Andrews curves.
 * - `manifold`: neighbour-graph and diffusion embeddings, `isomap` (exported here), Laplacian eigenmaps, locally linear
 *   embedding and diffusion maps, and self-organising maps.
 * - `neighbour`: neighbour embeddings for visualisation, `tsne` (exported here), UMAP and PaCMAP.
 *
 * The shared layer at this level holds the internal helpers of the children: squared distances and $k$-nearest
 * neighbour lists, the classical-MDS core $-\tfrac{1}{2}\Jmat\Dmat^{(2)}\Jmat$ that classical MDS and Isomap embed
 * through, and dense float64 views of tensors. Every estimator fits on `{ x }`, a matrix with one point per row; PCA,
 * kernel PCA, the latent models, FastICA and SOMs map new rows, and the rest are transductive (they place the training
 * rows only).
 */

export { pca } from './linear'
export { isomap } from './manifold'
export { tsne } from './neighbour'
