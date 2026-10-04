/**
 * `aifn-methods/unsupervised/embedding`: embeddings. The shared layer holds squared distances and neighbour graphs,
 * the classical-MDS core, and dense views of tensors; children: linear, manifold, neighbour.
 */

export { pca } from './linear'
export { isomap } from './manifold'
export { tsne } from './neighbour'
