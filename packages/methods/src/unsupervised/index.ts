/**
 * `aifn-methods/unsupervised`: unsupervised learning: clustering (k-means family, mixtures by EM, hierarchical,
 * density-based, spectral) and embeddings (linear, manifold, neighbour). `unsupervisedModelRegistry` lists every
 * registered estimator factory of the area by key.
 */

import { entries } from 'aifn-compute/foundation/registry'
import type { ModelEntry } from 'aifn-compute/learning/estimators'
import * as clustering from './clustering'
import * as embeddingLinear from './embedding/linear'
import * as manifold from './embedding/manifold'
import * as neighbour from './embedding/neighbour'

export { kmeans, gaussianMixture, dbscan } from './clustering'
export { pca } from './embedding/linear'
export { tsne, umap } from './embedding/neighbour'

/** Every registered estimator factory of `aifn-methods/unsupervised`, keyed by `info.key` (kind `model`). */
export const unsupervisedModelRegistry = entries('model', clustering, embeddingLinear, manifold, neighbour) as Readonly<
  Record<string, ModelEntry>
>
