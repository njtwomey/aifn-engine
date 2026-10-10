/**
 * `aifn-methods/unsupervised`: unsupervised learning, finding structure in rows with no targets: clusters,
 * low-dimensional embeddings and anomalies.
 *
 * - `clustering`: groups of rows, by centres (k-means family), mixtures fitted by EM, hierarchies, density (DBSCAN,
 *   OPTICS, mean shift) and spectral clustering; `kmeans`, `gaussianMixture` and `dbscan` are exported here.
 * - `embedding`: maps to a few coordinates, linear (PCA, MDS, latent-variable models), manifold (Isomap, Laplacian
 *   eigenmaps, LLE, diffusion maps, self-organising maps) and neighbour embeddings (t-SNE, UMAP, PaCMAP); `pca`,
 *   `tsne` and `umap` are exported here.
 * - `anomaly`: scores of how unusual each row is, by isolation, neighbour distances, kernel boundaries and models of
 *   the data, with thresholds and ensembles.
 *
 * `unsupervisedModelRegistry` lists every registered estimator factory of the clustering and embedding modules by key.
 * As scikit-learn's `sklearn.cluster`, `sklearn.decomposition`, `sklearn.manifold` and the outlier detectors.
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
