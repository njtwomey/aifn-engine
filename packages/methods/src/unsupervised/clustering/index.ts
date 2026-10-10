/**
 * `aifn-methods/unsupervised/clustering`: clustering estimators: k-means with k-means++ seeding, mini-batch and
 * k-medoids; Gaussian mixtures by EM; hierarchical clustering; DBSCAN, OPTICS and mean shift; spectral clustering.
 *
 * - Centres: `kmeans` (Lloyd's algorithm, best of several k-means++ restarts), `miniBatchKMeans` (for large $n$) and
 *   `kMedoids` (PAM, centres that are rows of the data, on all $n^2$ distances). Each predicts the nearest centre of
 *   new rows.
 * - Soft assignments: `gaussianMixture`, EM with full, diagonal or spherical covariances, whose `predictive` is the
 *   law of the component.
 * - A hierarchy: `agglomerative` and `linkage` (single, complete, average or Ward linkage, in SciPy's linkage format),
 *   with `cutTree` for flat clusters, `mergeTree` for the tree and `dendrogram` for its layout.
 * - Density: `dbscan` (core, border and noise points, with the roles `CORE`, `BORDER`, `NOISE`), `optics` (the
 *   reachability ordering, read off at any radius) and `meanShift` (the modes of a kernel density; the number of
 *   clusters is not given).
 * - A graph: `spectralClustering`, k-means on the top eigenvectors of a normalised `affinityMatrix`, for clusters that
 *   are connected rather than compact.
 * - Step by step: `kmeansSteps`, `miniBatchKMeansSteps`, `kMedoidsSteps`, `gaussianMixtureSteps`, `meanShiftSteps`
 *   and `agglomerativeSteps` are traceable algorithms that the estimators trace into `training`; the registries
 *   `clusteringAlgorithms` and `clusteringFunctions` list them.
 * - `canonicalLabels` renumbers labels in order of first appearance, to compare clusterings.
 *
 * Every estimator works on Euclidean distances between the rows of an $n \times d$ matrix, and labels clusters
 * $0, 1, \dots$ with $-1$ for noise. Randomness (seedings, restarts, batches) comes from the fit's stream. The
 * hierarchical, OPTICS and spectral models are transductive: they label the training rows only.
 */

export {
  kMedoids,
  kMedoidsSteps,
  kmeans,
  kmeansSteps,
  miniBatchKMeans,
  miniBatchKMeansSteps,
  type KMeansInit,
  type KMeansModel,
  type KMeansState,
  type KMedoidsModel,
  type KMedoidsState,
  type MiniBatchKMeansState,
} from './centroid'
export {
  gaussianMixture,
  gaussianMixtureSteps,
  type CovarianceType,
  type GaussianMixtureModel,
  type MixtureInit,
  type MixtureParameters,
  type MixtureState,
} from './mixture'
export {
  agglomerative,
  agglomerativeSteps,
  cutTree,
  dendrogram,
  linkage,
  mergeTree,
  type AgglomerationState,
  type AgglomerativeModel,
  type Dendrogram,
  type Linkage,
  type MergeData,
  type MergeTree,
} from './hierarchical'
export {
  BORDER,
  CORE,
  NOISE,
  dbscan,
  meanShift,
  meanShiftSteps,
  optics,
  type DbscanModel,
  type MeanShiftModel,
  type MeanShiftState,
  type OpticsModel,
} from './density'
export { affinityMatrix, spectralClustering, type Affinity, type SpectralClusteringModel } from './spectral'
export { canonical as canonicalLabels } from './util'
export { clusteringAlgorithms, clusteringFunctions } from './registry'
