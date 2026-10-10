/**
 * `aifn-methods/unsupervised/anomaly`: anomaly detectors. Isolation forest; k-nearest-neighbour distances and the local
 * outlier factor; the one-class SVM and support vector data description; Mahalanobis distances under the classical or
 * the robust (MCD) estimate; PCA reconstruction error; score ensembles; thresholds by quantile or peaks over threshold;
 * and `anomalyScores`, every detector behind one call.
 *
 * - Random partitions: `isolationForest` grows the trees and `isolationScore` scores points in $(0, 1]$;
 *   `averagePathLength` is its normaliser $c(n)$.
 * - Neighbours: `knnScore` (the distance to the $k$-th neighbour, or the mean to the $k$ nearest), and the local
 *   outlier factor `localOutlierFactor` (training points) with `localOutlierScore` (new points), which allows for
 *   clusters of different densities.
 * - Kernel boundaries: `oneClassSvm` and `supportVectorDataDescription`, dense quadratic programmes for a few hundred
 *   points, scored by `oneClassScore`.
 * - Models of the data: `mahalanobisModel` (classical or robust MCD) with `mahalanobisScore`, and `pcaModel` with
 *   `pcaReconstructionScore` for points off a linear subspace.
 * - Combining and deciding: `rankNormalise` and `combineScores` put detectors on one scale; `anomalyThreshold` turns
 *   scores into a cut-off by quantile or by a peaks-over-threshold tail.
 * - One call for all: `anomalyScores` fits any detector of `DETECTORS` by name, `compareDetectors` ranks them by AUROC
 *   and average precision against known labels; `anomalyFunctions` is the registry.
 *
 * Every score is higher for more anomalous points. The fitting functions return plain models (not estimators), and the
 * scoring functions take nested arrays or rank-2 tensors with one point per row and return a `Float64Array`. Two fits
 * are random: the isolation forest draws from the stream it is given, and the robust MCD estimate from its own default
 * stream, so both repeat exactly.
 */

export {
  anomalyScores,
  compareDetectors,
  DETECTORS,
  type AnomalyScores,
  type DetectorComparison,
  type DetectorKind,
  type DetectorOptions,
} from './detectors'
export {
  knnScore,
  localOutlierFactor,
  localOutlierScore,
  type KnnScoreOptions,
  type LocalOutlierFactor,
} from './distance'
export { combineScores, rankNormalise, type EnsembleOptions } from './ensemble'
export {
  averagePathLength,
  isolationForest,
  isolationScore,
  type IsolationForest,
  type IsolationForestOptions,
  type IsolationNode,
} from './isolation'
export {
  oneClassScore,
  oneClassSvm,
  supportVectorDataDescription,
  type OneClassModel,
  type OneClassOptions,
} from './oneclass'
export {
  anomalyThreshold,
  mahalanobisModel,
  mahalanobisScore,
  pcaModel,
  pcaReconstructionScore,
  type MahalanobisModel,
  type PcaModel,
  type ThresholdOptions,
} from './statistical'
export { anomalyFunctions } from './registry'
