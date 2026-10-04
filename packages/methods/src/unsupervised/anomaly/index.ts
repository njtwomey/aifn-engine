/**
 * `aifn-methods/unsupervised/anomaly`: anomaly detectors. Isolation forest; k-nearest-neighbour distances and the local
 * outlier factor; the one-class SVM and support vector data description; Mahalanobis distances under the classical or
 * the robust (MCD) estimate; PCA reconstruction error; score ensembles; thresholds by quantile or peaks over threshold;
 * and `anomalyScores`, every detector behind one call.
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
