/** The registry of `aifn-methods/unsupervised/anomaly`: the anomaly detectors, their scores and thresholds. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as detectors from './detectors'
import * as distance from './distance'
import * as ensemble from './ensemble'
import * as isolation from './isolation'
import * as oneclass from './oneclass'
import * as statistical from './statistical'

const fn = definer<FunctionInfo>('function', 'unsupervised/anomaly')
const OVERVIEW = 'anomaly-detection'

fn(
  {
    key: 'isolationForest',
    name: 'Isolation forest',
    summary: 'Random axis-aligned splits on subsamples; anomalies are isolated in fewer splits.',
    role: 'fit',
    random: true,
    notes: ['isolation-forest', OVERVIEW],
    cite: ['liu2008'],
  },
  isolation.isolationForest,
)
fn(
  {
    key: 'isolationScore',
    name: 'Isolation score',
    tex: 's(x) = 2^{-\\mathbb{E}[h(x)]/c(\\psi)}',
    role: 'transform',
    notes: ['isolation-forest'],
    cite: ['liu2008'],
  },
  isolation.isolationScore,
)
fn(
  {
    key: 'averagePathLength',
    name: 'Average path length c(n)',
    tex: 'c(n) = 2H(n-1) - 2(n-1)/n',
    role: 'property',
    notes: ['isolation-forest'],
    cite: ['liu2008'],
  },
  isolation.averagePathLength,
)
fn(
  {
    key: 'knnScore',
    name: 'k-nearest-neighbour anomaly score',
    summary: 'The distance to the k-th nearest neighbour, or the mean distance to the k nearest.',
    role: 'transform',
    notes: ['k-nearest-neighbour-anomaly-score'],
    cite: ['ramaswamy2000'],
  },
  distance.knnScore,
)
fn(
  {
    key: 'localOutlierFactor',
    name: 'Local outlier factor',
    summary: 'The mean local reachability density of a point’s neighbours divided by its own.',
    role: 'fit',
    notes: ['local-outlier-factor'],
    cite: ['breunig2000'],
  },
  distance.localOutlierFactor,
)
fn(
  {
    key: 'localOutlierScore',
    name: 'LOF of new points',
    role: 'transform',
    notes: ['local-outlier-factor'],
    cite: ['breunig2000'],
  },
  distance.localOutlierScore,
)
fn(
  {
    key: 'oneClassSvm',
    name: 'One-class SVM',
    summary: 'Separate the data from the origin in a Gaussian-kernel feature space with maximum margin (dual QP).',
    role: 'fit',
    notes: ['one-class-support-vector-machine'],
    cite: ['scholkopf2001'],
  },
  oneclass.oneClassSvm,
)
fn(
  {
    key: 'supportVectorDataDescription',
    name: 'Support vector data description',
    summary: 'The smallest feature-space ball holding all but a share ν of the data (dual QP).',
    role: 'fit',
    notes: ['support-vector-data-description'],
    cite: ['tax2004'],
  },
  oneclass.supportVectorDataDescription,
)
fn(
  {
    key: 'oneClassScore',
    name: 'One-class score',
    summary: 'Positive outside the one-class SVM’s or SVDD’s boundary, negative inside.',
    role: 'transform',
    notes: ['one-class-support-vector-machine', 'support-vector-data-description'],
  },
  oneclass.oneClassScore,
)
fn(
  {
    key: 'mahalanobisModel',
    name: 'Gaussian model of the data (classical or MCD)',
    role: 'fit',
    notes: ['mahalanobis-distance-outliers'],
    cite: ['rousseeuw1999'],
  },
  statistical.mahalanobisModel,
)
fn(
  {
    key: 'mahalanobisScore',
    name: 'Mahalanobis anomaly score',
    role: 'transform',
    notes: ['mahalanobis-distance-outliers'],
  },
  statistical.mahalanobisScore,
)
fn(
  { key: 'pcaModel', name: 'Principal subspace', role: 'fit', notes: ['principal-component-reconstruction-error'] },
  statistical.pcaModel,
)
fn(
  {
    key: 'pcaReconstructionScore',
    name: 'PCA reconstruction error',
    tex: '\\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2',
    role: 'transform',
    notes: ['principal-component-reconstruction-error'],
  },
  statistical.pcaReconstructionScore,
)
fn(
  {
    key: 'anomalyThreshold',
    name: 'Anomaly threshold',
    summary: 'A score threshold from an empirical quantile or a peaks-over-threshold tail fit at a chosen risk.',
    role: 'estimator',
    notes: ['extreme-value-theory-for-anomalies', 'evaluating-anomaly-detectors'],
    cite: ['pickands1975'],
  },
  statistical.anomalyThreshold,
)
fn(
  {
    key: 'combineScores',
    name: 'Detector ensemble',
    summary: 'Normalise each detector’s scores (ranks or z-scores) and combine them by mean or maximum.',
    role: 'transform',
    notes: ['anomaly-detector-ensembles'],
    cite: ['aggarwal2017'],
  },
  ensemble.combineScores,
)
fn(
  { key: 'rankNormalise', name: 'Rank normalisation', role: 'transform', notes: ['anomaly-detector-ensembles'] },
  ensemble.rankNormalise,
)
fn(
  {
    key: 'anomalyScores',
    name: 'Anomaly scores of any detector',
    summary: 'Fit a named detector and score the training points and query points (higher is more anomalous).',
    role: 'fit',
    notes: [OVERVIEW, 'evaluating-anomaly-detectors'],
  },
  detectors.anomalyScores,
)

fn(
  {
    key: 'compareDetectors',
    name: 'Detector comparison',
    summary: 'AUROC and average precision of every detector on points with known anomalies.',
    role: 'estimator',
    notes: ['evaluating-anomaly-detectors'],
  },
  detectors.compareDetectors,
)

/** The functions of the module, keyed by name. */
export const anomalyFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', isolation, distance, oneclass, statistical, ensemble, detectors) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
