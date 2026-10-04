/**
 * `aifn-methods/unsupervised/embedding/linear`: linear embeddings: PCA, kernel PCA, classical and metric MDS, Andrews
 * curves, and the linear latent-variable models (factor analysis, probabilistic PCA, FastICA).
 */

export {
  classicalMds,
  kernelPca,
  metricMds,
  pca,
  smacofSteps,
  stress,
  type ClassicalMds,
  type KernelPcaModel,
  type MetricMdsModel,
  type PcaModel,
  type SmacofState,
} from './linear'
export { andrewsCurves } from './andrews'
export {
  factorAnalysis,
  fastIca,
  fastIcaSteps,
  latentGaussianSteps,
  probabilisticPca,
  type FactorAnalysisModel,
  type FastIcaModel,
  type FastIcaState,
  type LatentGaussianModel,
  type LatentGaussianState,
  type LatentNoise,
  type ProbabilisticPcaModel,
} from './latent'
export { linearEmbeddingAlgorithms, linearEmbeddingFunctions } from './registry'
