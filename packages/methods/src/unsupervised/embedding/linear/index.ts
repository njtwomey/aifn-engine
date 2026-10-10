/**
 * `aifn-methods/unsupervised/embedding/linear`: linear embeddings, multidimensional scaling and linear latent-variable
 * models.
 *
 * - Projections that map new rows: `pca` (by the SVD of the centred data, with explained variance, whitening and
 *   reconstruction, as scikit-learn's `PCA`) and `kernelPca` (on the double-centred kernel matrix).
 * - Multidimensional scaling of distances: `classicalMds` (closed form, from the top eigenpairs of
 *   $-\tfrac{1}{2}\Jmat\Dmat^{(2)}\Jmat$) and `metricMds` (SMACOF on the rows' Euclidean distances), with
 *   `smacofSteps` to step through the Guttman transforms and `stress` to score any configuration.
 * - Latent-variable models, $\xvec = \Wmat\zvec + \muvec + \epsilonvec$: `factorAnalysis` (diagonal noise) and
 *   `probabilisticPca` (isotropic noise; closed form or EM), both by the EM of `latentGaussianSteps`; and independent
 *   components by `fastIca`, stepped through by `fastIcaSteps`.
 * - Plotting: `andrewsCurves`, each row as a Fourier series.
 * - The registry tables `linearEmbeddingAlgorithms` and `linearEmbeddingFunctions`.
 *
 * Data are matrices with one point per row. The estimators fit on `{ x }`; iterative ones keep their run in `training`
 * and draw any random start from the fit options' `stream`. Signs of axes, loadings and sources are arbitrary up to
 * the conventions stated on each model.
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
