/**
 * `aifn-methods/learning/gaussian-processes`: Gaussian processes on the kernels of `aifn-compute/learning/kernels`: regression
 * with the marginal likelihood and fitting, sparse approximations (fitted by L-BFGS or grown greedily, step by step), the relevance vector machine (Tipping and
 * Faul's fast algorithm and re-estimation), classification (Laplace and EP, with evidence
 * gradients and L-BFGS fitting), ordinal regression (Laplace; Chu & Ghahramani, 2005), and the GP latent variable
 * model (MAP GPLVM by L-BFGS from PCA, with the latent-to-data map `project`).
 */

export {
  fitGp,
  gaussianProcessRegressor,
  gpPosterior,
  gpPrior,
  kernelLogVector,
  logMarginalLikelihood,
  logMarginalLikelihoodGradient,
  samplePrior,
  type Draws,
  type FitGpOptions,
  type GaussianProcessRegressionModel,
  type GaussianProcessRegressorParams,
  type GpFit,
  type GpOptions,
  type GpPosterior,
  type LogMarginal,
  type LogMarginalGradient,
  type Prediction,
} from './regression'
export {
  fitSparseGp,
  sparseGaussianProcessRegressor,
  sparseGp,
  sparseGpAt,
  sparseGpFitSteps,
  sparseGpGrowSteps,
  sparseLogMarginal,
  type FitSparseGpOptions,
  type SparseGpFitFields,
  type SparseGpFitState,
  type SparseGpGrowOptions,
  type SparseGpGrowState,
  type SparseGp,
  type SparseGpFit,
  type SparseGaussianProcessRegressionModel,
  type SparseGaussianProcessRegressorParams,
  type SparseGpOptions,
  type SparseMethod,
} from './sparse'
export {
  fitGplvm,
  gplvmFitSteps,
  gplvmModel,
  gplvmProblem,
  type GplvmFit,
  type GplvmModel,
  type GplvmOptions,
  type GplvmProblem,
  type GplvmState,
} from './gplvm'
export {
  fitGpClassifier,
  gpClassifier,
  gpClassifierEvidenceGradient,
  gpEpLogMarginal,
  laplaceEvidence,
  laplaceLogMarginal,
  laplaceMode,
  type ClassificationLikelihood,
  type FitGpClassifierOptions,
  type GpClassificationMethod,
  type GpClassifierEvidenceGradient,
  type GpClassifierEvidenceOptions,
  type GpClassifierFit,
  type GpClassifierModel,
  type GpClassifierParams,
  type LaplaceEvidenceOptions,
  type LaplaceProblem,
  type LaplaceState,
  type LaplaceTerms,
} from './classification'
export {
  gpOrdinalRegression,
  ordinalLaplaceTerms,
  type GpOrdinalRegressionModel,
  type GpOrdinalRegressionParams,
} from './ordinal'
export {
  relevanceVectorMachine,
  rvmFastSteps,
  rvmModel,
  rvmPosterior,
  rvmProblem,
  rvmReestimationSteps,
  type RelevanceVectorMachineModel,
  type RelevanceVectorMachineParams,
  type RvmAction,
  type RvmModel,
  type RvmOptions,
  type RvmPosterior,
  type RvmProblem,
  type RvmState,
} from './rvm'
export { gpEp, gpEpEvidence, type GpEpOptions, type GpEpProblem } from './classification-ep'
export { gaussianProcessAlgorithms, gaussianProcessFunctions } from './registry'
