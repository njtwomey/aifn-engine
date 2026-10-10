/**
 * `aifn-methods/learning/gaussian-processes`: Gaussian processes on the kernels of `aifn-compute/learning/kernels`, for
 * regression, sparse regression, the relevance vector machine, classification, ordinal regression and the latent
 * variable model.
 *
 * - Exact regression: `gpPrior` and `samplePrior` for the prior, `gpPosterior` for the posterior by a Cholesky factor
 *   of $\Kmat + \sigma^2\Imat$, `logMarginalLikelihood` and `logMarginalLikelihoodGradient` for the evidence,
 *   `fitGp` for type-II maximum likelihood by L-BFGS (`kernelLogVector` gives its log-space layout), and the
 *   estimator `gaussianProcessRegressor`.
 * - Sparse regression through $m$ inducing inputs, at $O(nm^2)$: `sparseGp` and `sparseLogMarginal` for the VFE,
 *   FITC, DTC and SoR approximations; `fitSparseGp` (or step by step, `sparseGpFitSteps` with `sparseGpAt`) to fit
 *   the inducing inputs and hyperparameters by L-BFGS; `sparseGpGrowSteps` to choose inducing inputs greedily; and
 *   the estimator `sparseGaussianProcessRegressor`.
 * - The relevance vector machine: `rvmProblem`, `rvmPosterior` and `rvmModel`, fitted by Tipping and Faul's fast
 *   algorithm (`rvmFastSteps`, one basis function per step) or by re-estimation (`rvmReestimationSteps`); the
 *   estimator `relevanceVectorMachine`.
 * - Binary classification (labels 0 and 1): the Laplace approximation (`laplaceMode`, `laplaceLogMarginal`, and
 *   `laplaceEvidence` differentiable in $\Kmat$) or EP with the probit link (`gpEp`, `gpEpLogMarginal`,
 *   `gpEpEvidence`); `gpClassifierEvidenceGradient` and `fitGpClassifier` for the hyperparameters; the estimator
 *   `gpClassifier`.
 * - Ordinal regression (Chu and Ghahramani, 2005): the estimator `gpOrdinalRegression`, by Laplace with the
 *   cumulative probit likelihood of `ordinalLaplaceTerms`.
 * - The GP latent variable model: `gplvmProblem` and `gplvmFitSteps` (MAP by L-BFGS from PCA), `fitGplvm`, and
 *   `gplvmModel`, whose `project` maps latent points to data space.
 * - The registry tables `gaussianProcessAlgorithms` and `gaussianProcessFunctions`.
 *
 * Every function takes the kernel as an argument; hyperparameters are fitted in log space, and the fits and
 * iterations are traceable algorithms (`run`, `trace`) whose states carry what a figure plays. Inputs are $n \times d$
 * matrices of rows (or vectors of $n$ one-dimensional inputs). Jitter added to make a matrix factor is reported, not
 * hidden. The methods follow Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning", and the tests
 * compare them with scikit-learn's `sklearn.gaussian_process`.
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
