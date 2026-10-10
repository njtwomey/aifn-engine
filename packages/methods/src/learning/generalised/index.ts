/**
 * `aifn-methods/learning/generalised`: generalised models on the likelihoods of
 * `aifn-compute/probability/likelihoods`, from GLMs to additive and ordinal models.
 *
 * - `glm`: generalised linear models with classical inference, negative-binomial regression and logistic regression
 *   (as R's `glm` and statsmodels' `GLM`).
 * - `gam`: generalised additive models with penalised smooths, smoothing parameters by REML or GCV/UBRE, and
 *   several fitters; expectile GAMs, GAMLSS and the explainable boosting machine.
 * - `ordinal`: regression on ordered classes (cumulative-link, threshold-loss, binary-decomposition and deep models).
 *
 * The shared layer the three build on:
 *
 * - Fitting: `irls`, penalised iteratively reweighted least squares for any family and link, and `backfitting` with
 *   local scoring for additive models; both traceable.
 * - Diagnostics: `deviance` and `residuals` (response, Pearson, deviance, working).
 * - Smoothing-parameter selection: `penaltyMatrix` ($\Smat_\lambda = \sum_k \lambda_k \Smat_k$),
 *   `nullSpaceDimension`, `penalisedFit` (P-IRLS with the effective degrees of freedom and $\log\det\Hmat$) and
 *   `smoothingCriterion` (GCV/UBRE or REML).
 * - The registry: `generalisedAlgorithms` and `generalisedFunctions`.
 *
 * Binomial responses are proportions, with the trial counts as prior weights (as R's `glm`). Non-convergence is
 * reported in the algorithms' states and the fitted models, not thrown.
 */

export { deviance, irls, type IrlsProblem, type IrlsState } from './irls'
export { residuals, type ResidualInput, type ResidualKind } from './residuals'
export { backfitting, type BackfitProblem, type BackfitState } from './backfitting'
export {
  nullSpaceDimension,
  penalisedFit,
  penaltyMatrix,
  smoothingCriterion,
  type PenalisedData,
  type PenalisedDesign,
  type PenalisedFit,
} from './smoothing'
export { glm } from './glm'
export { gam } from './gam'
export { ordinalRegression } from './ordinal'
export { generalisedAlgorithms, generalisedFunctions } from './registry'
