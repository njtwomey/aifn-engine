/**
 * `aifn-methods/learning/generalised`: generalised models on the likelihoods of `aifn-compute/probability/likelihoods`. The
 * shared layer holds penalised iteratively reweighted least squares (`irls`), residuals, backfitting and
 * smoothing-parameter selection (penalised fits, GCV/UBRE and REML criteria); children: glm, gam, ordinal.
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
