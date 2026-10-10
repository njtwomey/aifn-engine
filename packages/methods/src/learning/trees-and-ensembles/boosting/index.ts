/**
 * `aifn-methods/learning/trees-and-ensembles/boosting`: boosted ensembles of CART trees, AdaBoost and gradient
 * boosting, with their stages as traceable algorithms.
 *
 * - Classification by reweighting: `adaBoost` (multiclass SAMME on stumps by default), and `adaBoostSteps`, one weak
 *   learner per step with its weight $\alpha_m$ and the next sample weights.
 * - Gradient boosting: `gradientBoosting` for squared error (regression) or the logistic deviance (binary or
 *   multinomial classification), and `gradientBoostingSteps`, one stage of regression trees per step on the
 *   pseudo-residuals, with optional subsampling (stochastic gradient boosting).
 *
 * The fitted models keep their run in `training` and read the ensemble after any number of stages (`votesUpTo`,
 * `rawUpTo`), as scikit-learn's `staged_predict`.
 */

export {
  adaBoost,
  adaBoostSteps,
  gradientBoosting,
  gradientBoostingSteps,
  type AdaBoostModel,
  type AdaBoostProblem,
  type AdaBoostState,
  type BoostingLoss,
  type GradientBoostingModel,
  type GradientBoostingProblem,
  type GradientBoostingState,
} from './ensembles'
export { boostingAlgorithms } from './registry'
