/**
 * `aifn-methods/learning/trees-and-ensembles/bagging`: bagging and random forests of CART classification and
 * regression trees.
 *
 * - The estimator: `randomForest`, trees on bootstrap samples searching random feature subsets (`bootstrap: false` and
 *   every feature searched give a plain ensemble; every feature with bootstrap gives bagging), with out-of-bag
 *   accuracy, impurity importances and the predictive of the first $t$ trees.
 * - The regressor: `randomForestRegressor`, regression trees on bootstrap samples, with every tree's prediction
 *   (`treePredictions`) and their mean and variance (`moments`, optionally adding the leaves' variances), the surrogate
 *   of SMAC.
 * - The growth: `forestGrowth`, one tree per step as a traceable algorithm, for watching a forest grow.
 *
 * Trees come from `aifn-methods/learning/trees-and-ensembles`; tree $t$ depends only on the root stream and $t$, so a
 * forest of more trees extends one of fewer.
 */
export { forestGrowth, randomForest, type ForestProblem, type ForestState, type RandomForestModel } from './forest'
export {
  randomForestRegressor,
  type ForestMomentsOptions,
  type RandomForestRegressionModel,
  type RandomForestRegressorParams,
} from './regressor'
export { baggingAlgorithms } from './registry'
