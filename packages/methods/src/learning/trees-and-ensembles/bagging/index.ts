/**
 * `aifn-methods/learning/trees-and-ensembles/bagging`: bagging and random forests of CART classification trees.
 *
 * - The estimator: `randomForest`, trees on bootstrap samples searching random feature subsets (`bootstrap: false` and
 *   every feature searched give a plain ensemble; every feature with bootstrap gives bagging), with out-of-bag
 *   accuracy, impurity importances and the predictive of the first $t$ trees.
 * - The growth: `forestGrowth`, one tree per step as a traceable algorithm, for watching a forest grow.
 *
 * Trees come from `aifn-methods/learning/trees-and-ensembles`; tree $t$ depends only on the root stream and $t$, so a
 * forest of more trees extends one of fewer.
 */
export { forestGrowth, randomForest, type ForestProblem, type ForestState, type RandomForestModel } from './forest'
export { baggingAlgorithms } from './registry'
