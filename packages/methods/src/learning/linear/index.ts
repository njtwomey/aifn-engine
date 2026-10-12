/**
 * `aifn-methods/learning/linear`: linear models for regression and binary classification, as scikit-learn's
 * `LinearRegression`, `Ridge`, `ElasticNet`, `Lasso`, `lars_path`, `SGDRegressor` and `Perceptron`.
 *
 * - Least squares and ridge: `linearRegression`, by SVD (minimum-norm when the design is rank deficient) or ridge with
 *   an unpenalised intercept, with a plug-in Gaussian predictive and the effective degrees of freedom; `ridgePath`
 *   fits many penalties through one SVD, and `effectiveDof` gives $\sum_i d_i^2 / (d_i^2 + \lambda)$.
 * - Sparse fits: `elasticNet`, the elastic net and (with `l1Ratio` 1) the lasso by coordinate descent, with exact
 *   zeros; `elasticNetPath` fits a decreasing grid of penalties with warm starts; `leastAngleRegression` gives the knots of
 *   the LARS path and, with its lasso modification, of the exact lasso path.
 * - Stochastic gradient: `linearSgdSteps`, penalised least squares one shuffled epoch per step, with the truncated
 *   $\ell_1$ penalty of `SGDRegressor` or the plain subgradient.
 * - Classification: `perceptron`, Rosenblatt's mistake-driven classifier for labels 0 and 1, optionally averaged;
 *   `perceptronSteps` is the same run as a step-through algorithm on labels $\pm 1$, one example per step.
 *
 * Every objective is in scikit-learn's scaling, $\frac{1}{2n}$ on the squared error, so results compare with it
 * directly. `linearAlgorithms` lists the step-through algorithms and `linearFunctions` the paths, by key. For logistic
 * regression and the other generalised linear models see `aifn-methods/learning/generalised`.
 */

export {
  perceptron,
  perceptronSteps,
  type PerceptronModel,
  type PerceptronProblem,
  type PerceptronState,
} from './perceptron'
export {
  effectiveDof,
  linearRegression,
  ridgePath,
  type LinearRegressionModel,
  type LinearRegressionParams,
  type RidgePath,
} from './leastSquares'
export {
  elasticNet,
  elasticNetPath,
  type ElasticNetModel,
  type ElasticNetParams,
  type ElasticNetPath,
  type ElasticNetPathOptions,
} from './elasticNet'
export { leastAngleRegression, type LeastAngleEvent, type LeastAngleOptions, type LeastAnglePath } from './lars'
export { linearSgdSteps, type LinearSgdProblem, type LinearSgdState } from './sgd'
export { linearAlgorithms, linearFunctions } from './registry'
