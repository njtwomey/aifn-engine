/**
 * `aifn-methods/learning/linear`: linear models for regression and binary classification, as scikit-learn's
 * `LinearRegression`, `Ridge` and `Perceptron`.
 *
 * - Regression: `linearRegression`, least squares by SVD (minimum-norm when the design is rank deficient) or ridge with
 *   an unpenalised intercept, with a plug-in Gaussian predictive.
 * - Classification: `perceptron`, Rosenblatt's mistake-driven classifier for labels 0 and 1, optionally averaged;
 *   `perceptronSteps` is the same run as a step-through algorithm on labels $\pm 1$, one example per step.
 *
 * `linearAlgorithms` lists the step-through algorithm by key. For logistic regression and the other generalised
 * linear models see `aifn-methods/learning/generalised`.
 */

export {
  perceptron,
  perceptronSteps,
  type PerceptronModel,
  type PerceptronProblem,
  type PerceptronState,
} from './perceptron'
export { linearRegression, type LinearRegressionModel, type LinearRegressionParams } from './leastSquares'
export { linearAlgorithms } from './registry'
