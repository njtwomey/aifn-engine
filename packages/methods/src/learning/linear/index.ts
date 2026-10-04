/**
 * `aifn-methods/learning/linear`: linear models: least squares and ridge regression, and the perceptron.
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
