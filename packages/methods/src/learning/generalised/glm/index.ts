/**
 * `aifn-methods/learning/generalised/glm`: generalised linear models: `glm` with any family and link,
 * negative-binomial regression with θ by maximum likelihood, multinomial logistic regression, and logistic regression.
 */

export {
  glm,
  negativeBinomialAlternation,
  negativeBinomialRegression,
  thetaMaximumLikelihood,
  type GlmData,
  type GlmModel,
  type GlmParams,
  type NegativeBinomialParams,
  type NegativeBinomialState,
} from './model'
export {
  multinomialLogisticRegression,
  type Contrasts,
  type MultinomialModel,
  type MultinomialParams,
} from './multinomial'
export {
  logisticRegression,
  softmaxNewton,
  type LogisticRegressionModel,
  type LogisticRegressionParams,
  type SoftmaxNewtonState,
  type SoftmaxProblem,
} from './logistic'
export { glmAlgorithms, glmFunctions } from './registry'
