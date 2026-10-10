/**
 * `aifn-methods/learning/generalised/glm`: generalised linear models with classical inference, as R's `glm` and
 * statsmodels' `GLM`.
 *
 * - Any family and link: `glm`, fitted by IRLS, with standard errors, Wald tests, deviance, dispersion, AIC and
 *   residuals.
 * - Overdispersed counts: `negativeBinomialRegression` (NB2, R's `glm.nb`), which alternates a `glm` fit with
 *   `thetaMaximumLikelihood` for the shape $\theta$; the alternation itself is the traceable
 *   `negativeBinomialAlternation`.
 * - Classification: `logisticRegression` (binary by IRLS, $K \ge 3$ classes by `softmaxNewton`; scikit-learn's
 *   objective, $C = 1/\lambda$) and `multinomialLogisticRegression`, the softmax model with the covariance of its
 *   weights and Wald tests of each class against a reference class.
 * - The registry: `glmAlgorithms` and `glmFunctions`.
 *
 * The estimators return fitted models whose Newton or IRLS run is kept in `training`; the intercept is the last
 * coefficient. A fit that does not converge within its step budget is reported in `converged`, not thrown.
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
