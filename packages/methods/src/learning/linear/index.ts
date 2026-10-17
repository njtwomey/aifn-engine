/**
 * `aifn-methods/learning/linear`: linear models for regression and binary classification, as scikit-learn's
 * `LinearRegression`, `Ridge`, `ElasticNet`, `Lasso`, `lars_path`, `SGDRegressor`, `BayesianRidge` and `Perceptron`.
 *
 * - Least squares and ridge: `linearRegression`, by SVD (minimum-norm when the design is rank deficient) or ridge with
 *   an unpenalised intercept, with a plug-in Gaussian predictive and the effective degrees of freedom; `ridgePath`
 *   fits many penalties through one SVD, and `effectiveDof` gives $\sum_i d_i^2 / (d_i^2 + \lambda)$.
 * - Sparse fits: `elasticNet`, the elastic net and (with `l1Ratio` 1) the lasso by coordinate descent, with exact
 *   zeros; `elasticNetPath` fits a decreasing grid of penalties with warm starts; `leastAngleRegression` gives the
 *   knots of the LARS path and, with its lasso modification, of the exact lasso path. `groupLasso` and
 *   `groupLassoPath` select whole groups of columns with the penalty $\alpha\sum_g \sqrt{p_g}\norm{\wvec_g}_2$, by
 *   block coordinate descent.
 * - Subset selection: `bestSubset`, the best least-squares fit of each size by exhaustive search (small $d$), with
 *   every subset's residual sum of squares; `stepwise`, forward, backward or bidirectional greedy selection by RSS,
 *   AIC or BIC; `forwardStagewise`, the $\varepsilon$-step path that tends to the monotone lasso path.
 * - Bayesian linear regression: `bayesianLinearEvidence`, the log evidence at given precisions with its Occam
 *   decomposition and the posterior; `evidenceMaximisation` and the step-through `evidenceMaximisationSteps`, MacKay's
 *   updates of the precisions (empirical Bayes).
 * - Hyperparameter gradients: `ridgeHypergradient`, the validation loss of ridge and its derivative in the penalty by
 *   the implicit function theorem; `ridgeHyperDescentSteps`, gradient descent on $\ln\lambda$ with it.
 * - Stochastic gradient: `linearSgdSteps`, penalised least squares one shuffled epoch per step, with the truncated
 *   $\ell_1$ penalty of `SGDRegressor` or the plain subgradient.
 * - Rule ensembles: `ruleFitPath`, RuleFit's lasso path over the rules of a tree ensemble and winsorised linear terms
 *   (the terms from `aifn-compute/learning/rules`), with `ruleFitPredict` and `ruleImportances`.
 * - Classification: `perceptron`, Rosenblatt's mistake-driven classifier for labels 0 and 1, optionally averaged;
 *   `perceptronSteps` is the same run as a step-through algorithm on labels $\pm 1$, one example per step.
 *
 * The penalised objectives are in scikit-learn's scaling, $\frac{1}{2n}$ on the squared error, so results compare with
 * it directly; ridge and its hypergradient are in that of `Ridge`, with no $\frac1n$. `linearAlgorithms` lists the
 * step-through algorithms and `linearFunctions` the paths, selections, evidence and hypergradient, by key. For
 * logistic regression and the other generalised linear models see `aifn-methods/learning/generalised`.
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
export {
  groupLasso,
  groupLassoPath,
  type GroupLassoModel,
  type GroupLassoParams,
  type GroupLassoPath,
  type GroupLassoPathOptions,
} from './groupLasso'
export {
  bestSubset,
  forwardStagewise,
  stepwise,
  type BestSubsetOptions,
  type BestSubsets,
  type ForwardStagewiseOptions,
  type ForwardStagewisePath,
  type Stepwise,
  type StepwiseOptions,
  type StepwiseStep,
} from './subsets'
export {
  bayesianLinearEvidence,
  evidenceMaximisation,
  evidenceMaximisationSteps,
  type EvidenceMaximisation,
  type EvidenceMaximisationOptions,
  type EvidenceOptions,
  type EvidenceProblem,
  type EvidenceState,
  type EvidenceTerms,
  type LinearEvidence,
} from './evidence'
export {
  ridgeHyperDescentSteps,
  ridgeHypergradient,
  type RidgeHyperDescentProblem,
  type RidgeHyperDescentState,
  type RidgeHypergradient,
} from './hypergradient'
export {
  ruleFitPath,
  ruleFitPredict,
  ruleImportances,
  type RuleFitOptions,
  type RuleFitPath,
  type RuleFitTerm,
} from './ruleFit'
export { linearAlgorithms, linearFunctions } from './registry'
