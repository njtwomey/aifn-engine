/**
 * `aifn-compute/learning`: the machinery of learning, from the estimator protocol to evaluation, as sklearn.base,
 * sklearn.metrics, sklearn.gaussian_process.kernels, sklearn.pipeline, sklearn.model_selection,
 * sklearn.calibration and sklearn.inspection.
 *
 * - `estimators`: the estimator protocol: fitted models' capabilities and their guards, datasets and row selection,
 *   predictive distributions, capability mixins (`withDecision`, `readout`), `defineModel`, and `evaluate`.
 * - `kernels`: covariance kernels $k(\xvec, \xvec')$ (`rbf`, `matern`, ...), their Gram matrices and log-space
 *   hyperparameters.
 * - `losses`: training losses (classification, regression, divergences, representation, adversarial, preference and
 *   others), each with its registry metadata.
 * - `metrics`: evaluation metrics (classification, curves, scoring rules, regression, clustering, ranking), defined
 *   once with the capability they read.
 * - `compose`: pipelines, column-wise transforms of tables, and target transforms with pushed-forward predictives.
 * - `validate`: splitters, cross-validation, grid and random search, and nested cross-validation.
 * - `calibration`: maps from a classifier's scores to calibrated probabilities (isotonic regression).
 * - `conformal`: split conformal prediction sets and intervals around any fitted model.
 * - `explain`: explanations of predictions (Shapley values, LIME, gradient attributions and others).
 * - `off-policy`: off-policy evaluation of a target policy from logged bandit feedback, slates and propensities.
 * - `subgroups`: subgroup discovery and exceptional model mining over a table.
 *
 * The family re-exports the most used names: `evaluate`, `rbf`, `matern`, `getLoss`, `listLosses`, `getMetric`,
 * `listMetrics`, `accuracy`, `auroc`, `pipeline`, `crossValidate`, `kFold`, `isotonicRegression` and
 * `poolAdjacentViolatorsSteps`; everything else is imported from its module.
 */

export { evaluate } from './estimators'
export { rbf, matern } from './kernels'
export { getLoss, listLosses } from './losses'
export { getMetric, listMetrics, accuracy, auroc } from './metrics'
export { pipeline } from './compose'
export { crossValidate, kFold } from './validate'
export { isotonicRegression, poolAdjacentViolatorsSteps } from './calibration'
