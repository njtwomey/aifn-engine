/**
 * `aifn-compute/learning`: the machinery of learning, as sklearn.base, sklearn.metrics, sklearn.gaussian_process.kernels,
 * sklearn.pipeline and sklearn.model_selection: estimator protocols, kernels, losses, metrics, composition and
 * validation. Children: estimators, kernels, losses, metrics, compose, validate.
 */

export { evaluate } from './estimators'
export { rbf, matern } from './kernels'
export { getLoss, listLosses } from './losses'
export { getMetric, listMetrics, accuracy, auroc } from './metrics'
export { pipeline } from './compose'
export { crossValidate, kFold } from './validate'
export { isotonicRegression, poolAdjacentViolatorsSteps } from './calibration'
