/**
 * `aifn-compute/learning/selection`: estimates of prediction error and rules for choosing among models, beside the
 * splitters and searches of `aifn-compute/learning/validate`.
 *
 * - Linear smoothers without refitting: `linearSmootherCv` (leave-one-out residuals $(y_i - \hat y_i)/(1 - S_{ii})$
 *   and GCV from a smoother matrix $\Smat$), `ridgeSmoother` (the ridge hat matrix) and `ridgeCv` (leave-one-out and
 *   GCV curves over a grid of penalties from one SVD, as scikit-learn's `RidgeCV`).
 * - Choosing from cross-validation: `oneStandardError`, the simplest candidate within one standard error of the best,
 *   read from a `gridSearch` or `randomSearch` result, a list of `crossValidate` results or a fold-score matrix.
 * - Resampling: `bootstrapPredictionError`, the leave-one-out bootstrap, .632 and .632+ estimates with the
 *   no-information rate and the out-of-bag counts, for any estimator whose model has `decide`.
 * - Penalised fit: `informationCriteria` (AIC, AICc, BIC) and `mallowsCp`, elementwise over candidates; Stein's
 *   unbiased risk estimate `sure`, in closed form for soft thresholding (`softThresholdSure`) and the lasso
 *   (`lassoSure`).
 * - Bayesian predictive criteria from an $S \times n$ matrix of pointwise log-likelihood draws: `waic` and `psisLoo`
 *   (with each observation's Pareto $\hat k$, as arviz-stats), the smoothing itself as `psisWeights`, and `dic` from
 *   the deviance at the draws and at the posterior mean.
 * - Averaging models from an $n \times J$ matrix of leave-one-out log densities (Yao et al., 2018):
 *   `stackingWeights` (the simplex weights that maximise the mixture's log score), `pseudoBmaWeights` and
 *   `pseudoBmaPlusWeights` (Bayesian-bootstrapped), as arviz-stats' `compare`.
 * - `nestedCvInterval`: a confidence interval for prediction error by nested cross-validation (Bates, Hastie and
 *   Tibshirani, 2024), with its bias correction and the naive interval beside it.
 *
 * Every estimate of error is lower-is-better except the elpd of `waic` and `psisLoo`, which is a log density (higher is
 * better; `waic` and `looic` give it on the deviance scale). Failures are reported where they arise: a leverage of 1
 * gives an infinite leave-one-out residual, a row never out of bag has a NaN `perPoint`, an unreliable importance
 * estimate has a large $\hat k$.
 */

export { linearSmootherCv, ridgeCv, ridgeSmoother, type LinearSmootherCv, type RidgeCv } from './smoothers'
export { oneStandardError, type OneStandardError, type OneStandardErrorOptions, type ScoredCandidates } from './rules'
export {
  bootstrapPredictionError,
  type BootstrapPredictionError,
  type BootstrapPredictionErrorOptions,
  type PointwiseLoss,
} from './bootstrap'
export {
  informationCriteria,
  lassoSure,
  mallowsCp,
  softThresholdSure,
  sure,
  type CandidateValues,
  type InformationCriteria,
  type MallowsCp,
} from './criteria'
export {
  dic,
  psisLoo,
  psisWeights,
  waic,
  type Dic,
  type ElpdEstimate,
  type PointwiseElpd,
  type PsisLoo,
  type PsisWeights,
  type Waic,
} from './predictive'
export { nestedCvInterval, type NestedCvInterval, type NestedCvIntervalOptions } from './nested'
export {
  pseudoBmaPlusWeights,
  pseudoBmaWeights,
  stackingWeights,
  type PseudoBmaPlusWeights,
  type PseudoBmaWeights,
  type StackingWeights,
} from './stacking'
export { selectionFunctions } from './registry'
