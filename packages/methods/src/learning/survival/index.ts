/**
 * `aifn-methods/learning/survival`: regression models for right-censored times, as lifelines' `CoxPHFitter` and
 * `WeibullAFTFitter`.
 *
 * - Proportional hazards: `coxPh` fits the Cox model by Newton's method on the partial likelihood (Efron or Breslow
 *   ties), with standard errors and Breslow's baseline cumulative hazard; `coxSurvival` gives a subject's survival
 *   curve, and `coxPartialLikelihood` the partial log-likelihood at any coefficients.
 * - Accelerated failure time: `aftModel` fits a Weibull or log-normal model by maximum likelihood (autodiff and
 *   L-BFGS); `aftSurvival` gives a subject's survival function, and `timeRatios` the factor $e^{\beta_k}$ by which each
 *   covariate stretches time.
 * - Evaluation: `harrellConcordance`, the share of comparable pairs a risk score ranks correctly.
 *
 * Data are covariates ($n \times p$), times ($n$ values) and event flags (1 an event, 0 right-censored). The
 * estimators that name no model (Kaplan–Meier, Nelson–Aalen, the log-rank test) are in
 * `aifn-compute/probability/tests`, and censored data generators in `aifn-methods/data/synthetic`.
 * `survivalFunctions` lists the module's functions by key.
 */

export { coxPartialLikelihood, coxPh, coxSurvival, harrellConcordance, type CoxFit, type CoxOptions } from './cox'
export { aftModel, aftSurvival, timeRatios, type AftFamily, type AftFit } from './aft'
export { survivalFunctions } from './registry'
