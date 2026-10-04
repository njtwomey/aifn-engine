/**
 * `aifn-methods/learning/survival`: regression models for right-censored times. The Cox proportional-hazards model by
 * Newton's method on the partial likelihood with Breslow or Efron ties, Breslow's baseline hazard and per-subject
 * survival curves (`cox.ts`); Weibull and log-normal accelerated-failure-time models by maximum likelihood (`aft.ts`).
 * The estimators that name no model (Kaplan–Meier, Nelson–Aalen, the log-rank test) are compute's
 * `aifn-compute/probability/tests`; censored data generators are in `aifn-methods/data/synthetic`.
 */

export { coxPartialLikelihood, coxPh, coxSurvival, harrellConcordance, type CoxFit, type CoxOptions } from './cox'
export { aftModel, aftSurvival, timeRatios, type AftFamily, type AftFit } from './aft'
export { survivalFunctions } from './registry'
