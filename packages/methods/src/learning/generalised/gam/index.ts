/**
 * `aifn-methods/learning/generalised/gam`: generalised additive models, from their terms and penalised problem to the
 * fitters that solve it, with expectile GAMs, GAMLSS and the explainable boosting machine.
 *
 * - Terms: `s` (P-splines, with optional shape constraints and `by` variables), `cyclic`, `thinPlate`, `te` (tensor
 *   products), `linearTerm` and `factorTerm`; `termBasis` gives a built term's basis and penalties on a grid.
 * - The penalised problem: `gamDesign` (the model matrix and penalties), `gamProblem` (the smoothing parameters by
 *   REML, GCV/UBRE or fixed, the active shape-constraint rows and the P-IRLS optimum), `resolveLikelihood`, and
 *   `smoothingPath` and `smoothingProfile`, the criterion and EDF against a common $\log\lambda$.
 * - Fitters of one problem, each a traceable algorithm from the same start: `gamPirls`, `gamBackfitting`,
 *   `gamGradientDescent`, `gamAdam`, `gamLbfgs` and `gamSgd`, chosen by name with `gamFitter`; `gamProblemChoices` and
 *   `gamTrainingRun` give choices and runs as plain data.
 * - Models: `gam`, the estimator (mgcv's `gam`, pygam's `GAM`); `gamModel`, the model at any coefficients, with EDF,
 *   partial effects, Bayesian bands and draws; `gamLinkBand`, the band on the link scale at new inputs.
 * - Expectiles: `expectileGam` and `expectileLaws` (LAWS, iteratively reweighted asymmetric least squares),
 *   `expectileFan` (several levels), and `expectileProblem` and `expectileTrainingRun`, the expectile objective for any
 *   fitter.
 * - GAMLSS: `gamlssProblem`, the RS algorithm `gamlssRs` and `gamlssTrace`, and `gamlssModel` (parameter and centile
 *   curves, quantile residuals, worm plot, GAIC).
 * - The explainable boosting machine: `explainableBoostingMachine` and its boosting loop `ebmBoosting` (interpret's
 *   EBM without interactions).
 * - Registries: `gamFitters`, `gamAlgorithms` and `gamFunctions`.
 *
 * Every GAM fitter minimises the same per-observation penalised deviance
 * $J(\betavec) = (D(\betavec) + \betavec^\top\Smat_\lambda\betavec)/(2n)$ at fixed $\lambda$, so their traces
 * compare step by step. Matrices inside the module are row-major `Float64Array`s; the public functions take tensors.
 * Errors are thrown as `ShapeError` (sizes that do not match) or `DomainError` (an invalid option or level).
 */

export {
  cyclic,
  factorTerm,
  linearTerm,
  s,
  te,
  termBasis,
  thinPlate,
  type BuiltTerm,
  type ShapeConstraint,
  type SmoothOptions,
  type TermBasis,
  type TermSpec,
} from './terms'
export {
  gamDesign,
  gamProblem,
  resolveLikelihood,
  smoothingPath,
  smoothingProfile,
  type GamData,
  type GamDesign,
  type GamEvaluation,
  type GamProblem,
  type GamSpec,
  type SmoothingMethod,
  type SmoothingPath,
} from './problem'
export {
  gamAdam,
  gamBackfitting,
  gamFitter,
  gamFitters,
  gamGradientDescent,
  gamLbfgs,
  gamPirls,
  gamProblemChoices,
  gamSgd,
  gamTrainingRun,
  type GamFitMethod,
  type GamProblemChoices,
  type GamRunRequest,
  type GamTrainingRun,
  type GamFitState,
  type GamFitterOptions,
  type GamGradientOptions,
  type GamSgdOptions,
} from './fitters'
export {
  gam,
  gamLinkBand,
  gamModel,
  type GamModel,
  type GamParams,
  type GamTermBasis,
  type PartialEffect,
} from './model'
export {
  expectileFan,
  expectileGam,
  expectileLaws,
  type ExpectileFan,
  type ExpectileGamParams,
  type ExpectileLawsOptions,
  type ExpectileState,
} from './expectile'
export {
  expectileProblem,
  expectileTrainingRun,
  type ExpectileFitMethod,
  type ExpectileFitState,
  type ExpectileRunRequest,
  type ExpectileTrainingRun,
} from './expectile-training'
export {
  gamlssModel,
  gamlssProblem,
  gamlssRs,
  gamlssTrace,
  type GamlssData,
  type GamlssModel,
  type GamlssParameterSpec,
  type GamlssProblem,
  type GamlssSmoothing,
  type GamlssSpec,
  type GamlssState,
} from './gamlss'
export { ebmBoosting, explainableBoostingMachine, type EbmModel, type EbmParams, type EbmState } from './ebm'
export { gamAlgorithms, gamFunctions } from './registry'
