/**
 * `aifn-methods/learning/generalised/gam`: generalised additive models: terms (smooths, cyclic, thin-plate, tensor
 * products, factor and linear terms, shape constraints) and their bases, the penalised problem (`gamProblem`: design,
 * smoothing parameters by REML, GCV or fixed, the P-IRLS optimum), fitters that solve it step by step (P-IRLS,
 * backfitting, gradient descent, SGD, Adam, L-BFGS), the model at any coefficients (`gamModel`) and the estimator
 * `gam`; expectile GAMs, GAMLSS (`gamlssRs`, the RS algorithm over distributional families) and the explainable boosting
 * machine.
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
