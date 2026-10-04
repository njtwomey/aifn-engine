/**
 * `aifn-compute/optim/first-order`: first-order methods, each a traceable `Algorithm`: gradient descent, momentum, Nesterov,
 * AdaGrad, RMSprop, Adam, AdamW; nonlinear and linear conjugate gradient; coordinate descent. The update rules behind
 * them work on parameter pytrees in optax's style (`sgdRule`, `adamRule`, …, `applyUpdates`, `chainRules`), for
 * training loops whose objective changes at every step.
 */

export {
  adagradRule,
  adamRule,
  adamwRule,
  applyUpdates,
  chainRules,
  clipByGlobalNorm,
  globalNorm,
  rmspropRule,
  sgdRule,
  stepSizeAt,
  type AdamRuleOptions,
  type AdaptiveRuleOptions,
  type RmspropRuleOptions,
  type RuleState,
  type SgdRuleOptions,
  type StepSize,
  type UpdateRule,
} from './rules'

export {
  adagrad,
  adam,
  adamw,
  gradientDescent,
  momentum,
  nesterov,
  rmsprop,
  type AdamOptions,
  type AdaptiveOptions,
  type FirstOrderOptions,
  type FirstOrderState,
  type GradientDescentOptions,
  type MomentumOptions,
  type RmspropOptions,
} from './firstOrder'
export {
  conjugateGradient,
  linearConjugateGradient,
  solveConjugateGradient,
  type ConjugateGradientOptions,
  type ConjugateGradientSolution,
  type ConjugateGradientState,
  type ConjugateGradientVariant,
  type LinearConjugateGradientOptions,
  type LinearConjugateGradientState,
} from './conjugateGradient'
export {
  coordinateDescent,
  type CoordinateDescentOptions,
  type CoordinateDescentState,
  type CoordinateRule,
} from './coordinateDescent'
export { firstOrderAlgorithms, firstOrderFunctions } from './registry'
