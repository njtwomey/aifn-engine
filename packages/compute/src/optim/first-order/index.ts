/**
 * `aifn-compute/optim/first-order`: first-order methods, which use $f$ and $\nabla f$ only, as traceable algorithms
 * and as update rules for training loops.
 *
 * - Gradient methods on one vector, each a step-through `Algorithm` started from `{ x0 }`: `gradientDescent` (fixed,
 *   scheduled or line-searched step), `momentum` (heavy ball), `nesterov`, and the adaptive `adagrad`, `rmsprop`,
 *   `adam` and `adamw`, which scale each coordinate by its own gradient history.
 * - Conjugate gradients: `conjugateGradient` (nonlinear, Fletcher–Reeves or Polak–Ribière, for a smooth $f$),
 *   `linearConjugateGradient` and `solveConjugateGradient` (for $\Amat\xvec = \bvec$ with $\Amat$ symmetric positive
 *   definite, needing only products $\Amat\vvec$).
 * - `coordinateDescent`: one coordinate per step, chosen cyclically, at random or greedily.
 * - Update rules on parameter pytrees in optax's style, for training loops whose objective changes at every step:
 *   `sgdRule`, `adagradRule`, `rmspropRule`, `adamRule`, `adamwRule`, with `applyUpdates` to add the updates,
 *   `chainRules` to compose rules, `clipByGlobalNorm` and `globalNorm`, and `stepSizeAt` to read a `StepSize`.
 *
 * The gradient methods run the update rules, so each method has one definition, and their steps are written with
 * primitives so `unrolled` differentiates through them. Non-convergence, divergence, a stalled line search and an
 * indefinite matrix are reported in the state, not thrown. `firstOrderAlgorithms` and `firstOrderFunctions` register
 * the algorithms and the rules.
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
