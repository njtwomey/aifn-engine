/**
 * `aifn-compute/learning/losses`: training losses, each a composition of `aifn-compute/foundation/tensor` primitives
 * with its registry metadata.
 *
 * - Classification: `binaryCrossEntropyWithLogits` (exact for any logit) and `binaryCrossEntropy` (from
 *   probabilities), `softmaxCrossEntropy` (labels or probability rows, with label smoothing), `focalLoss` and
 *   `softmaxFocalLoss`; the margin surrogates of the 0–1 loss for labels in $\{-1, +1\}$ (`hinge`, `squaredHinge`,
 *   `logisticLoss`, `exponentialLoss`, `modifiedHuber`, and the table `surrogates`); the multiclass hinges
 *   `crammerSingerHinge` (the worst rival) and `westonWatkinsHinge` (every rival).
 * - Regression, each named by the statistic its minimiser estimates: `meanSquaredErrorLoss` (mean),
 *   `meanAbsoluteErrorLoss` (median), `huber` and `logCosh` (in between), `pinball` (a quantile), `expectileLoss` (an
 *   expectile), and the negative log-likelihoods `poissonNll` and `gaussianNll`.
 * - Mixture density heads: `mixtureDensityNll` on raw outputs of width `mixtureHeadSize`, `mixtureDensityParams` (the
 *   differentiable split into weights, means and scales) and `mixtureDensityHead` (plain numbers, with moments, draws
 *   and modes).
 * - Divergences: `klLoss` between distribution objects, `jensenShannonLoss` between probability vectors,
 *   `distillation` between teacher and student logits.
 * - Representation: `infoNce` (in-batch negatives, symmetric as in CLIP) and `learnedTemperature`.
 * - Adversarial: `discriminatorLoss` and `generatorLoss` for the games of `adversarialGames`, and the WGAN-GP
 *   `gradientPenalty` with its weight `GRADIENT_PENALTY_WEIGHT`.
 * - Energy-based: `contrastiveDivergenceLoss`.
 * - Weak supervision: positive–unlabelled risks `unbiasedPu` and `nonNegativePu`, `proportionLoss` (label
 *   proportions of bags) and `complementaryLabelLoss`.
 * - Preference, on whole-response log-probabilities: `dpo`, `ipo` and `kto` against a reference policy, `simpo` and
 *   `orpo` without one.
 * - Registry and definition: `lossRegistry`, `getLoss`, `listLosses`; `defineLoss`, `isLoss`; `oneHot`; and, for
 *   losses defined elsewhere, `reduce`, `constantTarget`, `flatValues` and `expectRank`.
 *
 * Predictions may be traced, so every loss is differentiable in them; targets and labels are read as constants. Most
 * losses end with an options object whose `reduction` (`mean` by default, `sum` or `none`) combines the per-example
 * values, as PyTorch's losses do; the adversarial, energy-based and positive–unlabelled losses return their means
 * directly. Ranking and retrieval losses are in `aifn-methods/retrieval/losses`.
 */

export {
  defineLoss,
  isLoss,
  oneHot,
  type Loss,
  type LossFamily,
  type LossFunction,
  type LossInfo,
  type LossSpec,
  type LossInput,
  type Reduction,
  type ReductionOptions,
  type Target,
} from './core'
export {
  binaryCrossEntropy,
  binaryCrossEntropyWithLogits,
  crammerSingerHinge,
  exponentialLoss,
  focalLoss,
  hinge,
  logisticLoss,
  modifiedHuber,
  softmaxCrossEntropy,
  softmaxFocalLoss,
  squaredHinge,
  surrogates,
  westonWatkinsHinge,
  type BinaryCrossEntropyOptions,
  type FocalOptions,
  type MulticlassHingeOptions,
  type SoftmaxCrossEntropyOptions,
  type SurrogateName,
} from './classification'
export {
  expectileLoss,
  gaussianNll,
  huber,
  logCosh,
  meanAbsoluteErrorLoss,
  meanSquaredErrorLoss,
  pinball,
  poissonNll,
  type ExpectileLossOptions,
  type GaussianNllOptions,
  type HuberOptions,
  type PinballOptions,
  type PoissonNllOptions,
} from './regression'
export { distillation, jensenShannonLoss, klLoss, type DistillationOptions } from './divergence'
export {
  dpo,
  ipo,
  kto,
  orpo,
  simpo,
  type DpoOptions,
  type IpoOptions,
  type KtoOptions,
  type OrpoOptions,
  type SimpoOptions,
} from './preference'
export { infoNce, learnedTemperature, type InfoNceOptions } from './representation'
export {
  adversarialGames,
  discriminatorLoss,
  generatorLoss,
  gradientPenalty,
  GRADIENT_PENALTY_WEIGHT,
  type AdversarialGame,
} from './adversarial'
export { contrastiveDivergenceLoss, type ContrastiveDivergenceOptions } from './energy'
export {
  mixtureDensityHead,
  mixtureDensityNll,
  mixtureDensityParams,
  mixtureHeadSize,
  type MixtureDensity,
  type MixtureDensityNllOptions,
  type MixtureHeadOptions,
  type MixtureHeadParts,
  type MixtureMode,
  type MixtureModeOptions,
  type MixtureRow,
  type ScaleLink,
} from './mixture'
export {
  complementaryLabelLoss,
  nonNegativePu,
  proportionLoss,
  unbiasedPu,
  type ComplementaryLabelOptions,
  type NonNegativePuOptions,
  type ProportionLossOptions,
  type PuOptions,
  type PuSurrogate,
} from './weak'
export { getLoss, listLosses, lossRegistry } from './registry'

// Helpers for defining losses outside this module (the ranking and retrieval losses of `aifn-methods/retrieval`): the
// input conventions every loss here follows.
export { constant as constantTarget, expectRank, flatValues, reduce } from './core'
