/**
 * `aifn-compute/learning/losses`: training losses, each a composition of `aifn-compute/foundation/tensor` primitives with its registry
 * metadata: classification, regression, divergence, representation (InfoNCE, CLIP's learnable temperature),
 * adversarial (GAN games, gradient penalty), energy-based (contrastive divergence), mixture density (MDN heads)
 * weak-supervision (uPU, nnPU, LLP proportion, complementary-label) and preference (`dpo`, `ipo`, `kto`, `simpo`,
 * `orpo`) losses; `lossRegistry`, `getLoss`, `listLosses`. Ranking and retrieval losses are in `aifn-methods/retrieval/losses`.
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
