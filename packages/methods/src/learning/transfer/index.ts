/**
 * `aifn-methods/learning/transfer`: transfer, continual and meta-learning set-ups on small problems. Alignment penalties
 * (MMD², CORAL, the gradient-reversal layer) and unsupervised domain adaptation (source only, MMD, CORAL, DANN) with a
 * streamed run; label-shift estimation (BBSE, EM) and posterior correction; continual learning (naive fine-tuning,
 * EWC, experience replay); MAML (second and first order) on sinusoid regression against a pretrained baseline; prototypical networks trained
 * episodically on few-shot direction classes.
 */

export { coralLoss, gradientReversal, mmdSquared } from './alignment'
export {
  domainAdaptationRun,
  type AdaptationCheckpoint,
  type AdaptationMethod,
  type AdaptationOptions,
  type AdaptationRun,
} from './adaptation'
export { blackBoxShiftEstimate, priorShiftEm, reweightPosteriors } from './label-shift'
export { continualRun, type ContinualMethod, type ContinualOptions, type ContinualRun } from './continual'
export {
  mamlRun,
  sineSamples,
  sineTasks,
  type MamlCheckpoint,
  type MamlOptions,
  type MamlRun,
  type SineTask,
} from './meta'
export {
  fewShotEpisode,
  prototypeLogits,
  prototypicalRun,
  type FewShotEpisode,
  type FewShotEpisodeOptions,
  type PrototypicalCheckpoint,
  type PrototypicalOptions,
  type PrototypicalRun,
} from './prototypical'
export { transferFunctions } from './registry'
