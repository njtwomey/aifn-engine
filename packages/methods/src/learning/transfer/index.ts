/**
 * `aifn-methods/learning/transfer`: transfer, continual and meta-learning set-ups on small problems.
 *
 * - Alignment penalties between two samples of features, differentiable in both: `mmdSquared` (Gaussian-kernel
 *   $\mathrm{MMD}^2$), `coralLoss` (covariances only) and `gradientReversal` (DANN's layer).
 * - Unsupervised domain adaptation on 2-d data: `domainAdaptationRun` trains a feature extractor and classifier with
 *   no alignment, MMD, CORAL or DANN, and streams accuracies, features and the decision field.
 * - Label shift, from a source classifier's outputs: `blackBoxShiftEstimate` (BBSE: importance weights from the
 *   confusion matrix), `priorShiftEm` (EM on the target posteriors) and `reweightPosteriors` (the corrected
 *   posteriors).
 * - Continual learning: `continualRun` trains one network on tasks in turn by naive fine-tuning, EWC or experience
 *   replay, and records the accuracy on every task.
 * - Meta-learning: `mamlRun` (second- or first-order MAML on sinusoid regression against a pretrained baseline, with
 *   its tasks from `sineTasks` and `sineSamples`) and `prototypicalRun` (prototypical networks trained episodically on
 *   `fewShotEpisode` direction classes, scored by `prototypeLogits`).
 * - The registry: `transferFunctions`.
 *
 * The runs are generators of snapshots for figures, deterministic in their `seed`; each trains with Adam and
 * clipped gradients. Labels are integers $0, \dots, K - 1$.
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
