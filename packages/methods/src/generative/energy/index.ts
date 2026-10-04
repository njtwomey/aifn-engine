/**
 * `aifn-methods/generative/energy`: energy-based models on toy data, centred on JEM (Grathwohl et al., 2019): a
 * softmax classifier's logits read as an energy E(x) = −logsumexp_y f(x)[y], trained with cross-entropy plus the
 * contrastive-divergence term of log p(x).
 *
 * - `classifier`, `classifierLogits`, `classifierEnergy`, `classEnergy`, `classifierScore` (∇ₓ log p(x) or log p(x | y), for
 *   Langevin), `uniformBox` (restarts), `jemTraining` (JEM or plain cross-entropy, traceable), `logitShift`.
 * - A streamed run for a worker: `jemRun` (both models side by side, with fields, samples, calibration and
 *   out-of-distribution scores at checkpoints).
 */

export {
  classEnergy,
  classifier,
  classifierEnergy,
  classifierScore,
  jemTraining,
  classifierLogits,
  logitShift,
  uniformBox,
  type Classifier,
  type ClassifierOptions,
  type JemTrainingOptions,
  type LogitShift,
} from './jem'
export { jemRun, type JemCheckpoint, type JemData, type JemRun, type JemRunOptions, type JemTrack } from './run'
export { energyAlgorithms, energyFunctions } from './registry'
