/**
 * `aifn-methods/generative/gan`: generative adversarial networks on low-dimensional toy data.
 *
 * - The networks and training: `gan` (an MLP generator and discriminator), `latents`, `generatePoints`,
 *   `discriminate`, `ganTraining` (`aifn-compute/nn/training`'s `adversarialTraining` with a game of `aifn-compute/learning/losses`:
 *   minimax, non-saturating, Wasserstein with gradient penalty, hinge), `sampleGenerator`, `scoresAt`.
 * - Diagnostics against a known density (a labelled mixture of `aifn-methods/generative`'s shared layer, e.g. a
 *   classification truth's `model` from `aifn-methods/data`): `modeCoverage`, `optimalDiscriminator` (D* with a KDE
 *   of p_g).
 * - A streamed run for a worker: `ganRun` (snapshots of losses and checkpoints), `optimizerOf`.
 */

export {
  discriminate,
  gan,
  ganTraining,
  generatePoints,
  latents,
  sampleGenerator,
  scoresAt,
  type Gan,
  type GanOptions,
  type GanTrainingOptions,
} from './gan'
export { modeCoverage, optimalDiscriminator, type ModeCoverage, type ModeCoverageOptions } from './diagnostics'
export {
  ganRun,
  optimizerOf,
  type GanCheckpoint,
  type GanData,
  type GanRun,
  type GanRunOptions,
  type OptimizerSpec,
} from './run'
export { ganAlgorithms, ganFunctions } from './registry'
