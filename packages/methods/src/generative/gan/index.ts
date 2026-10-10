/**
 * `aifn-methods/generative/gan`: generative adversarial networks on low-dimensional toy data.
 *
 * - The networks: `gan` builds an MLP generator $G$ and discriminator $D$; `latents` draws
 *   $\zvec \sim \Gauss(\zeros, \Imat)$, `generatePoints` and `discriminate` apply $G$ and $D$ (traced, for losses),
 *   and `sampleGenerator` and `scoresAt` read them out as plain arrays.
 * - Training: `ganTraining` is `aifn-compute/nn/training`'s `adversarialTraining` with a game of
 *   `aifn-compute/learning/losses` (minimax, non-saturating, Wasserstein with gradient penalty, hinge), and
 *   `optimizerOf` builds an update rule from plain data.
 * - Diagnostics against a known density (a labelled mixture of `aifn-methods/generative`'s shared layer, such as a
 *   classification truth's `model` from `aifn-methods/data`): `modeCoverage` counts the modes holding high-quality
 *   generated points, and `optimalDiscriminator` gives $D^*$ with a KDE of $p_g$.
 * - A streamed run for a worker: `ganRun` yields snapshots of the losses and of checkpoints (generated points, the
 *   discriminator on a grid, $D^*$, the mode coverage).
 * - The registry: `ganAlgorithms` and `ganFunctions`.
 *
 * Points are rows, $[n, d]$; the networks' parameters are `Params[]`, as their `init` returns them. Every draw comes
 * from a stream, so a run is deterministic in its seed.
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
