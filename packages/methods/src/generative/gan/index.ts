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
 * - Semi-supervised GANs (Salimans et al., 2016): `semiSupervisedGan` builds a generator and a $K$-logit
 *   discriminator whose implicit $(K + 1)$-th class is "generated"; `semiSupervisedGanTraining` trains them with a
 *   feature-matching or non-saturating generator, a fixed sampler such as Dai et al.'s `complementSampler`, or none;
 *   `classAndFakeProbabilities`, `generatedPoints` and `initSemiSupervisedGan` read and start them, and
 *   `semiSupervisedGanRun` streams a run from plain data.
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
export {
  classAndFakeProbabilities,
  complementSampler,
  generatedPoints,
  initSemiSupervisedGan,
  semiSupervisedGan,
  semiSupervisedGanRun,
  semiSupervisedGanTraining,
  type CriticParams,
  type FakeSource,
  type FixedSampler,
  type SemiSupervisedGan,
  type SemiSupervisedGanCheckpoint,
  type SemiSupervisedGanOptions,
  type SemiSupervisedGanProblem,
  type SemiSupervisedGanRunOptions,
  type SemiSupervisedGanState,
  type SemiSupervisedGanTrainingOptions,
} from './semi-supervised'
export { ganAlgorithms, ganFunctions } from './registry'
