/**
 * `aifn-methods/generative/boltzmann`: energy-based networks of binary units: the restricted Boltzmann machine, the
 * deep belief network stacked from it, and classical and modern Hopfield networks.
 *
 * - The restricted Boltzmann machine (`rbm.ts`): `rbm` builds one; `hiddenProbabilities` and `visibleProbabilities`
 *   are its conditionals, `freeEnergy` its free energy $F(\vvec)$; `logPartition` and `rbmLogLikelihood` are exact
 *   for at most 20 hidden units (summed over the $2^H$ hidden states).
 * - Sampling and training an RBM: `gibbsChain` runs block Gibbs sampling; `contrastiveDivergenceStep` is one CD-$k$
 *   update, or PCD-$k$ when given persistent chains; `rbmRun` streams a training run with the exact log-likelihood,
 *   weights and samples.
 * - The deep belief network (`dbn.ts`): `dbnUp` is the recognition pass, `dbnSample` ancestral sampling from the top
 *   RBM down, and `dbnRun` streams greedy layer-wise training with an optional discriminative fine-tune.
 * - Hopfield networks (`hopfield.ts`): classical, with `hebbianWeights`, asynchronous recall (`hopfieldRecall`) and
 *   `hopfieldEnergy`; modern (dense), whose update `modernHopfieldUpdate` is attention, with `modernHopfieldEnergy`.
 *   `corruptPattern` makes cues, `overlaps` scores a state against the patterns, and `capacityCurve` compares the two
 *   networks' recall as the number of patterns grows.
 *
 * Models, vectors and weights are plain arrays (weights row-major); matrices of rows may be any `MatrixLike`. The
 * functions return new values and leave their arguments unchanged, and everything random draws from a `Stream`, so
 * runs are deterministic in their seed.
 */

export {
  contrastiveDivergenceStep,
  freeEnergy,
  gibbsChain,
  hiddenProbabilities,
  logPartition,
  rbm,
  rbmLogLikelihood,
  rbmRun,
  visibleProbabilities,
  type CdOptions,
  type Rbm,
  type RbmCheckpoint,
  type RbmRun,
  type RbmRunOptions,
} from './rbm'
export { dbnRun, dbnSample, dbnUp, type Dbn, type DbnCheckpoint, type DbnRun, type DbnRunOptions } from './dbn'
export {
  capacityCurve,
  corruptPattern,
  hebbianWeights,
  hopfieldEnergy,
  hopfieldRecall,
  modernHopfieldEnergy,
  modernHopfieldUpdate,
  overlaps,
  type HopfieldRecall,
} from './hopfield'
export { boltzmannFunctions } from './registry'
