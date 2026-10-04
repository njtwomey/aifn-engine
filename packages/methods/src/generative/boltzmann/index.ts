/**
 * `aifn-methods/generative/boltzmann`: energy-based networks of binary units. The restricted Boltzmann machine with
 * CD-k and persistent CD, exact log-likelihood for small hidden layers, and a streamed training run (`rbm.ts`); the
 * deep belief network stacked greedily from RBMs, with ancestral sampling and a discriminative fine-tune
 * (`dbn.ts`); the classical Hopfield network with Hebbian weights and asynchronous recall, and the modern (dense) Hopfield network
 * whose update is attention, with a capacity curve comparing them (`hopfield.ts`).
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
