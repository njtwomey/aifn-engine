/**
 * `aifn-methods/neural/semi-supervised`: semi-supervised training of small classifier networks by consistency
 * regularisation.
 *
 * - Training: `consistencyTraining`, one traceable algorithm over the unsupervised terms of the field ($\Pi$-model,
 *   temporal ensembling, mean teacher, VAT, UDA, MixMatch and FixMatch with fixed, FlexMatch, FreeMatch or SoftMatch
 *   thresholds), chosen by a `ConsistencyMethod`; `consistencyRun` runs it from plain data and streams checkpoints of
 *   plain arrays, for a worker.
 * - The rules the methods share: `sharpen` (temperature sharpening of probability rows), `annealingThreshold` (UDA's
 *   training-signal annealing), `flexMatchThresholds` (per-class thresholds), `rampUpWeight` (the Gaussian ramp-up of
 *   the unsupervised weight) and `predictProbabilities` (a network's softmax as a flat array).
 * - `semiSupervisedNetworkAlgorithms` and `semiSupervisedNetworkFunctions`: the module's registry entries.
 *
 * Probabilities are row-major $n \times K$ `Float64Array`s; the arg max of their rows is `rowArgmax` of
 * `aifn-methods/learning/semi-supervised`, and mean teacher's teacher is `parameterEma` of
 * `aifn-compute/nn/training`. Every draw comes from the step's stream, so a run is deterministic in its seeds.
 */

export {
  annealingThreshold,
  consistencyTraining,
  flexMatchThresholds,
  predictProbabilities,
  rampUpWeight,
  sharpen,
  type AnnealingSchedule,
  type ConfidenceThresholdRule,
  type ConsistencyMethod,
  type ConsistencyOptions,
  type ConsistencyProblem,
  type ConsistencyState,
  type Strength,
} from './consistency'
export { consistencyRun, type ConsistencyCheckpoint, type ConsistencyRunOptions } from './run'
export { semiSupervisedNetworkAlgorithms, semiSupervisedNetworkFunctions } from './registry'
