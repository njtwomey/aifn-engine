/**
 * `aifn-methods/learning/mixture-of-experts`: the mixture of experts and the hierarchical mixture of experts as
 * statistical models (Jacobs, Jordan, Nowlan and Hinton, 1991; Jordan and Jacobs, 1994) over the compute layer of
 * `aifn-compute/nn/experts`: linear or MLP experts, any routing gate or a two-level hierarchy, the conditional-mixture or
 * blended-output objective; fitted by EM (`moeEm`, a step-through algorithm) or by Adam with auxiliary losses
 * (`moeTraining`); and `mixtureOfExpertsRun`, a streaming training run for figures.
 */

export {
  expertLogLikelihood,
  moeForward,
  moeLoss,
  moeModel,
  moePredict,
  moeSpec,
  type AuxWeights,
  type ExpertKind,
  type GateChoice,
  type MoeConfig,
  type MoeForward,
  type MoeLoss,
  type MoeModel,
  type MoeObjective,
  type MoeParams,
  type MoePrediction,
  type MoeSpec,
} from './model'
export { emApplies, moeEm, moeNegativeLogLikelihood, type MoeData, type MoeEmOptions, type MoeEmState } from './em'
export {
  mixtureOfExpertsRun,
  moeTraining,
  type MoeHistory,
  type MoeRunOptions,
  type MoeSnapshot,
  type MoeTrainingOptions,
} from './training'
export { mixtureOfExpertsAlgorithms } from './registry'
