/**
 * `aifn-methods/learning/mixture-of-experts`: the mixture of experts and the hierarchical mixture of experts as
 * statistical models (Jacobs, Jordan, Nowlan and Hinton, 1991; Jordan and Jacobs, 1994), over the compute layer of
 * `aifn-compute/nn/experts`.
 *
 * - Building a model: `moeSpec` fills in a `MoeConfig` (linear or MLP experts, any routing gate or a two-level
 *   hierarchy, the conditional-mixture or blended-output objective) and `moeModel` builds it, with `init` for its
 *   parameters.
 * - Evaluating it: `moeForward` (the routing and every expert's output), `expertLogLikelihood`, `moeLoss` (the data
 *   term and the load-balancing, importance and router z auxiliary losses) and `moePredict` (plain numbers: gate
 *   weights, predictions, the assigned expert).
 * - Fitting it: `moeEm`, EM as a step-through algorithm (linear experts, a dense gate and the mixture objective, as
 *   `emApplies` checks; `moeNegativeLogLikelihood` is its objective), or `moeTraining`, Adam on any model with the
 *   auxiliary losses.
 * - For figures: `mixtureOfExpertsRun`, a generator that trains by EM, Adam or L-BFGS and streams the curves and
 *   checkpoints; the registry `mixtureOfExpertsAlgorithms`.
 *
 * Inputs are $T \times d$ and targets $T$ floats or 0/1 labels. The mixture objective is the likelihood of
 * $p(y \mid \xvec) = \sum_i g_i(\xvec) p_i(y \mid \xvec)$; the blend fits $\hat y = \sum_i g_i(\xvec) f_i(\xvec)$.
 * `moeForward`, `expertLogLikelihood` and `moeLoss` are differentiable in the parameters.
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
