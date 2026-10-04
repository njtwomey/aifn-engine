/**
 * `aifn-compute/nn/training`: the training loop (`trainingLoop`, a traceable `Algorithm` over any pytree update rule of
 * `aifn-compute/optim/first-order`), two-player adversarial training (`adversarialTraining`), persistent contrastive divergence
 * for energy-based models (`contrastiveDivergence`), full-batch training by a vector method (`fullBatchTraining`), one
 * state shape over first-order or L-BFGS training (`methodTraining`), differentially private training (DP-SGD,
 * `privateTraining`) and activation inspection (`activations`, `recordActivations`,
 * `inspect`).
 */

export { trainingLoop, type Batch, type TrainingOptions, type TrainingState } from './train'
export { activations, inspect, recordActivations, type Activations, type Inspection } from './inspect'
export { trainingAlgorithms } from './registry'
export { adversarialTraining, type AdversarialTrainingOptions, type AdversarialTrainingState } from './adversarial'
export {
  contrastiveDivergence,
  type ContrastiveDivergenceState,
  type ContrastiveDivergenceTrainingOptions,
} from './energy'
export { fullBatchTraining, treeObjective, type FullBatchOptions, type FullBatchState } from './fullBatch'
export { methodTraining, type MethodTrainingState, type TrainingMethod } from './method'
export { privateTraining, type PrivateTrainingOptions, type PrivateTrainingState } from './private'
