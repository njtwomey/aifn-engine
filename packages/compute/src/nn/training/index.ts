/**
 * `aifn-compute/nn/training`: training loops for networks, as traceable algorithms over parameter trees.
 *
 * - Minibatch training: `trainingLoop`, with any pytree update rule of `aifn-compute/optim/first-order` (Adam by
 *   default), shuffled epochs, gradient clipping, and layer buffers (batch norm) carried in the state.
 * - Full-batch training: `fullBatchTraining` by a vector method of `aifn-compute/optim/minimize` (L-BFGS by default),
 *   over the flat objective `treeObjective`; `methodTraining` gives first-order and L-BFGS training one state shape.
 * - Special objectives: `adversarialTraining` (a critic and a generator in alternation, as in a GAN),
 *   `contrastiveDivergence` (energy-based models with persistent Langevin negatives, and JEM) and `privateTraining`
 *   (DP-SGD, reporting the $\varepsilon$ spent).
 * - Inspection: `recordActivations` and `activations` record every tapped activation of a forward pass; `inspect` adds
 *   the gradients of a loss with respect to each activation and parameter.
 *
 * Every algorithm runs with `run` and `trace` from its initial parameters and draws its randomness from the step's
 * stream, so a step is a pure function of its state and context. `trainingAlgorithms` holds their registry entries.
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
