/**
 * `aifn-compute/inference/message-passing`: belief propagation on factor graphs (sum- and max-product; tree, flooding,
 * sequential or explicit schedules; damping), beliefs, the Bethe free energy, and Gaussian belief propagation.
 */

export {
  beliefPropagation,
  beliefPropagationSteps,
  betheLogZ,
  decodeBeliefs,
  factorBeliefs,
  type BeliefPropagationOptions,
  type BeliefPropagationResult,
  type BeliefPropagationState,
  type MessageUpdate,
  type Schedule,
} from './bp'
export {
  gaussianBeliefPropagation,
  gaussianBeliefPropagationSteps,
  type GaussianBpOptions,
  type GaussianBpState,
} from './gaussianBp'
export { messagePassingAlgorithms, messagePassingFunctions } from './registry'
