/**
 * `aifn-compute/inference/message-passing`: belief propagation on discrete factor graphs and on Gaussian Markov
 * random fields.
 *
 * - Discrete belief propagation: `beliefPropagation` runs sum-product (marginals) or max-product (max-marginals) to
 *   convergence, exact on trees in one sweep and loopy BP otherwise; `beliefPropagationSteps` steps it message by
 *   message or sweep by sweep, with a tree, flooding, sequential or explicit `Schedule`, damping and evidence.
 * - What the messages give: `factorBeliefs` (each factor's joint belief), `betheLogZ` (the Bethe approximation to
 *   $\log Z$, exact on trees) and `decodeBeliefs` (each variable's best value; the MAP after max-product on a tree).
 * - Gaussian BP: `gaussianBeliefPropagation` and `gaussianBeliefPropagationSteps`, on a precision $\Jmat$ and potential
 *   $\hvec$ in information form; the means are exact at a fixed point, the variances exact on trees, and a run that
 *   meets a non-positive precision stops as `diverged`.
 *
 * Messages are normalised and start uniform (zero for Gaussian BP); a run stops once a sweep changes no message by
 * more than its tolerance. None of it is differentiable. Exact inference on chains and small graphs is in
 * `aifn-compute/inference/exact`.
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
