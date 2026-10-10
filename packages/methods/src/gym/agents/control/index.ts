/**
 * `aifn-methods/gym/agents/control`: controllers and policy search for classic-control environments.
 *
 * - Model-based, from autodiff Jacobians of a differentiable dynamics model (`lineariseDynamics`): the pendulum's
 *   energy swing-up with an LQR hand-over (`swingUpAgent`, its gain from `pendulumLqr`), and bang-bang control along
 *   the LQR force for two-action problems such as the cart-pole (`lqrBangBangAgent`). They learn nothing.
 * - Policy search over linear threshold policies $a = [\wvec \cdot (\ovec, 1) > 0]$: the cross-entropy method
 *   (`crossEntropyAgent`), and one fixed policy (`linearPolicyAgent`).
 *
 * Controllers act deterministically from the observation; the model-based ones throw `TypeError` at `init` when the
 * environment has no suitable dynamics model.
 */

export {
  lineariseDynamics,
  pendulumLqr,
  swingUpAgent,
  type Linearisation,
  type PendulumPlant,
  type SwingUpOptions,
  type SwingUpState,
  type TorqueAction,
} from './swing-up'
export {
  crossEntropyAgent,
  linearPolicyAgent,
  lqrBangBangAgent,
  type CrossEntropyOptions,
  type CrossEntropyState,
  type GenerationSummary,
  type LqrBangBangOptions,
  type LqrBangBangState,
} from './cartpole'
export { controlAgentFunctions } from './registry'
