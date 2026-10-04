/**
 * `aifn-methods/gym/agents/control`: model-based controllers for classic-control environments: LQR about an
 * equilibrium from autodiff Jacobians of a differentiable dynamics model (`lineariseDynamics`), and the pendulum's energy
 * swing-up with an LQR hand-over (`swingUpAgent`).
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
