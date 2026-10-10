/**
 * `aifn-methods/gym/environments/control`: classic-control environments with continuous observations and
 * differentiable dynamics models.
 *
 * - The inverted pendulum, Gymnasium's `Pendulum-v1`: `pendulumEnvironment`, stepped by RK4 on compute's ODE solver,
 *   with a box or a discrete torque action; `pendulumEnergy` (conserved by the free swing) and `wrapAngle`.
 * - The cart-pole, Gymnasium's `CartPole-v1`: `cartPoleEnvironment`, two push actions, stepped by explicit Euler as
 *   Gymnasium is, so trajectories match its to rounding.
 *
 * Each `model.transition` is written with tensor primitives, so it is differentiable in the state and the action
 * (`jacobian` for LQR, iLQR and MPC), and `step` calls the same equations on numbers.
 */

export {
  pendulumEnergy,
  pendulumEnvironment,
  wrapAngle,
  type PendulumAction,
  type PendulumEnvironment,
  type PendulumOptions,
  type PendulumParameters,
  type PendulumState,
} from './pendulum'
export {
  cartPoleEnvironment,
  type CartPoleEnvironment,
  type CartPoleOptions,
  type CartPoleParameters,
  type CartPoleState,
} from './cartpole'
