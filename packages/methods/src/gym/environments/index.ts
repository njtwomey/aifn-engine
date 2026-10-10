/**
 * `aifn-methods/gym/environments`: environments on the gym protocol, from one-step bandits to finite MDPs and classic
 * control.
 *
 * - Bandits, horizon 1, whose oracle knows each arm's mean for pseudo-regret: `bernoulliBandit`, `gaussianBandit`, and
 *   the linear contextual `linearBandit`, whose arms are feature vectors. Every round draws every arm's reward, so
 *   agents on one stream see common random numbers.
 * - Finite MDPs as tables (`TabularMdp`): `gridworld` (Russell and Norvig's noisy grid), `cliffWalking`, `maze` (from
 *   rows of text; built-in layouts in `MAZES`) and `frozenLake` (Gymnasium's maps in `FROZEN_LAKE_MAPS`).
 * - Finite MDPs as environments: `mdpEnvironment` makes any `TabularMdp` one, with a tabular model for planners, an
 *   oracle of the optimal values, legal-action masking and a grid render; `gridworldEnvironment`,
 *   `cliffWalkingEnvironment`, `mazeEnvironment` and `frozenLakeEnvironment` are the builders' registered forms.
 * - Classic control, with continuous observations and differentiable dynamics models, in the child module `control`:
 *   `pendulumEnvironment` (with `pendulumEnergy` and `wrapAngle`) and `cartPoleEnvironment`.
 *
 * Every environment is plain data with pure `reset` and `step` that draw only from the stream they are given, and
 * registers under kind `environment`; `environmentFunctions` holds the registry entries of the builders.
 */

export {
  bernoulliBandit,
  gaussianBandit,
  linearBandit,
  type BanditEnvironment,
  type BernoulliBanditOptions,
  type GaussianBanditOptions,
  type LinearBanditOptions,
} from './bandits'
export {
  cliffWalking,
  cliffWalkingEnvironment,
  FROZEN_LAKE_MAPS,
  frozenLake,
  frozenLakeEnvironment,
  gridworld,
  gridworldEnvironment,
  MAZES,
  maze,
  mazeEnvironment,
  mdpEnvironment,
  type GridworldOptions,
  type HorizonOption,
  type MazeEnvironmentOptions,
  type MazeOptions,
  type MdpEnvironment,
} from './gridworlds'
export {
  pendulumEnergy,
  pendulumEnvironment,
  wrapAngle,
  type PendulumAction,
  type PendulumEnvironment,
  type PendulumOptions,
  type PendulumParameters,
  type PendulumState,
} from './control'
export {
  cartPoleEnvironment,
  type CartPoleEnvironment,
  type CartPoleOptions,
  type CartPoleParameters,
  type CartPoleState,
} from './control'
export { environmentFunctions } from './registry'
