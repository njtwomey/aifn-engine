/**
 * `aifn-methods/gym/environments`: environments on the gym protocol. Bandits (`bandits.ts`: Bernoulli, Gaussian, linear)
 * and finite MDPs (`gridworlds.ts`: the grid builders and `mdpEnvironment`, which makes any `TabularMdp` an
 * environment with a tabular model, an oracle and a grid render); child: `control` (classic control).
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
