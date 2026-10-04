/**
 * `aifn-compute/optim/derivative-free`: derivative-free methods: Nelder–Mead, CMA-ES and simulated annealing. Brent's and
 * golden-section minimisation of one variable is `minimizeScalar` in `aifn-compute/numerics/roots`.
 */

export {
  nelderMead,
  type NelderMeadOperation,
  type NelderMeadOptions,
  type NelderMeadState,
  type NelderMeadTrial,
} from './nelderMead'
export {
  cmaEs,
  simulatedAnnealing,
  type CmaEsOptions,
  type CmaEsState,
  type SimulatedAnnealingOptions,
  type SimulatedAnnealingState,
} from './stochastic'
export { derivativeFreeAlgorithms } from './registry'
