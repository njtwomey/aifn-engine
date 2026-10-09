/**
 * `aifn-compute/optim/derivative-free`: minimisation of $f : \reals^n \to \reals$ from its values alone, with no
 * gradient.
 *
 * - Deterministic: `nelderMead`, the simplex method of Nelder and Mead as specified by Lagarias et al. (1998), with
 *   optional dimension-adaptive coefficients; a good first choice in a few dimensions.
 * - Stochastic: `simulatedAnnealing`, a Metropolis random walk under a falling temperature, which can leave a local
 *   minimum; `cmaEs`, the covariance matrix adaptation evolution strategy, which learns the scale and orientation of
 *   the search distribution from its samples.
 * - `derivativeFreeAlgorithms` registers the three for generic views and workers.
 *
 * Each method is an `Algorithm` started with `{ x0 }`, to be run with `run` or `trace`; the objective returns a number
 * (or an object with a `value`). The state's `x` and `value` are the current iterate: the best vertex for Nelder–Mead,
 * the walk's position for simulated annealing (whose best point is `best`), and the distribution mean for CMA-ES.
 * Divergence is reported in `diverged`, not thrown. The stochastic methods draw only from the runner's step stream,
 * so a run is reproduced by its root `stream`. Brent's and golden-section minimisation of one variable is
 * `minimizeScalar` in `aifn-compute/numerics/roots`.
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
