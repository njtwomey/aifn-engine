/**
 * `aifn-compute/probability/markov`: finite Markov chains on a row-stochastic transition matrix $\Pmat$, analysed
 * exactly.
 *
 * - Construction and powers: `transitionMatrix` (validates $\Pmat$), `nStepTransition` ($\Pmat^k$) and
 *   `distributionAfter` ($\pvec_0\Pmat^t$).
 * - Structure: `classifyStates` (communicating classes, closed and transient states, absorbing states, periods).
 * - Long-run behaviour: `stationaryDistribution` ($\pivec\Pmat = \pivec$, for one closed class),
 *   `stationaryDistributions` (one per closed class) and `meanReturnTimes` (Kac's $1/\pi_i$).
 * - First-step analysis: `absorption` (the fundamental matrix, absorption probabilities, and the mean and variance of
 *   the time to absorption), `hittingProbabilities` and `expectedHittingTimes` of a target set.
 * - Convergence: `distanceToStationarity` (total variation from each start), `mixingTime` (with the relaxation-time
 *   bounds for a reversible chain), `spectralGap` (eigenvalues, gaps and relaxation time) and `isReversible`
 *   (detailed balance).
 * - Simulation: `simulateChain` (a path drawn from a stream), and `markovChainSteps`, an algorithm that steps a walker
 *   beside the exact $\pvec_0\Pmat^t$.
 * - Registries: `markovAlgorithms` and `markovFunctions`.
 *
 * Every function validates $\Pmat$ (square, non-negative, rows summing to 1) and throws `DomainError` otherwise.
 * States are numbered $0, \dots, n - 1$, distributions are row vectors, and matrices are small and dense: everything
 * but the simulation is computed by linear solves and matrix powers. None is differentiable.
 */

export {
  absorption,
  classifyStates,
  distanceToStationarity,
  distributionAfter,
  expectedHittingTimes,
  hittingProbabilities,
  isReversible,
  markovChainSteps,
  meanReturnTimes,
  mixingTime,
  nStepTransition,
  simulateChain,
  spectralGap,
  stationaryDistribution,
  stationaryDistributions,
  transitionMatrix,
  type Absorption,
  type ChainClasses,
  type MarkovChainState,
  type MixingTime,
  type Reversibility,
  type SpectralGap,
  type StationarityDistance,
} from './chain'
export { markovAlgorithms, markovFunctions } from './registry'
