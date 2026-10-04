/**
 * `aifn-compute/probability/markov`: finite Markov chains on a transition matrix P: classification of states, stationary
 * distributions, absorption and hitting probabilities and times (first-step analysis), the distance to stationarity,
 * mixing time, spectral gap and reversibility, and a simulator (`simulateChain`, and `markovChainSteps` beside the
 * exact distribution p₀Pᵗ).
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
