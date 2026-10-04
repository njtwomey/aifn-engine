/**
 * `aifn-compute/inference/stochastic`: Monte Carlo inference: Metropolis–Hastings (random-walk, independence), Gibbs over
 * conditionals or blocks (block Gibbs) with Rao–Blackwellised estimates, slice sampling, Hamiltonian Monte Carlo and
 * NUTS, MALA and SGLD, batched short-run Langevin with persistent chains in a replay buffer, sequential Monte Carlo and the particle filter, multi-chain runs and diagnostics (R̂, ESS, MCSE;
 * the autocorrelation function is `aifn-compute/probability/stats`'s), and Gibbs sampling on discrete factor graphs
 * (`factorGraphGibbs`) and on model descriptions (`modelGibbs`, by enumeration and conjugate updates).
 */

export {
  factorGraphGibbs,
  gibbsMarginals,
  modelGibbs,
  type ConditionalKind,
  type FactorGraphGibbsOptions,
  type FactorGraphGibbsState,
  type ModelGibbsOptions,
  type ModelGibbsState,
} from './factorGibbs'
export type { AcceptRejectState, ChainStart, ChainState, LogDensity, VectorLike } from './types'
export {
  independenceMetropolis,
  metropolisHastings,
  randomWalkMetropolis,
  type IndependentProposal,
  type MetropolisState,
  type Proposal,
  type RandomWalkOptions,
} from './metropolis'
export {
  bivariateGaussianConditionals,
  conditionalMean,
  gaussianConditionals,
  gibbs,
  sliceSampler,
  type Block,
  type Conditional,
  type GibbsOptions,
  type GibbsState,
  type SliceOptions,
  type SliceState,
} from './gibbs'
export { type DualAveragingOptions, type DualAveragingState } from './adaptation'
export {
  divergenceLimit,
  hmc,
  leapfrog,
  nuts,
  type DivergenceThreshold,
  type HmcOptions,
  type HmcState,
  type Leapfrog,
  type NutsOptions,
  type NutsState,
} from './hamiltonian'
export {
  langevinParticles,
  mala,
  persistentLangevin,
  chainBuffer,
  sgld,
  unadjustedLangevin,
  type BatchScore,
  type ParticleLangevinOptions,
  type ParticleLangevinState,
  type PersistentDraw,
  type PersistentLangevinOptions,
  type ChainBuffer,
  type LangevinOptions,
  type LangevinState,
  type MinibatchModel,
  type SgldOptions,
  type SgldState,
} from './langevin'
export {
  particleFilter,
  resample,
  resamplingSchemes,
  temperedSmc,
  type FilterStart,
  type ParticleFilterOptions,
  type ParticleFilterState,
  type ResamplingScheme,
  type StateSpaceModel,
  type TemperedModel,
  type TemperedSmcOptions,
  type TemperedSmcState,
} from './smc'
export {
  raoBlackwell,
  sampleChains,
  type ChainsResult,
  type RaoBlackwellEstimate,
  type SampleChainsOptions,
} from './chains'
export {
  effectiveSampleSize,
  integratedAutocorrelationTime,
  monteCarloStandardError,
  splitRhat,
  summarise,
  type ChainSummary,
  type Chains,
  type EssMethod,
  type RhatMethod,
} from './diagnostics'
export { stochasticAlgorithms, stochasticFunctions } from './registry'
