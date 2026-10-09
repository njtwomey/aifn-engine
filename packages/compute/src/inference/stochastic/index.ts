/**
 * `aifn-compute/inference/stochastic`: Monte Carlo inference, from Markov chain samplers to sequential Monte Carlo,
 * with the diagnostics of their output.
 *
 * - Metropolis–Hastings: `metropolisHastings` (any proposal, with its log ratio), `randomWalkMetropolis` (Gaussian
 *   steps) and `independenceMetropolis` (a fixed proposal that must cover the tails).
 * - Gibbs sampling: `gibbs` over one conditional per coordinate or over `Block`s drawn jointly (block Gibbs), with the
 *   Gaussian conditionals of any partition (`gaussianConditionals`, `bivariateGaussianConditionals`) and the
 *   `conditionalMean` that `raoBlackwell` averages; `sliceSampler` for any target, with no step size to tune.
 * - Discrete and model-based Gibbs: `factorGraphGibbs` on a discrete factor graph, with `gibbsMarginals`, and
 *   `modelGibbs` on a model description, by enumeration and conjugate updates.
 * - Gradient-based samplers: `hmc` and `nuts` (multinomial or slice), built on `leapfrog`, with dual-averaging step
 *   size adaptation and divergence reports (`divergenceLimit`); `unadjustedLangevin` (biased by $O(h)$), `mala`
 *   (exact) and `sgld` (minibatch gradients, no accept step).
 * - Energy-based models: `langevinParticles` moves a batch of particles with one score evaluation per step, and
 *   `persistentLangevin` keeps short-run chains alive in a `chainBuffer`.
 * - Sequential Monte Carlo: `particleFilter` (bootstrap, adaptive resampling, log evidence), `temperedSmc` (a static
 *   target and its evidence), and `resample` by each of the `resamplingSchemes`.
 * - Several chains and their diagnostics: `sampleChains` runs $m$ chains into an $m \times n \times d$ tensor, which
 *   `effectiveSampleSize`, `integratedAutocorrelationTime`, `splitRhat` ($\hat R$), `monteCarloStandardError` and
 *   `summarise` read (the autocorrelation function is `aifn-compute/probability/stats`'s).
 * - The registry entries of the module: `stochasticAlgorithms` and `stochasticFunctions`.
 *
 * Every sampler is an `Algorithm`, started from `{ x0 }` (or its own start) and run with `run`, `trace` or
 * `sampleChains`; step $t$ draws only from its step stream, so a run is reproducible from its root stream and states
 * are plain data. A target is a `LogDensity`: an unnormalised `logDensity` and its `dim`, with `grad` in closed form
 * where the gradient-based samplers should not differentiate it. Failures are reported in the state (`diverged`,
 * `divergent`) rather than thrown; misuse (a wrong shape, an option out of range) throws.
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
