/**
 * `aifn-methods/generative/diffusion`: diffusion models on toy data (plan §8.2).
 *
 * - Schedules: `linearSchedule` (Ho et al.), `cosineSchedule` (Nichol & Dhariwal), `scheduleFromBetas`, each with
 *   β, α, ᾱ and SNR per step; `alphaBarAt`, `betaAt`. Continuous forward SDEs with their marginals: `vpSde`,
 *   `subVpSde`, `veSde`. One time axis: `stepTime` (τ = t/T) and `timeStep`, `scheduleSde` (the VP SDE through a
 *   discrete schedule, exact at every step) and `sdeSchedule` (a schedule read off an SDE); every sampler reports τ.
 * - The forward process: `forwardNoise` (a draw from q(x_t | x₀)), `forwardPosterior` (q(x_{t−1} | x_t, x₀)),
 *   `forwardProcess` (the Markov chain, traceable).
 * - Noise predictors ε̂(x, ᾱ), the common interface of every model: `predictNoise` and `scoreFromPredictor` for any
 *   Gaussian marginal, `predictClean` (Tweedie).
 * - Gaussian-mixture data with exact noised marginals: `gaussianMixtureData`, `sampleMixture`, `mixtureLogDensity`,
 *   `mixtureScore`, `mixtureNoisePredictor` (a perfect predictor, so demos need no training), `mixtureMoments`.
 * - A learned predictor: `denoiser` (an MLP from `aifn-compute/nn`), `noiseLevelFeatures`, `denoiserInput`,
 *   `denoiserTraining` (traceable), `networkNoisePredictor`.
 * - Samplers, each a traceable `Algorithm` over all particles: `ddpmSampler`, `ddimSampler` (with `ddimTimesteps`),
 *   `reverseSdeSampler` (Euler–Maruyama), `probabilityFlowSampler` (RK4 or Euler).
 *
 * Conventions: points are [n, d] tensors; discrete steps run t = 1 … T with ᾱ₀ = 1; continuous time runs over [0, 1]
 * and sampling stops at t = ε (default 10⁻³).
 */

export {
  alphaBarAt,
  betaAt,
  cosineSchedule,
  linearSchedule,
  scheduleFromBetas,
  scheduleSde,
  sdeSchedule,
  stepTime,
  subVpSde,
  timeStep,
  veSde,
  vpSde,
  type ForwardSde,
  type NoiseSchedule,
} from './schedules'
export { forwardNoise, forwardPosterior, forwardProcess, type ForwardState, type Noised } from './forward'
export { predictClean, predictNoise, scoreFromPredictor, type NoisePredictor } from './predictor'
export {
  gaussianMixtureData,
  mixtureLogDensity,
  mixtureMoments,
  mixtureNoisePredictor,
  mixtureScore,
  sampleMixture,
  type GaussianMixture,
} from './mixture'
export {
  denoiser,
  denoiserInput,
  denoiserTraining,
  networkNoisePredictor,
  noiseLevelFeatures,
  type Denoiser,
  type DenoiserOptions,
  type DenoiserTrainingOptions,
} from './denoiser'
export {
  ddimSampler,
  ddimTimesteps,
  ddpmSampler,
  probabilityFlowSampler,
  reverseSdeSampler,
  type ContinuousOptions,
  type DdimOptions,
  type DdpmOptions,
  type ProbabilityFlowOptions,
  type SamplerStart,
  type SamplerState,
} from './samplers'
export { diffusionAlgorithms, diffusionFunctions } from './registry'
