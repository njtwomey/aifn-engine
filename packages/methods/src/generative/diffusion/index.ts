/**
 * `aifn-methods/generative/diffusion`: diffusion models on toy data, from noise schedules to samplers.
 *
 * - Schedules: `linearSchedule` (Ho et al.), `cosineSchedule` (Nichol & Dhariwal), `scheduleFromBetas`, each with
 *   $\beta_t$, $\alpha_t$, $\bar\alpha_t$ and $\mathrm{SNR}(t)$ per step; `alphaBarAt`, `betaAt`. Continuous forward
 *   SDEs with their marginals: `vpSde`, `subVpSde`, `veSde`. One time axis: `stepTime` ($\tau = t/T$) and `timeStep`,
 *   `scheduleSde` (the VP SDE through a discrete schedule, exact at every step) and `sdeSchedule` (a schedule read off
 *   an SDE); every sampler reports $\tau$.
 * - The forward process: `forwardNoise` (a draw from $q(\xvec_t \mid \xvec_0)$), `forwardPosterior`
 *   ($q(\xvec_{t-1} \mid \xvec_t, \xvec_0)$), `forwardProcess` (the Markov chain, traceable).
 * - Noise predictors $\hat\epsilonvec(\xvec, \bar\alpha)$, the common interface of every model: `predictNoise` and
 *   `scoreFromPredictor` for any Gaussian marginal, `predictClean` (Tweedie).
 * - Gaussian-mixture data with exact noised marginals: `gaussianMixtureData`, `sampleMixture`, `mixtureLogDensity`,
 *   `mixtureScore`, `mixtureNoisePredictor` (a perfect predictor, so demos need no training), `mixtureMoments`.
 * - A learned predictor: `denoiser` (an MLP from `aifn-compute/nn`), `noiseLevelFeatures`, `denoiserInput`,
 *   `denoiserTraining` (traceable), `networkNoisePredictor`.
 * - Samplers, each a traceable `Algorithm` over all particles: `ddpmSampler`, `ddimSampler` (with `ddimTimesteps`),
 *   `reverseSdeSampler` (Euler–Maruyama), `probabilityFlowSampler` (RK4 or Euler).
 * - The registry: `diffusionAlgorithms` and `diffusionFunctions`.
 *
 * Conventions: points are $[n, d]$ tensors; discrete steps run $t = 1, \dots, T$ with $\bar\alpha_0 = 1$; continuous
 * time runs over $[0, 1]$ and sampling stops at $t = \varepsilon$ (default $10^{-3}$). A step outside a schedule throws
 * `DomainError`; points of the wrong shape throw `ShapeError`.
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
