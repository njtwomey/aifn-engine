/**
 * The registry of `aifn-methods/generative/diffusion`: the forward chain, the four samplers and denoiser training as
 * traceable algorithms; schedules, forward SDEs, predictors and the Gaussian-mixture toy (exact score) as functions.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as denoiser from './denoiser'
import * as forward from './forward'
import * as mixture from './mixture'
import * as predictor from './predictor'
import * as samplers from './samplers'
import * as schedules from './schedules'

/** A table of the module's registry entries, keyed by name. */
type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'generative/diffusion')
const fn = definer<FunctionInfo>('function', 'generative/diffusion')

const DDPM = ['denoising-diffusion-probabilistic-models']
const SDE = ['diffusion-models-as-stochastic-differential-equations', 'score-based-generative-models']
const sampler = { iterate: 'x', flags: [] } as const

// ── Algorithms ───────────────────────────────────────────────────────────────────────────────────────────────────────

algorithm(
  {
    key: 'forwardProcess',
    name: 'Forward noising chain',
    summary: 'x_t = √αₜ x_{t−1} + √βₜ ε, one step per noise level, until x_T is nearly N(0, I).',
    problem: 'sequence',
    state: { iterate: 'x', objective: 'alphaBar', flags: [] },
    random: true,
    notes: DDPM,
    cite: ['ho2020', 'sohldickstein2015'],
  },
  forward.forwardProcess,
)
algorithm(
  {
    key: 'ddpmSampler',
    name: 'DDPM ancestral sampling',
    problem: 'sde',
    state: sampler,
    random: true,
    notes: DDPM,
    cite: ['ho2020'],
  },
  samplers.ddpmSampler,
)
algorithm(
  {
    key: 'ddimSampler',
    name: 'DDIM sampling',
    summary: 'Deterministic (η = 0) or partly stochastic sampling on a subsequence of the noise levels.',
    problem: 'sde',
    state: sampler,
    random: true,
    notes: ['denoising-diffusion-implicit-models', ...DDPM],
    cite: ['song2021'],
  },
  samplers.ddimSampler,
)
algorithm(
  {
    key: 'reverseSdeSampler',
    name: 'Reverse-time SDE sampler',
    summary: 'Euler–Maruyama on dx = [f − g²∇log p_t] dt + g dw̄, backwards in time.',
    problem: 'sde',
    state: sampler,
    random: true,
    notes: ['reverse-time-stochastic-differential-equations', ...SDE],
    cite: ['song2021b', 'anderson1982'],
  },
  samplers.reverseSdeSampler,
)
algorithm(
  {
    key: 'probabilityFlowSampler',
    name: 'Probability-flow ODE sampler',
    summary: 'The deterministic ODE dx = [f − ½g²∇log p_t] dt with the same marginals as the forward SDE.',
    problem: 'ode',
    state: sampler,
    random: true,
    notes: ['probability-flow-ode', ...SDE],
    cite: ['song2021b'],
  },
  samplers.probabilityFlowSampler,
)
algorithm(
  {
    key: 'denoiserTraining',
    name: 'Denoiser training',
    summary: 'Minibatch MSE between predicted and true noise at a random level per point (the simplified DDPM loss).',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', grad: 'grads', flags: ['diverged'] },
    random: true,
    notes: [...DDPM, 'score-based-generative-models'],
    cite: ['ho2020', 'vincent2011'],
  },
  denoiser.denoiserTraining,
)

// ── Schedules and forward SDEs ───────────────────────────────────────────────────────────────────────────────────────

fn(
  { key: 'scheduleFromBetas', name: 'Noise schedule from βs', role: 'construction', notes: DDPM },
  schedules.scheduleFromBetas,
)
fn(
  { key: 'linearSchedule', name: 'Linear β schedule', role: 'construction', notes: DDPM, cite: ['ho2020'] },
  schedules.linearSchedule,
)
fn(
  { key: 'cosineSchedule', name: 'Cosine ᾱ schedule', role: 'construction', notes: DDPM, cite: ['nichol2021'] },
  schedules.cosineSchedule,
)
fn({ key: 'alphaBarAt', name: 'ᾱ at a level', role: 'property', notes: DDPM }, schedules.alphaBarAt)
fn({ key: 'betaAt', name: 'β at a level', role: 'property', notes: DDPM }, schedules.betaAt)
fn(
  { key: 'vpSde', name: 'Variance-preserving SDE', role: 'construction', notes: SDE, cite: ['song2021b'] },
  schedules.vpSde,
)
fn({ key: 'subVpSde', name: 'Sub-VP SDE', role: 'construction', notes: SDE, cite: ['song2021b'] }, schedules.subVpSde)
fn(
  { key: 'veSde', name: 'Variance-exploding SDE', role: 'construction', notes: SDE, cite: ['song2021b'] },
  schedules.veSde,
)
fn({ key: 'stepTime', name: 'Discrete level to continuous time', role: 'transform', notes: SDE }, schedules.stepTime)
fn({ key: 'timeStep', name: 'Continuous time to discrete level', role: 'transform', notes: SDE }, schedules.timeStep)
fn({ key: 'scheduleSde', name: 'The SDE limit of a schedule', role: 'construction', notes: SDE }, schedules.scheduleSde)
fn({ key: 'sdeSchedule', name: 'A schedule from an SDE', role: 'construction', notes: SDE }, schedules.sdeSchedule)

// ── Forward process, predictors and the toy ──────────────────────────────────────────────────────────────────────────

fn(
  {
    key: 'forwardNoise',
    name: 'Noise to a level in one draw',
    tex: 'x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon',
    role: 'simulation',
    random: true,
    notes: DDPM,
  },
  forward.forwardNoise,
)
fn(
  {
    key: 'forwardPosterior',
    name: 'Forward posterior q(x_{t−1} | x_t, x₀)',
    role: 'inference',
    notes: DDPM,
    cite: ['ho2020'],
  },
  forward.forwardPosterior,
)
fn({ key: 'predictNoise', name: 'Predicted noise', role: 'inference', notes: DDPM }, predictor.predictNoise)
fn(
  {
    key: 'scoreFromPredictor',
    name: 'Score from a noise predictor',
    tex: '\\nabla \\log p_t = -\\hat\\varepsilon / \\sigma_t',
    role: 'inference',
    notes: ['tweedies-formula', ...SDE],
  },
  predictor.scoreFromPredictor,
)
fn(
  {
    key: 'predictClean',
    name: 'Predicted clean point (Tweedie)',
    role: 'inference',
    notes: ['tweedies-formula', ...DDPM],
    cite: ['efron2011'],
  },
  predictor.predictClean,
)
fn(
  { key: 'gaussianMixtureData', name: 'Gaussian-mixture toy data', role: 'construction', notes: DDPM },
  mixture.gaussianMixtureData,
)
fn({ key: 'sampleMixture', name: 'Sample the mixture', role: 'simulation', random: true }, mixture.sampleMixture)
fn(
  {
    key: 'mixtureLogDensity',
    name: 'Noised mixture log-density',
    role: 'property',
    notes: ['score-based-generative-models'],
  },
  mixture.mixtureLogDensity,
)
fn(
  {
    key: 'mixtureScore',
    name: 'Exact score of the noised mixture',
    role: 'property',
    notes: ['score-based-generative-models', 'tweedies-formula'],
  },
  mixture.mixtureScore,
)
fn(
  { key: 'mixtureNoisePredictor', name: 'Exact noise predictor of the mixture', role: 'construction', notes: DDPM },
  mixture.mixtureNoisePredictor,
)
fn({ key: 'mixtureMoments', name: 'Mixture mean and covariance', role: 'property' }, mixture.mixtureMoments)
fn({ key: 'denoiser', name: 'Denoiser network', role: 'construction', notes: DDPM }, denoiser.denoiser)
fn(
  { key: 'noiseLevelFeatures', name: 'Noise-level features', role: 'transform', notes: DDPM },
  denoiser.noiseLevelFeatures,
)
fn({ key: 'denoiserInput', name: 'Denoiser input', role: 'transform', notes: DDPM }, denoiser.denoiserInput)
fn(
  { key: 'networkNoisePredictor', name: 'Noise predictor of a trained network', role: 'construction', notes: DDPM },
  denoiser.networkNoisePredictor,
)
fn(
  {
    key: 'ddimTimesteps',
    name: 'DDIM level subsequence',
    role: 'construction',
    notes: ['denoising-diffusion-implicit-models'],
  },
  samplers.ddimTimesteps,
)

/** The algorithms of the module, keyed by factory name. */
export const diffusionAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  forward,
  samplers,
  denoiser,
) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const diffusionFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  schedules,
  forward,
  predictor,
  mixture,
  denoiser,
  samplers,
) as Table<FunctionInfo>
