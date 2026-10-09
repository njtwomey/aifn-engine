/**
 * The registry of `aifn-compute/probability/privacy`: differentially private mechanisms, composition and RDP/zCDP
 * accounting, and DP-SGD's clipped, noised gradient aggregation, each with its summary, notes and citations.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as accounting from './accounting'
import * as clipping from './clipping'
import * as mechanisms from './mechanisms'

const fn = definer<FunctionInfo>('function', 'probability/privacy')
const DP = ['differential-privacy']
const SGD = ['differentially-private-stochastic-gradient-descent', ...DP]

fn(
  {
    key: 'laplaceMechanism',
    name: 'Laplace mechanism',
    summary: 'Laplace noise of scale Δ₁/ε on each coordinate: ε-DP.',
    role: 'simulation',
    random: true,
    notes: DP,
    cite: ['dwork2006'],
  },
  mechanisms.laplaceMechanism,
)
fn(
  {
    key: 'gaussianMechanism',
    name: 'Gaussian mechanism',
    summary: 'Gaussian noise of standard deviation σ on each coordinate.',
    role: 'simulation',
    random: true,
    notes: DP,
    cite: ['dwork2014'],
  },
  mechanisms.gaussianMechanism,
)
fn(
  {
    key: 'classicGaussianSigma',
    name: 'Classic Gaussian calibration',
    summary: 'σ = Δ₂√(2 ln(1.25/δ))/ε, (ε, δ)-DP for ε < 1.',
    role: 'property',
    notes: DP,
    cite: ['dwork2014'],
  },
  mechanisms.classicGaussianSigma,
)
fn(
  {
    key: 'gaussianDelta',
    name: 'Gaussian privacy profile',
    summary: 'The exact δ(ε) of the Gaussian mechanism: Φ(Δ/2σ − εσ/Δ) − e^ε Φ(−Δ/2σ − εσ/Δ).',
    role: 'property',
    notes: DP,
  },
  mechanisms.gaussianDelta,
)
fn(
  {
    key: 'gaussianEpsilon',
    name: 'Gaussian mechanism ε at δ',
    summary: 'The smallest ε for which Gaussian noise σ is (ε, δ)-DP: the inverse of the privacy profile.',
    role: 'property',
    notes: DP,
  },
  mechanisms.gaussianEpsilon,
)
fn(
  {
    key: 'analyticGaussianSigma',
    name: 'Analytic Gaussian calibration',
    summary: 'The smallest σ whose exact privacy profile meets (ε, δ), for any ε > 0.',
    role: 'solver',
    notes: DP,
  },
  mechanisms.analyticGaussianSigma,
)
fn(
  {
    key: 'exponentialMechanismProbabilities',
    name: 'Exponential mechanism probabilities',
    summary: 'Selection probabilities ∝ exp(ε u/(2Δu)).',
    role: 'property',
    notes: DP,
    cite: ['dwork2014'],
  },
  mechanisms.exponentialMechanismProbabilities,
)
fn(
  {
    key: 'exponentialMechanism',
    name: 'Exponential mechanism',
    summary: 'A candidate drawn with probability ∝ exp(ε u/(2Δu)): ε-DP selection.',
    role: 'simulation',
    random: true,
    notes: DP,
    cite: ['dwork2014'],
  },
  mechanisms.exponentialMechanism,
)
fn(
  {
    key: 'randomisedResponse',
    name: 'Randomised response',
    summary: 'Each bit reported truthfully with probability e^ε/(1 + e^ε), else flipped.',
    role: 'simulation',
    random: true,
    notes: DP,
    cite: ['warner1965'],
  },
  mechanisms.randomisedResponse,
)
fn(
  {
    key: 'randomisedResponseKeep',
    name: 'Randomised response truth probability',
    role: 'property',
    notes: DP,
    cite: ['warner1965'],
  },
  mechanisms.randomisedResponseKeep,
)
fn(
  {
    key: 'randomisedResponseEstimate',
    name: 'Randomised response estimate',
    summary: 'The unbiased share of ones recovered from randomised reports.',
    role: 'estimator',
    notes: DP,
    cite: ['warner1965'],
  },
  mechanisms.randomisedResponseEstimate,
)
fn(
  {
    key: 'sequentialComposition',
    name: 'Sequential composition',
    summary: 'Privacy losses add: (Σεᵢ, Σδᵢ).',
    role: 'property',
    notes: DP,
    cite: ['dwork2014'],
  },
  accounting.sequentialComposition,
)
fn(
  {
    key: 'advancedComposition',
    name: 'Advanced composition',
    summary: 'k uses of (ε, δ)-DP: (ε√(2k ln(1/δ′)) + kε(e^ε − 1), kδ + δ′).',
    role: 'property',
    notes: DP,
    cite: ['dwork2010', 'kairouz2015'],
  },
  accounting.advancedComposition,
)
fn(
  {
    key: 'rdpSubsampledGaussian',
    name: 'RDP of the subsampled Gaussian',
    summary: 'Rényi DP at each order of k steps of the Poisson-subsampled Gaussian mechanism.',
    role: 'property',
    notes: SGD,
    cite: ['mironov2017', 'abadi2016dp'],
  },
  accounting.rdpSubsampledGaussian,
)
fn(
  {
    key: 'rdpToEpsilon',
    name: 'RDP to (ε, δ)',
    summary: 'The tightest ε at δ over the RDP orders.',
    role: 'property',
    notes: SGD,
    cite: ['mironov2017'],
  },
  accounting.rdpToEpsilon,
)
fn(
  {
    key: 'dpSgdEpsilon',
    name: 'DP-SGD privacy spent',
    summary: 'ε at δ after k DP-SGD steps with sampling rate q and noise multiplier σ, by RDP accounting.',
    role: 'property',
    notes: SGD,
    cite: ['abadi2016dp', 'mironov2017'],
  },
  accounting.dpSgdEpsilon,
)
fn(
  { key: 'gaussianZcdp', name: 'Gaussian zCDP', summary: 'ρ = Δ²/(2σ²).', role: 'property', notes: DP },
  accounting.gaussianZcdp,
)
fn(
  {
    key: 'zcdpToEpsilon',
    name: 'zCDP to (ε, δ)',
    summary: 'ρ-zCDP implies (ρ + 2√(ρ ln(1/δ)), δ)-DP.',
    role: 'property',
    notes: DP,
  },
  accounting.zcdpToEpsilon,
)
fn(
  {
    key: 'clipAndNoise',
    name: 'Clip and noise (DP-SGD aggregation)',
    summary: 'Per-example gradients clipped to norm C, summed, noised with N(0, σ²C²) and averaged.',
    role: 'transform',
    random: true,
    notes: SGD,
    cite: ['abadi2016dp'],
  },
  clipping.clipAndNoise,
)

/** The functions of the module, keyed by name. */
export const privacyFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', mechanisms, accounting, clipping) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
