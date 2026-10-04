/**
 * The functions of `aifn-compute/signal/spectral`, registered with the notes they serve.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn-compute/foundation/registry'
import * as estimation from './estimation'
import * as inverse from './inverse'
import * as spectral from './spectral'
import * as uneven from './uneven'

const fn = definer<FunctionInfo>('function', 'signal/spectral')
const STFT = ['short-time-fourier-transform', 'time-frequency-uncertainty']

fn(
  {
    key: 'periodogram',
    name: 'Periodogram',
    summary: 'The squared magnitude of the windowed DFT, scaled to a power spectral density.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['periodogram', 'spectral-leakage-and-windows', 'autocorrelation-and-wiener-khinchin'],
  },
  spectral.periodogram,
)
fn(
  {
    key: 'welch',
    name: "Welch's method",
    summary: 'The average of windowed periodograms of overlapping segments.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['welch-method', 'periodogram'],
    cite: ['welch1967'],
  },
  spectral.welch,
)
fn(
  {
    key: 'spectrogram',
    name: 'Spectrogram',
    role: 'estimator',
    returns: 'time-frequency',
    notes: [...STFT, 'mel-spectrogram'],
  },
  spectral.spectrogram,
)
fn(
  {
    key: 'stft',
    name: 'Short-time Fourier transform',
    role: 'transform',
    returns: 'time-frequency',
    notes: STFT,
    cite: ['allen1977', 'gabor1946'],
  },
  spectral.stft,
)
fn(
  {
    key: 'istft',
    name: 'Inverse STFT (overlap-add)',
    role: 'transform',
    returns: 'signal',
    notes: ['short-time-fourier-transform', 'overlap-add-and-overlap-save'],
    cite: ['griffin1984'],
  },
  inverse.istft,
)
fn(
  { key: 'checkCola', name: 'Constant overlap-add check', role: 'property', notes: ['overlap-add-and-overlap-save'] },
  inverse.checkCola,
)
fn(
  { key: 'checkNola', name: 'Nonzero overlap-add check', role: 'property', notes: ['overlap-add-and-overlap-save'] },
  inverse.checkNola,
)
fn(
  {
    key: 'dpss',
    name: 'Discrete prolate spheroidal sequences',
    summary: 'The Slepian tapers: the sequences most concentrated in a band.',
    role: 'construction',
    notes: ['multitaper-spectral-estimation'],
    cite: ['slepian1978'],
  },
  spectral.dpss,
)
fn(
  {
    key: 'multitaper',
    name: 'Multitaper spectral estimate',
    summary: 'The average of periodograms under orthogonal Slepian tapers, optionally with adaptive weights.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['multitaper-spectral-estimation'],
    cite: ['thomson1982'],
  },
  spectral.multitaper,
)
fn(
  {
    key: 'welchDof',
    name: 'Equivalent degrees of freedom of a Welch average',
    summary: '2K for K independent segments, less when overlapping segments are correlated through the window.',
    role: 'property',
    notes: ['welch-method', 'confidence-intervals'],
    cite: ['welch1967'],
  },
  spectral.welchDof,
)
fn(
  {
    key: 'bartlett',
    name: "Bartlett's method",
    summary: 'The mean of rectangular-window periodograms of non-overlapping segments.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['welch-method', 'periodogram'],
    cite: ['bartlett1946'],
  },
  estimation.bartlett,
)
fn(
  {
    key: 'blackmanTukey',
    name: 'Blackman–Tukey estimate',
    summary: 'The Fourier transform of the sample autocovariance under a lag window.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['autocorrelation-and-wiener-khinchin', 'periodogram', 'spectral-leakage-and-windows'],
    cite: ['stoica2005'],
  },
  estimation.blackmanTukey,
)
fn(
  {
    key: 'csd',
    name: 'Cross-spectral density',
    summary:
      'The Welch average of conj(X)·Y over segments: magnitude and phase of the linear coupling at each frequency.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['coherence-and-cross-spectra'],
    cite: ['carter1973'],
  },
  estimation.csd,
)
fn(
  {
    key: 'coherence',
    name: 'Magnitude-squared coherence',
    summary: '|P_xy|² / (P_xx P_yy): the fraction of power at each frequency that is linearly coupled.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['coherence-and-cross-spectra'],
    cite: ['carter1973'],
  },
  estimation.coherence,
)
fn(
  {
    key: 'coherenceThreshold',
    name: 'Null threshold of coherence',
    summary: 'The coherence two independent signals exceed with a given probability, from K segments.',
    role: 'property',
    notes: ['coherence-and-cross-spectra'],
    cite: ['carter1973'],
  },
  estimation.coherenceThreshold,
)
fn(
  {
    key: 'spectralConfidence',
    name: 'χ² confidence interval of a spectral estimate',
    summary: 'νŜ/S ~ χ²_ν: a band of constant width in dB from the equivalent degrees of freedom.',
    role: 'inference',
    notes: ['periodogram', 'welch-method', 'confidence-intervals'],
    cite: ['welch1967'],
  },
  estimation.spectralConfidence,
)
fn(
  {
    key: 'chiSquareQuantile',
    name: 'χ² quantile',
    role: 'property',
    notes: ['confidence-intervals'],
  },
  estimation.chiSquareQuantile,
)
fn(
  {
    key: 'logSpectralError',
    name: 'Log-spectral error',
    summary: 'Bias, spread and root-mean-square of 10 log₁₀(Ŝ/S) over frequency.',
    role: 'property',
    notes: ['periodogram', 'parametric-spectral-estimation'],
  },
  estimation.logSpectralError,
)
fn(
  {
    key: 'replicateSpectralError',
    name: 'Bias and variance of a spectral estimator',
    summary: 'Squared bias and variance in dB over replicate realisations, averaged over frequency.',
    role: 'property',
    notes: ['welch-method', 'periodogram', 'multitaper-spectral-estimation'],
  },
  estimation.replicateSpectralError,
)
fn(
  {
    key: 'peakDip',
    name: 'Two-peak resolution',
    summary: 'The dip between two peaks of an estimate, in dB: resolved when at least 3 dB.',
    role: 'property',
    notes: ['spectral-leakage-and-windows', 'subspace-frequency-estimation'],
  },
  estimation.peakDip,
)
fn(
  {
    key: 'crossSpectralDelay',
    name: 'Delay from the cross-spectral phase',
    summary: 'The coherence-weighted slope of the unwrapped cross-spectral phase, −2πfτ.',
    role: 'estimator',
    notes: ['coherence-and-cross-spectra', 'linear-phase-and-group-delay'],
  },
  estimation.crossSpectralDelay,
)
fn(
  {
    key: 'gridSamples',
    name: 'Uneven samples on a regular grid',
    summary: 'Linear interpolation or zero filling onto a regular grid, for DFT-based estimates.',
    role: 'transform',
    returns: 'signal',
    notes: ['lomb-scargle-periodogram'],
  },
  uneven.gridSamples,
)
fn(
  {
    key: 'lombScargle',
    name: 'Lomb–Scargle periodogram',
    summary:
      'A sinusoid (plus a constant) fitted by least squares at each frequency: a periodogram for uneven sampling.',
    role: 'estimator',
    returns: 'spectrum',
    notes: ['lomb-scargle-periodogram'],
    cite: ['lomb1976', 'scargle1982', 'zechmeister2009', 'vanderplas2018'],
  },
  uneven.lombScargle,
)
fn(
  {
    key: 'lombScargleFrequencies',
    name: 'Lomb–Scargle frequency grid',
    role: 'construction',
    notes: ['lomb-scargle-periodogram'],
    cite: ['vanderplas2018'],
  },
  uneven.lombScargleFrequencies,
)
fn(
  {
    key: 'falseAlarmProbability',
    name: 'False-alarm probability (Baluev)',
    summary: 'An upper bound on the probability that noise alone gives a Lomb–Scargle peak this high.',
    role: 'inference',
    notes: ['lomb-scargle-periodogram'],
    cite: ['vanderplas2018'],
  },
  uneven.falseAlarmProbability,
)
fn(
  {
    key: 'falseAlarmLevel',
    name: 'False-alarm level',
    summary: 'The Lomb–Scargle power whose false-alarm probability is a given level.',
    role: 'inference',
    notes: ['lomb-scargle-periodogram'],
    cite: ['vanderplas2018'],
  },
  uneven.falseAlarmLevel,
)
fn(
  {
    key: 'spectralWindow',
    name: 'Spectral window of a sampling pattern',
    summary: 'The periodogram of a constant observed at the sample times; its peaks are the aliases.',
    role: 'property',
    returns: 'spectrum',
    notes: ['lomb-scargle-periodogram', 'aliasing'],
    cite: ['vanderplas2018'],
  },
  uneven.spectralWindow,
)

/** The functions of the module, keyed by name. */
export const spectralFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', spectral, inverse, estimation, uneven) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
