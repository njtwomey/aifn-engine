/**
 * `aifn-compute/signal/spectral`: nonparametric spectral estimation, returning `Spectrum`s: the periodogram, Welch and Bartlett
 * averages, the Blackman–Tukey lag-window estimate, Thomson's multitaper with DPSS tapers and adaptive weights, χ²
 * confidence intervals (`spectralConfidence`, with each estimate's `dof`), the cross-spectral density and coherence,
 * and the log-spectral error against a reference; for uneven sampling, the Lomb–Scargle periodogram (classic,
 * floating-mean, generalised) with Baluev's false-alarm probability and the spectral window. Also the short-time
 * Fourier transform and spectrogram, returning `TimeFrequency` rasters, and the inverse STFT (`istft`) with the
 * COLA/NOLA window tests. Parametric estimates (AR, MUSIC, ESPRIT) are in `aifn-compute/signal/statistical`.
 */

export {
  dpss,
  multitaper,
  periodogram,
  spectrogram,
  stft,
  welch,
  welchDof,
  type Detrend,
  type Dpss,
  type MultitaperOptions,
  type SegmentOptions,
} from './spectral'
export {
  bartlett,
  blackmanTukey,
  chiSquareQuantile,
  coherence,
  coherenceThreshold,
  crossSpectralDelay,
  csd,
  logSpectralError,
  peakDip,
  replicateSpectralError,
  spectralConfidence,
  type BlackmanTukeyOptions,
  type SpectralErrorOptions,
  type SpectralInterval,
} from './estimation'
export {
  falseAlarmLevel,
  falseAlarmProbability,
  gridSamples,
  lombScargle,
  lombScargleFrequencies,
  spectralWindow,
  type LombScargle,
  type LombScargleOptions,
} from './uneven'
export { checkCola, checkNola, istft, type IstftOptions } from './inverse'
export { spectralFunctions } from './registry'
