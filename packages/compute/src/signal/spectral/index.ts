/**
 * `aifn-compute/signal/spectral`: nonparametric spectral estimation and the short-time Fourier transform, with
 * scipy.signal's conventions.
 *
 * - Power spectral densities, returned as `Spectrum`s: `periodogram`, `welch` (with `welchDof`), `bartlett`,
 *   `blackmanTukey` (lag window) and `multitaper` (Thomson, with `dpss` tapers and optional adaptive weights).
 * - Uncertainty: `spectralConfidence`, $\chi^2$ intervals from each estimate's `dof`, with `chiSquareQuantile`.
 * - Two signals: `csd` (cross-spectral density), `coherence` with its null level `coherenceThreshold`, and
 *   `crossSpectralDelay`, a delay from the cross-spectrum's phase.
 * - Judging an estimate: `logSpectralError` against a reference, `replicateSpectralError` (bias and variance over
 *   realisations) and `peakDip` (are two close peaks resolved?).
 * - Uneven sampling: `lombScargle` (classic, floating-mean, generalised) on `lombScargleFrequencies`, Baluev's
 *   `falseAlarmProbability` and `falseAlarmLevel`, the `spectralWindow` of a sampling pattern, and `gridSamples` to
 *   regularise samples.
 * - Short-time transforms, returned as `TimeFrequency` rasters: `stft` and `spectrogram`, the inverse `istft`, and the
 *   window tests `checkCola` and `checkNola`.
 * - `spectralFunctions` lists the functions with the notes they serve.
 *
 * Densities are one-sided by default (doubled off DC and Nyquist) in power per Hz, so they integrate over
 * $[0, f_s/2]$ to the variance; $f_s$ comes from a `Signal` or an `fs` option, and is 1 for bare samples. Parametric
 * estimates (AR, MUSIC, ESPRIT) are in `aifn-compute/signal/statistical`.
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
