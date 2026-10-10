/**
 * `aifn-compute/signal`: signal processing, as scipy.signal, PyWavelets, PyEMD, librosa and stumpy.
 *
 * The family's shared layer holds the objects every module passes around: `signal` (samples with a sample rate, start
 * time and unit; `isSignal`, `sampleTimes`), `spectrum` and `timeFrequency`, and the readers of a spectrum's values,
 * `magnitude`, `phase`, `unwrapPhase` and `spectrumDecibels`. The family also re-exports the most used functions of
 * its children: `getWindow`, `firwin`, `butter`, `lfilter`, `sosfilt`, `filtfilt`, `freqz`, `periodogram`, `welch`,
 * `hilbert`, `dwt`, `idwt`, `yuleWalker`, `burg` and `emd`.
 *
 * - `windows`: window functions, for spectral estimation and FIR design, with their main-lobe widths and side-lobe
 *   levels.
 * - `filters`: digital filters: FIR, IIR and equiripple design, filtering (forwards, or zero-phase with `filtfilt`)
 *   and frequency responses.
 * - `spectral`: nonparametric spectral estimation (periodogram, Welch, multitaper, Lomb–Scargle for uneven sampling,
 *   cross-spectra and coherence) and the short-time Fourier transform and its inverse.
 * - `time-frequency`: the analytic signal, the constant-Q transform, Wigner–Ville distributions and sharpened
 *   spectrograms.
 * - `wavelets`: discrete and continuous wavelet transforms, and denoising by wavelet shrinkage.
 * - `statistical`: autoregressive fits, parametric and line spectra (ARMA, AR, MUSIC, ESPRIT) and adaptive filters.
 * - `cepstrum`: the cepstrum, cepstral smoothing, and pitch by the cepstrum and by YIN.
 * - `multirate`: sample-rate change and filter banks.
 * - `audio`: audio features on the mel scale: the mel filter bank and MFCCs.
 * - `decompositions`: adaptive decompositions of a signal into modes: empirical mode decomposition and its ensemble
 *   variants, and variational mode decomposition.
 * - `similarity`: similarity search in time series: distance and matrix profiles, dynamic time warping, SAX.
 * - `image`: classical image processing on greyscale images: filters, edges, corners, blobs, Hough transforms,
 *   morphology and pyramids.
 * - `sparse`: sparse representations over a dictionary: greedy and convex pursuits, sparse coding, and dictionary
 *   learning (also convolutional).
 *
 * Functions take a `SignalInput`, a `Signal` or bare samples (which have a sample rate of 1 unless an `fs` option
 * says otherwise), and return `Signal`, `Spectrum` and `TimeFrequency` objects, so axes and units carry through a
 * chain; frequencies are in Hz.
 */

export {
  isSignal,
  magnitude,
  phase,
  spectrumDecibels,
  unwrapPhase,
  sampleTimes,
  signal,
  spectrum,
  timeFrequency,
  type Signal,
  type SignalInput,
  type SignalOptions,
  type Spectrum,
  type TimeFrequency,
} from './signal'
export { getWindow } from './windows'
export { firwin, butter, lfilter, sosfilt, filtfilt, freqz } from './filters'
export { periodogram, welch } from './spectral'
export { hilbert } from './time-frequency'
export { dwt, idwt } from './wavelets'
export { yuleWalker, burg } from './statistical'
export { emd } from './decompositions'
export { signalFunctions } from './signal'
