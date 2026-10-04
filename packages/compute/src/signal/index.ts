/**
 * `aifn-compute/signal`: signal processing, as scipy.signal, pywt and PyEMD: windows, filters, spectral estimation,
 * time–frequency analysis, wavelets, statistical signal processing and decompositions. The shared layer holds the
 * signal objects: `signal` (samples with a sample rate), `spectrum`, `timeFrequency`, and the spectrum readers
 * `magnitude`, `phase` and `spectrumDecibels`. Children: windows, filters, spectral, time-frequency, wavelets,
 * statistical, cepstrum, multirate, audio (mel scale, mel filter bank, MFCC), decompositions, similarity, image.
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
