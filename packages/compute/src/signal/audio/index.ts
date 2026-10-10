/**
 * `aifn-compute/signal/audio`: audio features on the mel scale, as librosa.
 *
 * - The mel scale: `hzToMel` and its inverse `melToHz`, on HTK's scale $2595 \log_{10}(1 + f / 700)$ (the default) or
 *   Slaney's, linear below 1 kHz and logarithmic above.
 * - `melFilterbank`: triangular filters equally spaced in mel, as weights on the one-sided bins of an FFT, normalised
 *   to unit area (Slaney) or unit peak.
 * - `mfcc`: mel-frequency cepstral coefficients, the orthonormal DCT-II (of `aifn-compute/foundation/fourier`) of log
 *   mel energies of the short-time Fourier transform (of `aifn-compute/signal/spectral`), returned with the power
 *   spectrogram and log mel energies they came from.
 * - `audioFunctions`: the module's functions with the notes and citations that define them.
 *
 * Frequencies are in hertz and times in seconds. Nothing here is differentiable: the functions return plain tensors.
 */

export {
  hzToMel,
  melFilterbank,
  melToHz,
  mfcc,
  type MelFilterbank,
  type MelScale,
  type Mfcc,
  type MfccOptions,
} from './audio'
export { audioFunctions } from './registry'
