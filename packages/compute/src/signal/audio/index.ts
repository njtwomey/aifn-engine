/**
 * `aifn-compute/signal/audio`: audio features: the mel scale (`hzToMel`, `melToHz`; HTK and Slaney), the triangular mel filter
 * bank (`melFilterbank`) and mel-frequency cepstral coefficients (`mfcc`) from the short-time Fourier transform of
 * `aifn-compute/signal/spectral` and the DCT of `aifn-compute/foundation/fourier`.
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
