/**
 * `aifn-compute/signal/cepstrum`: the cepstrum and pitch. The real cepstrum (`realCepstrum`), the complex cepstrum and its
 * inverse (`complexCepstrum`, `inverseComplexCepstrum`), cepstral pitch (`cepstralPitch`), and the YIN estimator for
 * one frame (`yinPitch`, with `yinDifference`) and as a frame-by-frame tracker (`yin`).
 */

export {
  cepstralEnvelope,
  cepstralPitch,
  complexCepstrum,
  inverseComplexCepstrum,
  realCepstrum,
  type CepstralEnvelope,
  type CepstralPitch,
  type CepstralPitchOptions,
  type ComplexCepstrum,
} from './cepstrum'
export {
  yin,
  yinDifference,
  yinPitch,
  type PitchTrack,
  type YinDifference,
  type YinEstimate,
  type YinOptions,
} from './yin'
export { cepstrumFunctions } from './registry'
