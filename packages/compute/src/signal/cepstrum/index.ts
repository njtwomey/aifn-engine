/**
 * `aifn-compute/signal/cepstrum`: the cepstrum, cepstral smoothing, and pitch by the cepstrum and by YIN.
 *
 * - Cepstra: `realCepstrum` ($\log \abs{X}$ only, even in quefrency), and `complexCepstrum` (with the unwrapped phase,
 *   linear phase removed) with its inverse `inverseComplexCepstrum`.
 * - Spectral envelope: `cepstralEnvelope`, the log spectrum smoothed by keeping only the low quefrencies (liftering).
 * - Pitch of one frame: `cepstralPitch` (the largest cepstral peak in a quefrency range) or `yinPitch` (the first dip
 *   of YIN's normalised difference $d'$ below a threshold, with `yinDifference` for $d$ and $d'$ themselves).
 * - Pitch over time: `yin`, YIN frame by frame, NaN where a frame is unvoiced.
 * - `cepstrumFunctions` lists the functions with the notes they serve.
 *
 * Quefrencies and lags are in samples inside the transforms; the pitch functions report seconds and Hz, with the
 * sample rate from the signal or an `fs` option.
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
