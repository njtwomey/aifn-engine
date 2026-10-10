/**
 * `aifn-compute/signal/statistical`: statistical signal processing: autoregressive models, parametric spectra and
 * adaptive filters.
 *
 * - Autoregressive (AR) fits, each returning the coefficients, innovation variance and reflection coefficients:
 *   `yuleWalker` (the Yule–Walker equations by `levinsonDurbin` of `aifn-compute/numerics/linalg`; always
 *   stationary), `burg` (Burg's method; stationary, and sharper on short series) and `leastSquaresAr` (the covariance
 *   or modified covariance method; sharpest, but not guaranteed stationary).
 * - Parametric spectra: `armaSpectrum`, the exact one-sided PSD of an ARMA model, and `arPsd`, the PSD of an AR model
 *   fitted by any of the three methods.
 * - Line spectra, for a few sinusoids in noise: `music` (a pseudospectrum whose peaks are the frequencies), `esprit`
 *   (the frequencies without a grid search) and `sinusoidFit` (amplitudes and phases at known frequencies).
 * - Adaptive FIR filters as step-through algorithms, one sample per step, run with `run`: `lms`, `nlms` (LMS
 *   insensitive to the input's scale) and `rls` (exact exponentially weighted least squares, fastest to converge).
 *
 * Series are single-channel signals or their samples; frequencies are in Hz with `fs` (default 1, cycles per sample)
 * and spectra are one-sided, so they overlay the estimates of `aifn-compute/signal/spectral`.
 */

export { burg, yuleWalker, type AutoregressiveFit } from './autoregression'
export {
  armaSpectrum,
  arPsd,
  esprit,
  leastSquaresAr,
  music,
  sinusoidFit,
  type ArmaModel,
  type ArPsd,
  type LineSpectrum,
  type SinusoidFit,
  type SubspaceOptions,
} from './parametric'
export {
  lms,
  nlms,
  rls,
  type AdaptiveFilterOptions,
  type AdaptiveFilterStart,
  type AdaptiveFilterState,
  type RlsState,
} from './adaptive'
export { statisticalAlgorithms, statisticalFunctions } from './registry'
