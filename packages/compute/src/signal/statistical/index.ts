/**
 * `aifn-compute/signal/statistical`: statistical signal processing. Autoregressive (AR) estimation by the Yule–Walker
 * equations (`yuleWalker`, through `aifn-compute/numerics/linalg`'s `levinsonDurbin`), by Burg's method (`burg`) and by least
 * squares (`leastSquaresAr`); parametric spectra: the ARMA spectral density (`armaSpectrum`), AR spectral estimates
 * (`arPsd`), and the line-spectrum estimators MUSIC and ESPRIT with least-squares sinusoid amplitudes
 * (`sinusoidFit`); adaptive FIR filters as step-through algorithms (`lms`, `nlms`, `rls`), one sample per step.
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
