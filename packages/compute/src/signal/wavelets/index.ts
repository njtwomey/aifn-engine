/**
 * `aifn-compute/signal/wavelets`: wavelets: the discrete wavelet transform and its inverse (`dwt`, `idwt`), multilevel
 * decomposition and reconstruction (`wavedec`, `waverec`), filters and wavelet functions, and the continuous wavelet
 * transform with the Morlet wavelet; wavelet shrinkage (`waveletThreshold`, `waveletDenoise`); `waveletRegistry` lists the wavelets with their vanishing moments and lengths.
 */

export {
  cwt,
  dwt,
  idwt,
  morlet,
  wavedec,
  waveletFilters,
  wavefun,
  waverec,
  waveletDenoise,
  waveletThreshold,
  type Cwt,
  type WaveletDecomposition,
  type WaveletFilters,
  type WaveletDenoised,
  type WaveletDenoiseOptions,
  type WaveletName,
} from './wavelets'
export { waveletRegistry, waveletsFunctions } from './registry'
