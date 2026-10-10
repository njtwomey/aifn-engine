/**
 * `aifn-compute/signal/wavelets`: orthogonal discrete wavelet transforms, the Morlet continuous wavelet transform, and
 * wavelet shrinkage.
 *
 * - Discrete transforms, periodic and orthogonal (Haar and Daubechies `db1` to `db10`): one level with `dwt` and
 *   `idwt`, several with `wavedec` and `waverec` (details finest first; the length must divide by $2^J$).
 * - The wavelets themselves: `waveletFilters` (the four filters, as pywt's `filter_bank`), `wavefun` (the scaling
 *   function $\phi$ and wavelet $\psi$ by the cascade algorithm) and `morlet`.
 * - Continuous transform: `cwt`, the Morlet scalogram at given frequencies in Hz, returned as a `TimeFrequency`.
 * - Denoising: `waveletThreshold` (soft or hard) and `waveletDenoise` (shrinkage at the universal threshold).
 * - `waveletRegistry` lists the wavelets with their vanishing moments and lengths; `waveletsFunctions` the functions.
 *
 * Conventions follow PyWavelets with periodic extension. The transforms read a `Signal` or bare samples; unknown
 * wavelet names throw `DomainError` and lengths that do not halve throw `ShapeError`.
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
