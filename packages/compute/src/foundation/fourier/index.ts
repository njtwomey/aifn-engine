/**
 * `aifn-compute/foundation/fourier`: discrete Fourier transforms with numpy.fft's conventions, on complex128 tensors (design K
 * §8.2).
 *
 * - Primitives (linear, differentiable, batched along any axis): `fft`, `ifft`, `rfft`, `irfft`, each with
 *   `{ axis, n, norm: 'backward' | 'ortho' | 'forward' }`. Any length is fast (radix-2 or Bluestein).
 * - Compositions: `fftn`, `ifftn`, `fft2`, `ifft2`, `fftshift`, `ifftshift`; grids `fftfreq`, `rfftfreq`;
 *   `nextPowerOfTwo`, `isPowerOfTwo`.
 * - The definition: `dftMatrix(n, { norm })` and `dft(x)`, the O(n²) product with it.
 * - `dct`, `idct`, `dctMatrix` (orthonormal DCT-II and its inverse); `decibels`.
 *
 * Fourier analysis (windows, spectra, time-frequency) is `aifn-compute/signal`.
 */

export { decibels, type Signal } from './complex'
export {
  dft,
  dftMatrix,
  fft,
  fft2,
  fftfreq,
  fftn,
  fftshift,
  ifft,
  ifft2,
  ifftn,
  ifftshift,
  irfft,
  isPowerOfTwo,
  nextPowerOfTwo,
  rfft,
  rfftfreq,
  type FftNorm,
  type FftOptions,
  type FftnOptions,
} from './fft'
export { dct, dctMatrix, idct } from './dct'

// Raw-array readers for `aifn-compute/signal` and `aifn-compute/foundation/convolution`.
export { readSignal, readValues } from './complex'
// `transformInPlace` is the primitives' kernel only (not exported): call fft/rfft/ifft/irfft.
export { fourierFunctions } from './registry'
