/**
 * `aifn-compute/foundation/fourier`: discrete Fourier and cosine transforms with numpy.fft's conventions, on complex128
 * tensors (design K §8.2).
 *
 * - The transforms: `fft`, `ifft`, `rfft` (a real signal's non-negative half-spectrum) and `irfft`, each with
 *   `{ axis, n, norm: 'backward' | 'ortho' | 'forward' }`. They are primitives: linear, differentiable in both modes,
 *   batched along any axis, and fast at any length (radix-2, or Bluestein's algorithm).
 * - Several axes: `fftn`, `ifftn`, and `fft2`, `ifft2` over the last two axes.
 * - Frequency grids and ordering: `fftfreq`, `rfftfreq` give each bin's frequency; `fftshift`, `ifftshift` move the
 *   zero frequency to the centre and back. `nextPowerOfTwo` and `isPowerOfTwo` choose padded lengths.
 * - The definition: `dftMatrix(n, { norm, inverse })` and `dft(x)`, the $O(n^2)$ product with it, to check the fast
 *   transforms against.
 * - Cosine transforms: `dct`, `idct` and `dctMatrix` (the orthonormal DCT-II and its inverse, as products with the
 *   matrix).
 * - `decibels` for powers and amplitudes; `readSignal` and `readValues` copy real values into a `Float64Array` for
 *   raw-array code; `fourierFunctions` is the registry of the non-primitive functions.
 *
 * Every transform returns a new tensor and leaves its input alone. Fourier analysis (windows, spectra,
 * time-frequency) is `aifn-compute/signal`.
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
