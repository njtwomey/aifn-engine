/**
 * `aifn-compute/foundation/convolution`: one convolution family (design K §8.3).
 *
 * - `conv(x, w, options)`: n-dimensional convolution (`flip`, the default) or cross-correlation of [N, C, ...S] inputs
 *   with [O, C/groups, ...K] kernels (layouts `ncw`, `nchw`, `ncdhw`; a rank-1 signal is [1, 1, n] underneath), with
 *   stride, dilation, groups, padding (integers, [lo, hi] pairs, `valid`, `same`, `full`) and a kernel `method`
 *   (`direct`, `fft`, `overlapAdd`, `auto`). `convTranspose` is its input adjoint (upsampling, synthesis). Both are
 *   bilinear primitives with a closed family of transposes, so they differentiate to any order and batch without a loop.
 * - `pad(x, widths, mode)`: constant, reflect, symmetric, edge and wrap borders; a linear movement primitive.
 * - Signals, with numpy's and scipy.signal's conventions: `convolve`, `correlate` and `fftConvolve` in `full`, `same`
 *   or `valid` mode, `correlationLags`, and `upfirdn` (interpolation by a strided transposed convolution, decimation by
 *   a strided slice).
 * - Images, as scipy.ndimage: `correlate2d`, `convolve2d` and `separableFilter` with border modes (pad, then `valid`).
 * - Recurrences: `linearFilter(b, a, x, { axis, zi })`, the difference equation Σ aₖ y[t−k] = Σ bₖ x[t−k] (IIR feedback
 *   is not a convolution), a primitive differentiable in b, a, x and zi, real or complex, batched along other axes.
 */

export {
  conv,
  convOutputSize,
  convTranspose,
  type ConvLayout,
  type ConvMethod,
  type ConvOptions,
  type ConvPadding,
  type Ints,
} from './conv'
export { pad, type PadMode, type PadWidths } from './pad'
export {
  convolve,
  correlate,
  correlationLags,
  fftConvolve,
  upfirdn,
  type ConvolutionMode,
  type ConvolveOptions,
} from './convolution'
export { correlate2d, convolve2d, readImage, separableFilter, type Border, type ImageInput } from './image'
export { linearFilter, type LinearFilterOptions } from './filter'
export { convolutionFunctions } from './registry'
