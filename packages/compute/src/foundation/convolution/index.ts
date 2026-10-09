/**
 * `aifn-compute/foundation/convolution`: convolution, correlation, padding and recursive filtering of signals, images
 * and volumes, as one differentiable family (design K §8.3).
 *
 * - The convolution: `conv(x, w, options)` is the $n$-dimensional convolution (`flip`, the default) or
 *   cross-correlation of `[N, C, ...S]` inputs with `[O, C/groups, ...K]` kernels (layouts `ncw`, `nchw`, `ncdhw`; a
 *   rank-1 signal is `[1, 1, n]` underneath), with stride, dilation, groups, padding (integers, `[lo, hi]` pairs,
 *   `valid`, `same`, `full`) and a kernel `method` (`direct`, `fft`, `overlapAdd`, `auto`). `convTranspose` is its
 *   input adjoint (upsampling, synthesis), and `convOutputSize` the output length of a layer.
 * - Padding: `pad(x, widths, mode)` with constant, reflect, symmetric, edge and wrap borders, as `numpy.pad`.
 * - Signals, with `numpy`'s and `scipy.signal`'s conventions: `convolve`, `correlate` and `fftConvolve` in `full`,
 *   `same` or `valid` mode, `correlationLags` for the lag of each correlation output, and `upfirdn` (interpolation by a
 *   strided transposed convolution, decimation by a strided slice).
 * - Images, as `scipy.ndimage`: `correlate2d`, `convolve2d` and `separableFilter` with border modes (pad, then
 *   `valid`), and `readImage` for filters written over raw arrays.
 * - Recurrences: `linearFilter(b, a, x, { axis, zi })`, the difference equation
 *   $\sum_k a_k y_{t-k} = \sum_k b_k x_{t-k}$ (IIR feedback is not a convolution), real or complex, batched along
 *   the other axes.
 * - `convolutionFunctions`: the registry entries of the functions that are not primitives.
 *
 * `conv`, `convTranspose`, `pad` and `linearFilter` are primitives and everything else is composed from them, so the
 * whole module differentiates to any order in every input (`conv` and `convTranspose` are bilinear, with a closed
 * family of transposes; `linearFilter`'s rules are written with itself) and batches under `vmap`, without a loop
 * except for `linearFilter` with batched coefficients. Malformed shapes throw `ShapeError`.
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
