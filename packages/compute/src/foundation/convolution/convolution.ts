/**
 * Convolution and cross-correlation of one-dimensional signals with `numpy`'s and `scipy.signal`'s conventions, and
 * multirate filtering.
 *
 * The functions are compositions over the one convolution family (`conv`, `convTranspose`), so they differentiate,
 * batch and switch between the direct and FFT kernels as it does. Signals are rank-1 tensors or arrays of numbers;
 * the modes `full`, `same` and `valid` and the argument orders are those of `scipy.signal`.
 */

import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  conj,
  fromData,
  isTensor,
  isTraced,
  shapeOfValue,
  slice,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { conv, convTranspose, type ConvMethod } from './conv'

/** A signal: a rank-1 tensor or an array of numbers. */
type Signal = VectorLike

/**
 * Which part of the full convolution to return: `full` (all of it, length $n + m - 1$ for inputs of lengths $n$ and
 * $m$), `same` (the centre, with the first input's length $n$) or `valid` (the part computed without zero padding,
 * length $\lvert n - m \rvert + 1$).
 */
export type ConvolutionMode = 'full' | 'same' | 'valid'

/** Options for `convolve` and `correlate`. */
export interface ConvolveOptions {
  /** Which part of the full result to return (default `full`). */
  mode?: ConvolutionMode
  /** `direct` ($O(nm)$), `fft`, `overlapAdd`, or `auto` (the cheaper of direct and FFT). Default `auto`. */
  method?: ConvMethod
}

/**
 * A signal as a rank-1 value: an array of numbers becomes a tensor, and tensors and traced values pass through. Throws
 * `ShapeError` when the result is not rank-1.
 *
 * @param x The signal: a rank-1 tensor, a traced value or an array of numbers.
 * @param what The caller's name for error messages.
 * @returns The signal as a rank-1 value.
 */
function signal(x: Value | Signal, what: string): Value {
  const v = isTraced(x) || isTensor(x) ? (x as Value) : fromData(Float64Array.from(x as ArrayLike<number>))
  const shape = shapeOfValue(v)
  if (shape.length !== 1)
    throw new ShapeError(what, `${what}: expected a rank-1 signal, got shape [${shape.join(', ')}]`)
  return v
}

/**
 * A rank-1 value reversed.
 *
 * @param v The value.
 * @returns Its samples in reverse order (a slice, so differentiable).
 */
const reversed = (v: Value): Value => slice(v, [null, null, -1])

/**
 * The convolution $(x * h)_n = \sum_k x_k h_{n-k}$, as `scipy.signal.convolve`: of length
 * $\lvert x \rvert + \lvert h \rvert - 1$ in `full` mode (the default). Convolution commutes, so in `valid` mode the
 * shorter input is the kernel whichever argument it is. An empty input gives an empty result. Differentiable in both
 * inputs and batched by `vmap`; complex inputs give a complex result (no conjugation).
 *
 * @param x The first signal: a rank-1 tensor or an array of numbers. Its length is the output length in `same` mode.
 * @param h The second signal (the filter), in the same forms.
 * @param options The `mode` (default `full`) and the kernel `method` (default `auto`).
 * @returns The convolution, rank-1.
 *
 * @example Multiplying polynomials
 * // (1 + 2z)(1 + 3z + z^2): coefficients from the constant term up.
 * print('product =', convolve([1, 2], [1, 3, 1]))
 *
 * @example A moving average in each mode
 * const x = [1, 2, 3, 4, 5]
 * const h = [1 / 3, 1 / 3, 1 / 3]
 * print('full  ', convolve(x, h))
 * print('same  ', convolve(x, h, { mode: 'same' }))
 * print('valid ', convolve(x, h, { mode: 'valid' }))
 */
export function convolve(x: Signal, h: Signal, options?: ConvolveOptions): Tensor
export function convolve(x: Value | Signal, h: Value | Signal, options?: ConvolveOptions): Value
export function convolve(
  x: Value | Signal,
  h: Value | Signal,
  { mode = 'full', method = 'auto' }: ConvolveOptions = {},
): Value {
  let a = signal(x, 'convolve')
  let b = signal(h, 'convolve')
  const [n] = shapeOfValue(a)
  const [m] = shapeOfValue(b)
  if (n === 0 || m === 0) return fromData(new Float64Array(0))
  // Convolution commutes; scipy's `valid` takes the shorter input as the kernel.
  if (mode === 'valid' && m > n) [a, b] = [b, a]
  return conv(a, b, { padding: mode, method, flip: true })
}

/**
 * Convolution by FFT, as `scipy.signal.fftconvolve`: `convolve` with the `fft` method, so equal to it up to rounding
 * and differentiable in the same way.
 *
 * @param x The first signal: a rank-1 tensor or an array of numbers.
 * @param h The second signal (the filter), in the same forms.
 * @param options The `mode` (default `full`).
 * @returns The convolution, rank-1.
 *
 * @example The same result as the direct sum, up to rounding
 * const x = [1, 2, 3, 4]
 * const h = [0.5, -1, 0.25]
 * print('fft    ', fftConvolve(x, h))
 * print('direct ', convolve(x, h, { method: 'direct' }))
 */
export function fftConvolve(x: Signal, h: Signal, options?: { mode?: ConvolutionMode }): Tensor
export function fftConvolve(x: Value | Signal, h: Value | Signal, options?: { mode?: ConvolutionMode }): Value
export function fftConvolve(
  x: Value | Signal,
  h: Value | Signal,
  { mode = 'full' }: { mode?: ConvolutionMode } = {},
): Value {
  return convolve(x, h, { mode, method: 'fft' })
}

/**
 * The cross-correlation $z_k = \sum_n x_{n + k - (\lvert y \rvert - 1)}\, \overline{y_n}$, as
 * `scipy.signal.correlate`: the convolution of $x$ with $y$ reversed (and conjugated when complex). Use
 * `correlationLags` for the lag of each output. An empty input gives an empty result. Differentiable in both inputs.
 *
 * @param x The signal searched: a rank-1 tensor or an array of numbers.
 * @param y The signal (template) it is correlated with, in the same forms; conjugated when complex.
 * @param options The `mode` (default `full`) and the kernel `method` (default `auto`).
 * @returns The cross-correlation, rank-1, in order of increasing lag.
 *
 * @example Find where a template occurs
 * const x = [0, 0, 1, 2, 1, 0, 0]
 * const y = [1, 2, 1]
 * print('z    =', correlate(x, y))
 * print('lags =', correlationLags(x.length, y.length))
 */
export function correlate(x: Signal, y: Signal, options?: ConvolveOptions): Tensor
export function correlate(x: Value | Signal, y: Value | Signal, options?: ConvolveOptions): Value
export function correlate(
  x: Value | Signal,
  y: Value | Signal,
  { mode = 'full', method = 'auto' }: ConvolveOptions = {},
): Value {
  const a = signal(x, 'correlate')
  const b = conj(signal(y, 'correlate'))
  const [n] = shapeOfValue(a)
  const [m] = shapeOfValue(b)
  if (n === 0 || m === 0) return fromData(new Float64Array(0))
  // correlate(x, y) = convolve(x, reversed y) = convolve(reversed y, x): the shorter one is the kernel in `valid`.
  if (mode === 'valid' && m > n) return conv(reversed(b), a, { padding: 'valid', method, flip: true })
  return conv(a, b, { padding: mode, method, flip: false })
}

/**
 * The lags of `correlate(x, y, { mode })` for inputs of lengths $n_1$ and $n_2$, as `scipy.signal.correlation_lags`:
 * output $k$ of the correlation compares $x$ shifted by the $k$-th lag against $y$.
 *
 * @param n1 The length of the first input, `x`.
 * @param n2 The length of the second input, `y`.
 * @param mode The mode the correlation was computed in.
 * @returns The lags as an int32 tensor, one per output of the correlation: $-(n_2 - 1)$ to $n_1 - 1$ in `full` mode,
 *   the centre $n_1$ of those in `same`, and the lags without zero padding in `valid`.
 *
 * @example The lags of each mode
 * print('full  ', correlationLags(5, 3))
 * print('same  ', correlationLags(5, 3, 'same'))
 * print('valid ', correlationLags(5, 3, 'valid'))
 */
export function correlationLags(n1: number, n2: number, mode: ConvolutionMode = 'full'): Tensor {
  let lags = Array.from({ length: n1 + n2 - 1 }, (_, i) => i - (n2 - 1))
  if (mode === 'same') {
    const mid = Math.floor(lags.length / 2)
    const bound = Math.floor(n1 / 2)
    lags = n1 % 2 === 0 ? lags.slice(mid - bound, mid + bound) : lags.slice(mid - bound, mid + bound + 1)
  } else if (mode === 'valid') {
    const bound = n1 - n2
    lags =
      bound >= 0
        ? Array.from({ length: bound + 1 }, (_, i) => i)
        : Array.from({ length: 1 - bound }, (_, i) => bound + i)
  }
  return fromData(Int32Array.from(lags))
}

/**
 * Upsample, filter, downsample, as `scipy.signal.upfirdn`: $u - 1$ zeros after each sample of $x$, the full
 * convolution with $h$, then every $q$-th sample, for a length of
 * $\lfloor ((\lvert x \rvert - 1) u + \lvert h \rvert - 1) / q \rfloor + 1$ ($u$ = `up`, $q$ = `down`). The first two
 * steps are one transposed convolution with stride $u$ (interpolation); the last is a strided slice (decimation),
 * equal to a convolution with stride $q$. Differentiable in $x$ and $h$. Throws `ShapeError` when `up` or `down` is
 * not a positive integer.
 *
 * @param h The FIR filter: a rank-1 tensor or an array of numbers. First, as in scipy.
 * @param x The signal to resample, in the same forms.
 * @param options The resampling factors and the kernel.
 * @param options.up The upsampling factor $u$: $u - 1$ zeros are inserted after each sample (default 1, none).
 * @param options.down The downsampling factor $q$: every $q$-th sample of the filtered signal is kept, from the first
 *   (default 1, all).
 * @param options.method The convolution kernel (default `auto`).
 * @returns The resampled signal, rank-1.
 *
 * @example Upsample by 2 with linear interpolation
 * print('y =', upfirdn([0.5, 1, 0.5], [1, 2, 3], { up: 2 }))
 *
 * @example Average pairs, then keep every second sample
 * print('y =', upfirdn([0.5, 0.5], [1, 3, 5, 7, 9, 11], { down: 2 }))
 */
export function upfirdn(
  h: Value | Signal,
  x: Value | Signal,
  { up = 1, down = 1, method = 'auto' }: { up?: number; down?: number; method?: ConvMethod } = {},
): Value {
  if (!(Number.isInteger(up) && up >= 1 && Number.isInteger(down) && down >= 1))
    throw new ShapeError('upfirdn', 'upfirdn: up and down must be positive integers')
  const xs = signal(x, 'upfirdn')
  const hs = signal(h, 'upfirdn')
  const y = convTranspose(xs, hs, { stride: up, flip: false, method })
  return down === 1 ? y : slice(y, [null, null, down])
}
