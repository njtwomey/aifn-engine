/**
 * Convolution and cross-correlation of 1-D signals with numpy's and scipy.signal's conventions, and multirate
 * filtering: compositions over the one convolution family (`conv`, `convTranspose`), so they differentiate, batch and
 * switch between the direct and FFT kernels as it does.
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

/** Which part of the full convolution to return: all of it, the centre with the first input's length, or the part without zero padding. */
export type ConvolutionMode = 'full' | 'same' | 'valid'

/** Options for `convolve` and `correlate`. */
export interface ConvolveOptions {
  mode?: ConvolutionMode
  /** `direct` O(nm), `fft`, `overlapAdd`, or `auto` (the cheaper of direct and FFT). Default `auto`. */
  method?: ConvMethod
}

/** A signal as a rank-1 value (traced values pass through). */
function signal(x: Value | Signal, what: string): Value {
  const v = isTraced(x) || isTensor(x) ? (x as Value) : fromData(Float64Array.from(x as ArrayLike<number>))
  const shape = shapeOfValue(v)
  if (shape.length !== 1)
    throw new ShapeError(what, `${what}: expected a rank-1 signal, got shape [${shape.join(', ')}]`)
  return v
}

const reversed = (v: Value): Value => slice(v, [null, null, -1])

/** The convolution (x ∗ h)[n] = Σ_k x[k] h[n − k], as `scipy.signal.convolve` (full length |x| + |h| − 1 by default). */
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

/** Convolution by FFT, as `scipy.signal.fftconvolve`. */
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
 * The cross-correlation z[k] = Σ_n x[n + k − (|y| − 1)] conj(y[n]), as `scipy.signal.correlate`: the convolution of x
 * with y reversed (and conjugated when complex). Use `correlationLags` for the lag of each output.
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

/** The lags of `correlate(x, y, { mode })` for inputs of lengths n1 and n2, as `scipy.signal.correlation_lags`. */
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
 * Upsample, filter, downsample, as `scipy.signal.upfirdn`: `up − 1` zeros after each sample of x, the full convolution
 * with h, then every `down`-th sample; length ⌊((|x| − 1)·up + |h| − 1)/down⌋ + 1. The first two steps are one
 * transposed convolution with stride `up` (interpolation); the last is a strided slice (decimation), equal to a
 * convolution with stride `down`. Differentiable in x and h.
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
