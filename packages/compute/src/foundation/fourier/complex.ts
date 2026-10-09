/**
 * Readers that copy real signals into fresh `Float64Array`s, for the raw-array code in `aifn-compute/signal` and
 * `aifn-compute/foundation/convolution`, and `decibels`.
 *
 * Complex values are complex128 tensors (`aifn-compute/foundation/tensor`): `abs`, `angle`, `realPart` and `imagPart`
 * take them apart.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import { copy, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { VectorLike as Signal } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { VectorLike as Signal } from 'aifn-compute/foundation/contracts'

/**
 * A real signal's values as a fresh `Float64Array`, for the raw-array transforms. A tensor must have rank 1 (it is
 * converted to float64); an array is copied as it is. The copy never shares memory with `x`.
 *
 * @param x The signal: a rank-1 tensor or an array of numbers.
 * @param what The caller's name for error messages.
 * @returns A new `Float64Array` holding the values of `x`, in order.
 *
 * @example Copy a signal into a raw array
 * print('from an array:', readSignal([1, 2, 3], 'example'))
 * print('from a tensor:', readSignal(tensor([0.5, -1]), 'example'))
 *
 * @example A matrix is not a signal
 * try {
 *   readSignal(tensor([[1, 2], [3, 4]]), 'mySpectrum')
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
export function readSignal(x: Signal, what: string): Float64Array {
  if (isTensor(x)) {
    if (x.shape.length !== 1)
      throw new ShapeError(what, `${what}: expected a rank-1 signal, got shape [${x.shape.join(', ')}]`)
    return copy(x, 'float64').data as Float64Array
  }
  return Float64Array.from(x)
}

/**
 * Any real tensor's values, in row-major order, as a fresh `Float64Array` (an array is copied as it is). Unlike
 * `readSignal`, any rank is accepted.
 *
 * @param x A real tensor of any shape, or an array of numbers.
 * @returns A new `Float64Array` of every value of `x`, flattened row by row.
 *
 * @example A matrix flattened row by row
 * print('values =', readValues(tensor([[1, 2], [3, 4]])))
 */
export function readValues(x: Tensor | ArrayLike<number>): Float64Array {
  return isTensor(x) ? (copy(x, 'float64').data as Float64Array) : Float64Array.from(x)
}

/**
 * Decibels, elementwise: $10 \log_{10}(x / r)$ for powers (`power: true`, the default) or $20 \log_{10}(x / r)$ for
 * amplitudes, with $r$ the reference. Zero maps to $-\infty$ and a negative value to NaN; there is no floor (clip for
 * display in the figure, where the choice is visible). Not differentiable: the values are read out of `x`.
 *
 * @param x The powers or amplitudes: a tensor of any shape, or an array of numbers.
 * @param options How `x` is read.
 * @param options.power True (default) when `x` holds powers (factor 10); false when it holds amplitudes (factor 20).
 * @param options.reference The value $r$ that maps to 0 dB (default 1), in the units of `x`.
 * @returns A float64 tensor of decibels with the shape of `x` (a vector when `x` is an array).
 *
 * @example Powers and amplitudes
 * print('power dB =', decibels([1, 10, 100, 0.5]))
 * print('amplitude dB =', decibels([1, 10, 100, 0.5], { power: false }))
 *
 * @example Relative to a reference, and zero
 * print('dB re 2 =', decibels([2, 4, 0], { reference: 2 }))
 */
export function decibels(
  x: Tensor | ArrayLike<number>,
  { power: isPower = true, reference = 1 }: { power?: boolean; reference?: number } = {},
): Tensor {
  const v = readValues(x)
  const k = isPower ? 10 : 20
  const out = v.map((u) => k * Math.log10(u / reference))
  return isTensor(x) ? fromData(out, x.shape) : fromData(out)
}
