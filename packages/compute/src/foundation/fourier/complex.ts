/**
 * Readers of real signals for the raw-array code in `aifn-compute/signal` and `aifn-compute/foundation/convolution`, and `decibels`.
 * Complex values are complex128 tensors (`aifn-compute/foundation/tensor`): `abs`, `angle`, `realPart`, `imagPart` replace
 * the former `{ re, im }` pair helpers.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import { copy, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import type { VectorLike as Signal } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { VectorLike as Signal } from 'aifn-compute/foundation/contracts'

/**
 * A real signal's values as a fresh Float64Array, for the raw-array transforms (rank 1 required for tensors); `what`
 * names the caller in errors.
 */
export function readSignal(x: Signal, what: string): Float64Array {
  if (isTensor(x)) {
    if (x.shape.length !== 1)
      throw new ShapeError(what, `${what}: expected a rank-1 signal, got shape [${x.shape.join(', ')}]`)
    return copy(x, 'float64').data as Float64Array
  }
  return Float64Array.from(x)
}

/** Any real tensor's values (row-major) as a fresh Float64Array. */
export function readValues(x: Tensor | ArrayLike<number>): Float64Array {
  return isTensor(x) ? (copy(x, 'float64').data as Float64Array) : Float64Array.from(x)
}

/**
 * Decibels: 10 log₁₀(x / reference) for powers (`power: true`, the default) or 20 log₁₀(x / reference) for
 * amplitudes. Zero maps to −∞; there is no floor (clip for display in the figure, where the choice is visible).
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
