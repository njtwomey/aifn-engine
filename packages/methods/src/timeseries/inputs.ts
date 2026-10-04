/**
 * Private input conversions of `aifn-methods/timeseries`: series and coefficient arguments (`VectorLike`,
 * `MatrixLike`) as arrays of numbers for the scalar recursions, read with `aifn-compute/foundation/tensor`'s `dense`
 * converters. Matrix algebra uses `aifn-compute/foundation/tensor` and `aifn-compute/numerics/linalg`; the Kalman recursions use
 * `aifn-compute/inference/filtering`'s working form (rows of numbers), which `toSeries` produces.
 */

import { dense, isTensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

// Types defined once, in `aifn-compute/foundation/contracts` (re-exported for this module).
export type { VectorLike, MatrixLike } from 'aifn-compute/foundation/contracts'

/** A vector input as an array of numbers (`dense.toF64`). */
export function toVec(v: VectorLike, where: string): number[] {
  return Array.from(dense.toF64(v, where))
}

/** Observations as T rows of m numbers: a vector is T scalar observations, a matrix is T×m (`dense.toMatrixF64`). */
export function toSeries(y: VectorLike | MatrixLike, where: string): number[][] {
  const isVector = isTensor(y)
    ? y.shape.length === 1
    : (y as ArrayLike<unknown>).length === 0 || typeof (y as ArrayLike<unknown>)[0] === 'number'
  if (isVector) return Array.from(dense.toF64(y as VectorLike, where), (v) => [v])
  const { data, m, n } = dense.toMatrixF64(y as MatrixLike, where)
  return Array.from({ length: m }, (_, i) => Array.from(data.subarray(i * n, (i + 1) * n)))
}
