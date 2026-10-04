/**
 * Shared helpers of `aifn-methods/unsupervised/embedding`: float64 views of tensors (tensor's `dense.data`), shape checks
 * and small tensors.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrixShape } from 'aifn-compute/learning/estimators'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The elements of `t` in row-major order as float64 (`dense.data`: shared when already dense; do not mutate). */
export const values = (t: Tensor): Float64Array => dense.data(t)

/** Rows, columns and values of a matrix [n, d], or throw naming the caller. */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/** A square matrix's size and values, or throw. */
export function square(x: Tensor, where: string): { n: number; v: Float64Array } {
  if (x.shape.length !== 2 || x.shape[0] !== x.shape[1])
    throw new ShapeError(where, `${where}: expected a square matrix [n, n]`)
  return { n: x.shape[0], v: values(x) }
}

export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
