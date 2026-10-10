/**
 * Internal helpers of `aifn-methods/unsupervised/embedding`: float64 views of tensors (tensor's `dense.data`), shape
 * checks that throw naming the caller, and tensors made from flat arrays.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { matrixShape } from 'aifn-compute/learning/estimators'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The elements of a tensor in row-major order as float64, by `dense.data`: the tensor's own storage when it is already
 * dense, so the result must not be modified.
 *
 * @param t The tensor to read.
 * @returns Its elements in row-major order.
 */
export const values = (t: Tensor): Float64Array => dense.data(t)

/**
 * Rows, columns and values of a matrix; throws `ShapeError` (by `matrixShape`) when `x` is not a matrix.
 *
 * @param x The matrix, $n \times d$: one point per row.
 * @param where The caller's name, for error messages.
 * @returns `n` and `d`, and `v`, the values in row-major order (shared with `x` when it is dense: do not modify).
 */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  const [n, d] = matrixShape(x, where)
  return { n, d, v: dense.data(x) }
}

/**
 * A square matrix's size and values; throws `ShapeError` when `x` is not an $n \times n$ matrix.
 *
 * @param x The matrix, $n \times n$ (a distance or affinity matrix).
 * @param where The caller's name, for error messages.
 * @returns `n` and `v`, the values in row-major order (shared with `x` when it is dense: do not modify).
 */
export function square(x: Tensor, where: string): { n: number; v: Float64Array } {
  if (x.shape.length !== 2 || x.shape[0] !== x.shape[1])
    throw new ShapeError(where, `${where}: expected a square matrix [n, n]`)
  return { n: x.shape[0], v: values(x) }
}

/**
 * A matrix tensor over a flat array, without copying it.
 *
 * @param v The values in row-major order, $n \times d$ of them.
 * @param n The number of rows.
 * @param d The number of columns.
 * @returns The $n \times d$ tensor.
 */
export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])
/**
 * A vector tensor holding a float64 copy of `v`.
 *
 * @param v The values.
 * @returns The vector of `v.length` values.
 */
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
