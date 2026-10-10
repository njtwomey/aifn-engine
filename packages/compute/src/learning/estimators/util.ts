/**
 * Input checks every estimator makes: a feature matrix's shape and a target vector's values. Dense views of tensors
 * come from `aifn-compute/foundation/tensor`'s `dense.data`. Each check throws `ShapeError` naming its caller, so the
 * message says which estimator was given the wrong shape.
 */

import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Shape, Size } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The number of elements of a shape (private): the product of its sizes, 1 for a scalar.
 *
 * @param shape The shape.
 * @returns The number of elements.
 */
export function sizeOf(shape: Shape): Size {
  return shape.reduce((a, b) => a * b, 1)
}

/**
 * The rows and columns of a feature matrix, $n \times d$; throws `ShapeError` naming the caller for any other rank.
 *
 * @param x The feature matrix.
 * @param where The caller's name, for error messages.
 * @returns $[n, d]$.
 *
 * @example A matrix, and a vector that is refused
 * print('shape:', matrixShape(tensor([[1, 2, 3], [4, 5, 6]]), 'myModel'))
 * try {
 *   matrixShape(tensor([1, 2, 3]), 'myModel')
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
export function matrixShape(x: Tensor, where: string): [Size, Size] {
  if (x.shape.length !== 2)
    throw new ShapeError(where, `${where}: expected a matrix [n, d], got shape [${x.shape.join(', ')}]`)
  return [x.shape[0], x.shape[1]]
}

/**
 * The values of a target vector ($n$ values, or $n \times 1$) as a read-only Float64Array (shared with the tensor when
 * it is contiguous float64); throws `ShapeError` naming the caller for any other shape.
 *
 * @param y The targets.
 * @param where The caller's name, for error messages.
 * @returns The $n$ target values; not to be written.
 *
 * @example A column of targets is read as a vector
 * print('values:', targetValues(tensor([[1], [2], [3]]), 'myModel'))
 * try {
 *   targetValues(tensor([[1, 2], [3, 4]]), 'myModel')
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
export function targetValues(y: Tensor, where: string): Float64Array {
  if (y.shape.length > 2 || (y.shape.length === 2 && y.shape[1] !== 1)) {
    throw new ShapeError(where, `${where}: expected a vector of targets [n], got shape [${y.shape.join(', ')}]`)
  }
  return dense.data(y)
}
