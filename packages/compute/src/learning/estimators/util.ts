/**
 * Input checks every estimator makes: a feature matrix's shape and a target vector's values. Dense views of tensors
 * come from `aifn-compute/foundation/tensor`'s `dense.data`.
 */

import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Shape, Size } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'

/** The number of elements of a shape (private). */
export function sizeOf(shape: Shape): Size {
  return shape.reduce((a, b) => a * b, 1)
}

/** The rows and columns of a feature matrix [n, d]; throws naming the caller for any other rank. */
export function matrixShape(x: Tensor, where: string): [Size, Size] {
  if (x.shape.length !== 2)
    throw new ShapeError(where, `${where}: expected a matrix [n, d], got shape [${x.shape.join(', ')}]`)
  return [x.shape[0], x.shape[1]]
}

/**
 * The values of a target vector ([n], or [n, 1]) as a read-only Float64Array (shared with the tensor when it is
 * contiguous float64); throws naming the caller for any other shape.
 */
export function targetValues(y: Tensor, where: string): Float64Array {
  if (y.shape.length > 2 || (y.shape.length === 2 && y.shape[1] !== 1)) {
    throw new ShapeError(where, `${where}: expected a vector of targets [n], got shape [${y.shape.join(', ')}]`)
  }
  return dense.data(y)
}
