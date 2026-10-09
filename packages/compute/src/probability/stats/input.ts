/**
 * Internal helpers: the inputs to the statistics functions (plain numeric arrays or `aifn-compute/foundation/tensor`
 * tensors), read as one sequence or as every element, and reductions of a tensor along one axis. Private to
 * `aifn-compute/probability/stats`, apart from the `AxisOption` and `Data` types.
 */

import { copy, fromData, isTensor, permute, type Tensor } from 'aifn-compute/foundation/tensor'
import type { DataLike as Data, Scalar } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { DataLike as Data } from 'aifn-compute/foundation/contracts'

/** Options of a reduction that can run along one axis of a tensor. */
export type AxisOption = {
  /**
   * Reduce a tensor along this axis only (negative counts from the end), returning a tensor of the other axes. Without
   * it, every element is reduced to one number.
   */
  axis?: number
  /** With `axis`, keep the reduced axis with length 1 (NumPy's `keepdims`). Default false. */
  keepDims?: boolean
}

/**
 * The elements of a tensor in row-major order, as a new Float64Array (any strides and dtype).
 *
 * @param t The tensor to read; not modified.
 * @returns A float64 copy of its elements.
 */
function tensorValues(t: Tensor): Float64Array {
  return copy(t, 'float64').data as Float64Array
}

/**
 * The values of a sequence: an array as is, or a rank-1 tensor's elements. A tensor of another rank throws
 * `ShapeError`.
 *
 * @param x The sequence: a plain numeric array (returned without copying) or a rank-1 tensor (copied).
 * @param what The caller's name for `x`, used in the error message.
 * @returns The values, in order.
 */
export function toSequence(x: Data, what: string): ArrayLike<number> {
  if (!isTensor(x)) return x
  if (x.shape.length !== 1)
    throw new ShapeError('stats', `stats: ${what} needs a rank-1 tensor, got shape [${x.shape.join(', ')}]`)
  return tensorValues(x)
}

/**
 * Every element of the data: an array as is, or a tensor of any rank flattened in row-major order.
 *
 * @param x A plain numeric array (returned without copying) or a tensor of any rank (copied).
 * @returns The values.
 */
export function allValues(x: Data): ArrayLike<number> {
  return isTensor(x) ? tensorValues(x) : x
}

/**
 * Apply a statistic of one sequence to every lane of a tensor along `axis`: the axis is moved last and the tensor
 * copied contiguous, so each lane is one run of values. The result has the other axes (with a length-1 axis in place
 * of `axis` when `keepDims`). An axis out of range throws `ShapeError`.
 *
 * @param x The tensor to reduce; not modified.
 * @param axis The axis to reduce along; a negative one counts from the end.
 * @param keepDims Keep the reduced axis with length 1 instead of dropping it.
 * @param statistic The statistic of one lane: it receives the lane's values (a view of a scratch array, an empty one
 *   when the axis has length 0) and returns one number.
 * @param what The caller's name for error messages.
 * @returns A float64 tensor of the statistic of each lane.
 */
export function alongAxis(
  x: Tensor,
  axis: number,
  keepDims: boolean,
  statistic: (lane: Float64Array) => number,
  what: string,
): Tensor {
  const rank = x.shape.length
  const a = axis < 0 ? axis + rank : axis
  if (!Number.isInteger(a) || a < 0 || a >= rank)
    throw new ShapeError('stats', `stats: ${what}: axis ${axis} is out of range for rank ${rank}`)
  const order = [...Array.from({ length: rank }, (_, k) => k).filter((k) => k !== a), a]
  const lanes = tensorValues(permute(x, order))
  const length = x.shape[a]
  const count = length === 0 ? x.shape.filter((_, k) => k !== a).reduce((p, d) => p * d, 1) : lanes.length / length
  const out = new Float64Array(count)
  for (let i = 0; i < count; i++) out[i] = statistic(lanes.subarray(i * length, (i + 1) * length))
  const shape = keepDims ? x.shape.map((d, k) => (k === a ? 1 : d)) : x.shape.filter((_, k) => k !== a)
  return fromData(out, shape)
}

/**
 * Run a reduction: along `options.axis` of a tensor (a tensor result), or over every element (a number). An axis
 * with array input throws `DomainError`, since an array has no axes to choose between.
 *
 * @param x The data: a plain numeric array or a tensor of any rank.
 * @param options The axis to reduce along (none: every element) and whether to keep it with length 1.
 * @param statistic The statistic of a sequence of values, applied to every element or to each lane.
 * @param what The caller's name for error messages.
 * @returns The statistic as a number, or a tensor of it per lane when an axis is given.
 */
export function reduce(
  x: Data,
  options: AxisOption,
  statistic: (values: ArrayLike<number>) => number,
  what: string,
): Scalar | Tensor {
  if (options.axis === undefined) return statistic(allValues(x))
  if (!isTensor(x)) throw new DomainError('stats', `stats: ${what}: axis needs a tensor input`)
  return alongAxis(x, options.axis, options.keepDims ?? false, statistic, what)
}

/**
 * A rank-1 tensor over an array of results (float64, or int32 for indices), without copying typed arrays.
 *
 * @param a The results: a `Float64Array` or `Int32Array` (used as the tensor's storage) or a plain array (copied to
 *   float64).
 * @returns The rank-1 tensor of the values.
 */
export function vectorOf(a: Float64Array | Int32Array | readonly number[]): Tensor {
  const data = a instanceof Float64Array || a instanceof Int32Array ? a : Float64Array.from(a)
  return fromData(data, [data.length])
}
