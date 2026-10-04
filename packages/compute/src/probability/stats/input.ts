/**
 * Inputs to the statistics functions: plain numeric arrays or `aifn-compute/foundation/tensor` tensors, and reductions of a tensor along
 * one axis. Private to `aifn-compute/probability/stats`.
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

/** The elements of a tensor in row-major order, as a new Float64Array (any strides and dtype). */
function tensorValues(t: Tensor): Float64Array {
  return copy(t, 'float64').data as Float64Array
}

/** The values of a sequence: an array as is, or a rank-1 tensor's elements. Other ranks are an error. */
export function toSequence(x: Data, what: string): ArrayLike<number> {
  if (!isTensor(x)) return x
  if (x.shape.length !== 1)
    throw new ShapeError('stats', `stats: ${what} needs a rank-1 tensor, got shape [${x.shape.join(', ')}]`)
  return tensorValues(x)
}

/** Every element of the data: an array as is, or a tensor of any rank flattened in row-major order. */
export function allValues(x: Data): ArrayLike<number> {
  return isTensor(x) ? tensorValues(x) : x
}

/**
 * Apply a statistic of one sequence to every lane of a tensor along `axis`: the axis is moved last and the tensor
 * copied contiguous, so each lane is one run of values. The result has the other axes (with a length-1 axis in place
 * of `axis` when `keepDims`).
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
 * with array input is an error, since an array has no axes to choose between.
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

/** A rank-1 tensor over an array of results (float64, or int32 for indices), without copying typed arrays. */
export function vectorOf(a: Float64Array | Int32Array | readonly number[]): Tensor {
  const data = a instanceof Float64Array || a instanceof Int32Array ? a : Float64Array.from(a)
  return fromData(data, [data.length])
}
