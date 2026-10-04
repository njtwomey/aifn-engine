/**
 * The tensor type, its storage and the strided iteration every other tensor function builds on.
 *
 * The layout follows NumPy's ndarray (Harris et al., 2020, "Array programming with NumPy", Nature 585): a flat typed
 * array, a shape, per-axis strides in elements and an offset. A view is a new header over the same data.
 */

import type { Axes, DType, Tensor, TensorBrand, TensorData } from 'aifn-compute/foundation/contracts'
import { DTypeError, ShapeError } from 'aifn-compute/foundation/errors'
import { promoteTypes, weakType } from './dtype'

/**
 * The tensor brand: a symbol key that only `fromData` (and the view constructor built on the same header) sets, so
 * that `isTensor` never mistakes another object with `shape`, `strides` and `data` keys for a tensor. It is registered
 * (`Symbol.for`) so that two copies of this module agree. `structuredClone` drops symbol keys: a tensor that crossed a
 * worker boundary is re-branded with `revive`. Its type is `aifn-compute/foundation/contracts`' `TensorBrand`, so the contract's
 * `Tensor` and this one are the same type.
 */
export const TENSOR: TensorBrand = Symbol.for('aifn.tensor') as TensorBrand

// The tensor types are defined once, in `aifn-compute/foundation/contracts`.
export type {
  Axes,
  DType,
  Matrix,
  NestedArray,
  Tensor,
  TensorBrand,
  TensorData,
  TensorLike,
  Vector,
} from 'aifn-compute/foundation/contracts'

/** Allocate storage for `size` elements of a dtype (2·size doubles for complex128, interleaved re, im). */
export function allocate(dtype: DType, size: number): TensorData {
  if (dtype === 'float64') return new Float64Array(size)
  if (dtype === 'complex128') return new Float64Array(2 * size)
  if (dtype === 'float32') return new Float32Array(size)
  if (dtype === 'bool') return new Uint8Array(size)
  return new Int32Array(size)
}

/** The dtype a typed array holds by default (a Float64Array is float64 unless complex128 is said explicitly). */
export function dtypeOf(data: TensorData): DType {
  if (data instanceof Float64Array) return 'float64'
  if (data instanceof Float32Array) return 'float32'
  if (data instanceof Uint8Array) return 'bool'
  return 'int32'
}

/** Number of elements of a shape (1 for the scalar shape `[]`). */
export function sizeOf(shape: readonly number[]): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

/** Number of elements of a tensor. */
export function size(t: Tensor): number {
  return sizeOf(t.shape)
}

/** Row-major (C order) strides of a shape, in elements. */
export function rowMajorStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length)
  let step = 1
  for (let k = shape.length - 1; k >= 0; k--) {
    strides[k] = step
    step *= shape[k]
  }
  return strides
}

/** Check that a shape is a list of non-negative integers. */
export function checkShape(shape: readonly number[], where: string): void {
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0)
      throw new ShapeError(where, `${where}: invalid shape [${shape.join(', ')}]`, [shape])
  }
}

/**
 * Wrap a typed array as a tensor without copying. `shape` defaults to all of `data` as a vector; strides are
 * row-major. `dtype` defaults to the typed array's own (`dtypeOf`); pass `'complex128'` for a Float64Array of
 * interleaved (re, im) pairs, which holds `data.length / 2` elements.
 *
 * The caller hands over `data`: it must not be mutated afterwards, since tensors are immutable by convention.
 */
export function fromData(data: TensorData, shape?: readonly number[], dtype: DType = dtypeOf(data)): Tensor {
  const width = dtype === 'complex128' ? 2 : 1
  if (dtype !== dtypeOf(data) && !(dtype === 'complex128' && data instanceof Float64Array))
    throw new DTypeError('fromData', `fromData: a ${data.constructor.name} cannot hold ${dtype}`, [dtype])
  const dims = shape ?? [data.length / width]
  checkShape(dims, 'fromData')
  if (sizeOf(dims) * width !== data.length) {
    throw new ShapeError(
      'fromData',
      `fromData: ${data.length} values do not fill shape [${dims.join(', ')}] of ${dtype}`,
      [dims],
    )
  }
  return { [TENSOR]: true, shape: [...dims], strides: rowMajorStrides(dims), offset: 0, dtype, data }
}

/**
 * A tensor header over the data of an existing tensor; used internally to build views. It is the only other place a
 * tensor is constructed, and it brands its result as `fromData` does. `dtype` differs from `t`'s only for the float64
 * views of a complex tensor's parts (`complexPartView`).
 */
export function view(
  t: Tensor,
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  dtype: DType = t.dtype,
): Tensor {
  return { [TENSOR]: true, shape: [...shape], strides: [...strides], offset, dtype, data: t.data }
}

/**
 * The real (`part` 0) or imaginary (`part` 1) parts of a complex128 tensor as a float64 view of the same storage:
 * strides doubled, offset 2o + part (design K §8.1). Zero-copy.
 */
export function complexPartView(z: Tensor, part: 0 | 1): Tensor {
  return view(
    z,
    z.shape,
    z.strides.map((s) => 2 * s),
    2 * z.offset + part,
    'float64',
  )
}

/** True when a value is a tensor (made by `fromData` or a view of one): it carries the brand. */
export function isTensor(x: unknown): x is Tensor {
  return typeof x === 'object' && x !== null && (x as { [TENSOR]?: unknown })[TENSOR] === true
}

/** True for an unbranded object with a tensor's fields, as `structuredClone` or a worker message leaves one. */
function isTensorShaped(x: object): x is Omit<Tensor, typeof TENSOR> {
  const t = x as Partial<Tensor>
  return (
    Array.isArray(t.shape) &&
    Array.isArray(t.strides) &&
    typeof t.offset === 'number' &&
    typeof t.dtype === 'string' &&
    (t.data instanceof Float64Array ||
      t.data instanceof Float32Array ||
      t.data instanceof Int32Array ||
      t.data instanceof Uint8Array)
  )
}

/**
 * Re-brand the tensors inside a value that lost its brands (a `structuredClone`, a worker message): every unbranded
 * object with a tensor's fields becomes a tensor over the same data, with its shape, strides and offset kept. Arrays
 * and plain objects are walked and rebuilt; tensors, numbers and anything else are returned as they are.
 */
export function revive<T>(x: T): T {
  if (typeof x !== 'object' || x === null || isTensor(x)) return x
  if (isTensorShaped(x)) {
    const t = x as Omit<Tensor, typeof TENSOR>
    const out: Tensor = {
      [TENSOR]: true,
      shape: [...t.shape],
      strides: [...t.strides],
      offset: t.offset,
      dtype: t.dtype,
      data: t.data,
    }
    return out as T
  }
  if (Array.isArray(x)) return x.map((v) => revive(v as unknown)) as T
  const proto = Object.getPrototypeOf(x) as unknown
  if (proto !== Object.prototype && proto !== null) return x
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(x)) out[k] = revive(v)
  return out as T
}

/**
 * True when the tensor's elements lie in row-major order in a single run of `data`, starting at `offset`. Axes of length
 * 1 are ignored, since their stride is never used.
 */
export function isContiguous(t: Tensor): boolean {
  let step = 1
  for (let k = t.shape.length - 1; k >= 0; k--) {
    if (t.shape[k] === 1) continue
    if (t.strides[k] !== step) return false
    step *= t.shape[k]
  }
  return true
}

/**
 * A zero-copy, read-only view of a tensor's elements in row-major order (design-core §3.1 `readonly(t)`): a subarray of
 * its storage when the tensor is contiguous (any offset), else null, so a hot loop reads a tensor without the copy
 * `toFlat` makes and falls back to a copy itself when it gets null. complex128 views hold interleaved (re, im) pairs.
 * Tensors are immutable: never write to the result.
 */
export function readonlyData(t: Tensor): TensorData | null {
  if (!isContiguous(t)) return null
  const w = t.dtype === 'complex128' ? 2 : 1
  const size = t.shape.reduce((a, b) => a * b, 1)
  if (t.offset === 0 && t.data.length === size * w) return t.data
  return t.data.subarray(t.offset * w, (t.offset + size) * w) as TensorData
}

/** Normalise an axis (negative counts from the end) and check its range. */
export function normaliseAxis(axis: number, rank: number, where: string): number {
  const a = axis < 0 ? axis + rank : axis
  if (!Number.isInteger(a) || a < 0 || a >= rank)
    throw new ShapeError(where, `${where}: axis ${axis} is out of range for rank ${rank}`)
  return a
}

/** Normalise one or several axes; `undefined` or `null` means every axis. Duplicates are an error. */
export function normaliseAxes(axis: Axes | null | undefined, rank: number, where: string): number[] {
  if (axis === undefined || axis === null) return Array.from({ length: rank }, (_, k) => k)
  const list = typeof axis === 'number' ? [axis] : [...axis]
  const out = list.map((a) => normaliseAxis(a, rank, where))
  if (new Set(out).size !== out.length) throw new ShapeError(where, `${where}: repeated axis in [${list.join(', ')}]`)
  return out.sort((a, b) => a - b)
}

/**
 * Visit every element of a strided layout in row-major order of `shape`, calling `body(offset, k)` with the element's
 * position in `data` and its row-major index k.
 */
export function forEachOffset(
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  body: (offset: number, k: number) => void,
): void {
  const n = sizeOf(shape)
  if (n === 0) return
  const rank = shape.length
  if (rank === 0) {
    body(offset, 0)
    return
  }
  // An odometer over all but the last axis; the last axis is a tight inner loop.
  const last = rank - 1
  const inner = shape[last]
  const step = strides[last]
  const index = new Array<number>(rank).fill(0)
  let base = offset
  for (let k = 0; k < n;) {
    let off = base
    for (let j = 0; j < inner; j++, k++, off += step) body(off, k)
    // Advance the odometer on axes last-1 … 0.
    let axis = last - 1
    while (axis >= 0) {
      index[axis]++
      base += strides[axis]
      if (index[axis] < shape[axis]) break
      base -= strides[axis] * shape[axis]
      index[axis] = 0
      axis--
    }
    if (axis < 0) break
  }
}

/**
 * Visit the elements of two strided layouts in lockstep over a shared `shape` (strides already broadcast), calling
 * `body(offsetA, offsetB, k)`.
 */
export function forEachOffset2(
  shape: readonly number[],
  stridesA: readonly number[],
  offsetA: number,
  stridesB: readonly number[],
  offsetB: number,
  body: (a: number, b: number, k: number) => void,
): void {
  const n = sizeOf(shape)
  if (n === 0) return
  const rank = shape.length
  if (rank === 0) {
    body(offsetA, offsetB, 0)
    return
  }
  const last = rank - 1
  const inner = shape[last]
  const stepA = stridesA[last]
  const stepB = stridesB[last]
  const index = new Array<number>(rank).fill(0)
  let baseA = offsetA
  let baseB = offsetB
  for (let k = 0; k < n;) {
    let a = baseA
    let b = baseB
    for (let j = 0; j < inner; j++, k++, a += stepA, b += stepB) body(a, b, k)
    let axis = last - 1
    while (axis >= 0) {
      index[axis]++
      baseA += stridesA[axis]
      baseB += stridesB[axis]
      if (index[axis] < shape[axis]) break
      baseA -= stridesA[axis] * shape[axis]
      baseB -= stridesB[axis] * shape[axis]
      index[axis] = 0
      axis--
    }
    if (axis < 0) break
  }
}

/**
 * Copy a tensor's elements, in row-major order, into a new typed array of the given dtype. Complex128 storage is
 * interleaved (two slots per element); a real tensor converted to complex128 gets zero imaginary parts; conversion to
 * bool maps non-zero to 1. Complex to a real dtype is a `DTypeError`: take `realPart`, `imagPart` or `abs` instead.
 */
export function flatData(t: Tensor, dtype: DType = t.dtype): TensorData {
  const n = size(t)
  const src = t.data
  if (t.dtype === 'complex128') {
    if (dtype !== 'complex128')
      throw new DTypeError('astype', `astype: cannot convert complex128 to ${dtype}; take realPart, imagPart or abs`, [
        t.dtype,
        dtype,
      ])
    if (isContiguous(t)) return src.slice(2 * t.offset, 2 * (t.offset + n))
    const out = new Float64Array(2 * n)
    forEachOffset(t.shape, t.strides, t.offset, (off, k) => {
      out[2 * k] = src[2 * off]
      out[2 * k + 1] = src[2 * off + 1]
    })
    return out
  }
  if (isContiguous(t) && dtype === t.dtype) return src.slice(t.offset, t.offset + n)
  const out = allocate(dtype, n)
  if (dtype === 'complex128') forEachOffset(t.shape, t.strides, t.offset, (off, k) => (out[2 * k] = src[off]))
  else if (dtype === 'bool') forEachOffset(t.shape, t.strides, t.offset, (off, k) => (out[k] = src[off] !== 0 ? 1 : 0))
  else forEachOffset(t.shape, t.strides, t.offset, (off, k) => (out[k] = src[off]))
  return out
}

/** The elements in row-major order as a Float64Array; no copy (`readonlyData`) when the tensor is contiguous float64. */
export function float64Data(t: Tensor): Float64Array {
  if (t.dtype === 'complex128') throw new DTypeError('float64Data', 'float64Data: expected real values, got complex128')
  const view = t.dtype === 'float64' ? readonlyData(t) : null
  return view !== null ? (view as Float64Array) : (flatData(t, 'float64') as Float64Array)
}

/** The result dtype of combining two dtypes: the promotion table of `dtype.ts` (`promoteTypes`). */
export const promote = promoteTypes

/** The dtype a plain number takes next to a tensor of dtype `other`: NumPy's weak scalar rule (`weakType`). */
export const scalarDType = weakType

/** Format a shape for error messages. */
export function showShape(shape: readonly number[]): string {
  return `[${shape.join(', ')}]`
}
