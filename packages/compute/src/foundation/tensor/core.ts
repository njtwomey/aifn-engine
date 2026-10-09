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
 * worker boundary is re-branded with `revive`. Its type is `aifn-compute/foundation/contracts`' `TensorBrand`, so the
 * contract's `Tensor` and this one are the same type.
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

/**
 * Allocate zeroed storage for `size` elements of a dtype: a `Float64Array` for float64, one of $2 \cdot \text{size}$
 * doubles for complex128 (interleaved re, im), a `Float32Array`, a `Uint8Array` for bool, and an `Int32Array` for
 * int32.
 *
 * @param dtype The element type, which picks the typed array.
 * @param size The number of elements (not of storage slots: a complex128 element takes two).
 * @returns A new zero-filled typed array.
 */
export function allocate(dtype: DType, size: number): TensorData {
  if (dtype === 'float64') return new Float64Array(size)
  if (dtype === 'complex128') return new Float64Array(2 * size)
  if (dtype === 'float32') return new Float32Array(size)
  if (dtype === 'bool') return new Uint8Array(size)
  return new Int32Array(size)
}

/**
 * The dtype a typed array holds by default (a Float64Array is float64 unless complex128 is said explicitly).
 *
 * @param data A tensor's storage: a `Float64Array` (float64), `Float32Array` (float32), `Uint8Array` (bool) or
 *   `Int32Array` (int32, also the answer for anything else).
 * @returns The dtype of that storage.
 */
export function dtypeOf(data: TensorData): DType {
  if (data instanceof Float64Array) return 'float64'
  if (data instanceof Float32Array) return 'float32'
  if (data instanceof Uint8Array) return 'bool'
  return 'int32'
}

/**
 * Number of elements of a shape (1 for the scalar shape `[]`).
 *
 * @param shape The length of each axis.
 * @returns The product of the lengths.
 */
export function sizeOf(shape: readonly number[]): number {
  let n = 1
  for (const d of shape) n *= d
  return n
}

/**
 * Number of elements of a tensor: the product of its shape (1 for a scalar, and complex elements count once).
 *
 * @param t The tensor.
 * @returns How many elements it holds.
 *
 * @example Count the elements of a tensor
 * print('size of a 2 by 3 matrix =', size(zeros([2, 3])))
 * print('size of a scalar =', size(scalar(7)))
 */
export function size(t: Tensor): number {
  return sizeOf(t.shape)
}

/**
 * Row-major (C order) strides of a shape, in elements: the last axis has stride 1, and each axis before it steps over
 * a whole block of the axes after it.
 *
 * @param shape The length of each axis.
 * @returns One stride per axis: element $(i_0, i_1, \dots)$ is at offset $\sum_k i_k s_k$.
 *
 * @example The strides of a 2 by 3 by 4 array
 * print('strides =', rowMajorStrides([2, 3, 4]))
 */
export function rowMajorStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length)
  let step = 1
  for (let k = shape.length - 1; k >= 0; k--) {
    strides[k] = step
    step *= shape[k]
  }
  return strides
}

/**
 * Check that a shape is a list of non-negative integers; throws `ShapeError` if not.
 *
 * @param shape The shape to check.
 * @param where The caller's name, for the error message.
 */
export function checkShape(shape: readonly number[], where: string): void {
  for (const d of shape) {
    if (!Number.isInteger(d) || d < 0)
      throw new ShapeError(where, `${where}: invalid shape [${shape.join(', ')}]`, [shape])
  }
}

/**
 * Wrap a typed array as a tensor without copying. `shape` defaults to all of `data` as a vector; strides are
 * row-major. `dtype` defaults to the typed array's own (`dtypeOf`); pass `'complex128'` for a Float64Array of
 * interleaved (re, im) pairs, which holds `data.length / 2` elements. Throws `DTypeError` when the typed array cannot
 * hold `dtype`, and `ShapeError` when the values do not fill `shape` exactly.
 *
 * The caller hands over `data`: it must not be mutated afterwards, since tensors are immutable by convention.
 *
 * @param data The storage, in row-major order; it becomes the tensor's own (not copied).
 * @param shape The shape to give it. Left out, the tensor is a vector of every element in `data`.
 * @param dtype The element type; it must match the typed array, except that a `Float64Array` may be complex128.
 * @returns A contiguous tensor over `data`, with offset 0.
 *
 * @example Wrap a typed array as a matrix
 * const t = fromData(new Float64Array([1, 2, 3, 4, 5, 6]), [2, 3])
 * print('t =', t)
 * print('strides =', t.strides)
 *
 * @example Interleaved pairs as complex numbers
 * const z = fromData(new Float64Array([1, 2, 0, -1]), [2], 'complex128')
 * print('shape =', z.shape)
 * print('re =', realPart(z))
 * print('im =', imagPart(z))
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
 * views of a complex tensor's parts (`complexPartView`). Nothing is checked: the caller makes the header fit the data.
 *
 * @param t The tensor whose storage the view shares.
 * @param shape The view's shape (copied).
 * @param strides The view's per-axis strides, in storage elements of the view's dtype (copied).
 * @param offset The position in the storage of the view's first element, in the same units as `strides`.
 * @param dtype The view's dtype; `t`'s own when left out.
 * @returns A branded tensor over `t.data`.
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
 * strides doubled, offset $2o + \text{part}$ for the tensor's offset $o$ (design K §8.1). Zero-copy.
 *
 * @param z A complex128 tensor (not checked).
 * @param part 0 for the real parts, 1 for the imaginary parts.
 * @returns A float64 tensor with `z`'s shape, sharing its storage.
 *
 * @example Real and imaginary parts as views
 * const z = tensor([{ re: 1, im: 2 }, { re: 3, im: -1 }])
 * print('re =', complexPartView(z, 0))
 * print('im =', complexPartView(z, 1))
 * print('shares storage:', complexPartView(z, 0).data === z.data)
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

/**
 * True when a value is a tensor (made by `fromData` or a view of one): it carries the brand.
 *
 * @param x Any value.
 * @returns Whether `x` is a branded tensor; an object that merely has a tensor's fields is not.
 *
 * @example The brand, not the fields, makes a tensor
 * print('tensor:', isTensor(tensor([1, 2])))
 * print('look-alike:', isTensor({ shape: [2], strides: [1], offset: 0, dtype: 'float64', data: new Float64Array(2) }))
 * print('number:', isTensor(3))
 */
export function isTensor(x: unknown): x is Tensor {
  return typeof x === 'object' && x !== null && (x as { [TENSOR]?: unknown })[TENSOR] === true
}

/**
 * True for an unbranded object with a tensor's fields, as `structuredClone` or a worker message leaves one.
 *
 * @param x A non-null object; its `shape`, `strides`, `offset`, `dtype` and `data` fields are checked by type.
 * @returns Whether it has every field a tensor has, with `data` one of the four storage typed arrays.
 */
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
 *
 * @param x Any value: a tensor-shaped object, or arrays and plain objects that hold some. Not modified.
 * @returns `x` with every tensor-shaped object branded; arrays and plain objects on the way are new copies.
 *
 * @example A tensor after a structured clone
 * const sent = structuredClone({ w: tensor([1, 2, 3]) })
 * print('before:', isTensor(sent.w))
 * const back = revive(sent)
 * print('after:', isTensor(back.w), back.w)
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
 * True when the tensor's elements lie in row-major order in a single run of `data`, starting at `offset`. Axes of
 * length 1 are ignored, since their stride is never used.
 *
 * @param t The tensor; only its shape and strides are read.
 * @returns Whether its strides are the row-major strides of its shape.
 *
 * @example A transpose is a view with swapped strides
 * const a = tensor([[1, 2, 3], [4, 5, 6]])
 * print('a contiguous:', isContiguous(a))
 * print('transpose(a) contiguous:', isContiguous(transpose(a)))
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
 * A zero-copy, read-only view of a tensor's elements in row-major order (design-core §3.1 `readonly(t)`): a subarray
 * of its storage when the tensor is contiguous (any offset), else null, so a hot loop reads a tensor without the copy
 * `toFlat` makes and falls back to a copy itself when it gets null. complex128 views hold interleaved (re, im) pairs.
 * Tensors are immutable: never write to the result.
 *
 * @param t The tensor to read.
 * @returns Its storage itself when it covers exactly the tensor, a subarray of it when the tensor is a contiguous part,
 *   or null when the tensor is not contiguous.
 *
 * @example Contiguous tensors are read in place
 * const a = tensor([[1, 2, 3], [4, 5, 6]])
 * print('a:', readonlyData(a))
 * print('transpose(a):', readonlyData(transpose(a)))
 */
export function readonlyData(t: Tensor): TensorData | null {
  if (!isContiguous(t)) return null
  const w = t.dtype === 'complex128' ? 2 : 1
  const size = t.shape.reduce((a, b) => a * b, 1)
  if (t.offset === 0 && t.data.length === size * w) return t.data
  return t.data.subarray(t.offset * w, (t.offset + size) * w) as TensorData
}

/**
 * Normalise an axis (negative counts from the end) and check its range; throws `ShapeError` when it is not an integer
 * in $[-\text{rank}, \text{rank})$.
 *
 * @param axis The axis as given: $0$ to $\text{rank} - 1$, or $-1$ for the last and so on.
 * @param rank The number of axes of the tensor it indexes.
 * @param where The caller's name, for the error message.
 * @returns The axis as a non-negative index.
 */
export function normaliseAxis(axis: number, rank: number, where: string): number {
  const a = axis < 0 ? axis + rank : axis
  if (!Number.isInteger(a) || a < 0 || a >= rank)
    throw new ShapeError(where, `${where}: axis ${axis} is out of range for rank ${rank}`)
  return a
}

/**
 * Normalise one or several axes; `undefined` or `null` means every axis. Duplicates are an error (`ShapeError`), as is
 * an axis out of range.
 *
 * @param axis One axis, a list of axes (negative counts from the end), or `undefined` or `null` for all of them.
 * @param rank The number of axes of the tensor they index.
 * @param where The caller's name, for error messages.
 * @returns The axes as non-negative indices, sorted ascending.
 */
export function normaliseAxes(axis: Axes | null | undefined, rank: number, where: string): number[] {
  if (axis === undefined || axis === null) return Array.from({ length: rank }, (_, k) => k)
  const list = typeof axis === 'number' ? [axis] : [...axis]
  const out = list.map((a) => normaliseAxis(a, rank, where))
  if (new Set(out).size !== out.length) throw new ShapeError(where, `${where}: repeated axis in [${list.join(', ')}]`)
  return out.sort((a, b) => a - b)
}

/**
 * Visit every element of a strided layout in row-major order of `shape`, calling `body(offset, k)` with the element's
 * position in `data` and its row-major index $k$.
 *
 * @param shape The shape to iterate over; nothing is visited when it has no elements, and one element when it is `[]`.
 * @param strides The stride of each axis, in elements of `data`.
 * @param offset The position in `data` of the first element.
 * @param body Called once per element with its position in `data` and its row-major index $k$ (0, 1, 2, ...).
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
 *
 * @param shape The shared shape to iterate over, in row-major order.
 * @param stridesA The strides of the first layout over `shape` (0 on a broadcast axis).
 * @param offsetA The position of the first layout's first element in its storage.
 * @param stridesB The strides of the second layout over `shape` (0 on a broadcast axis).
 * @param offsetB The position of the second layout's first element in its storage.
 * @param body Called once per element with its position in each layout and its row-major index $k$.
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
 *
 * @param t The tensor to copy (any strides and offset).
 * @param dtype The dtype of the copy; `t`'s own when left out. Conversion to int32 truncates as an `Int32Array` does.
 * @returns A new typed array of `size(t)` elements (twice that many slots for complex128), never `t`'s storage.
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

/**
 * The elements in row-major order as a Float64Array; no copy (`readonlyData`) when the tensor is contiguous float64.
 * Other real dtypes are converted; complex128 is a `DTypeError`.
 *
 * @param t A real tensor.
 * @returns Its elements as float64: a view of its storage when possible (do not write to it), else a copy.
 *
 * @example Read any real tensor as float64
 * print('float64:', float64Data(tensor([1.5, 2.5])))
 * print('from int32:', float64Data(tensor([1, 2, 3], undefined, 'int32')))
 */
export function float64Data(t: Tensor): Float64Array {
  if (t.dtype === 'complex128') throw new DTypeError('float64Data', 'float64Data: expected real values, got complex128')
  const view = t.dtype === 'float64' ? readonlyData(t) : null
  return view !== null ? (view as Float64Array) : (flatData(t, 'float64') as Float64Array)
}

/** The result dtype of combining two dtypes: the promotion table of `dtype.ts` (`promoteTypes`). */
export const promote = promoteTypes

/** The dtype a plain number takes next to a tensor of dtype `other`: NumPy's weak scalar rule (`weakType`). */
export const scalarDType = weakType

/**
 * Format a shape for error messages, as `[2, 3]`.
 *
 * @param shape The shape to format.
 * @returns The lengths, comma-separated, in square brackets.
 *
 * @example A shape as text
 * print(showShape([2, 3]))
 * print(showShape([]))
 */
export function showShape(shape: readonly number[]): string {
  return `[${shape.join(', ')}]`
}
