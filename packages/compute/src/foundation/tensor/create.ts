/**
 * Constructors and converters between tensors and plain arrays (the chart boundary).
 *
 * Complex values. `tensor` accepts `{ re, im }` leaves (or numbers with `dtype: 'complex128'`); `full`, `zeros`,
 * `ones` and `eye` take a dtype. On the way out, the number-typed converters give a complex element as the pair
 * (re, im): `toFlat` interleaves them (`[re₀, im₀, re₁, im₁, …]`, twice as many entries as elements), `toArray` adds
 * a trailing axis of length 2 (`[[re, im], …]`), `toRows` interleaves each row. `toComplexFlat` and `toComplexArray`
 * give `{ re, im }` objects instead, and `complexItem` the single element. `item` of a complex tensor is an error.
 */

import type { ComplexNumber } from 'aifn-compute/foundation/contracts'
import { AifnError, DTypeError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  allocate,
  checkShape,
  flatData,
  forEachOffset,
  fromData,
  isTensor,
  showShape,
  size,
  sizeOf,
  type DType,
  type NestedArray,
  type Tensor,
  type TensorData,
} from './core'

/** Nested arrays of complex numbers (`{ re, im }` leaves), as `tensor` accepts and `toComplexArray` returns. */
export type NestedComplex = ComplexNumber | NestedComplex[]

/** True for a `{ re, im }` leaf. */
function isComplexLeaf(x: unknown): x is ComplexNumber {
  return typeof x === 'object' && x !== null && !('length' in x) && typeof (x as ComplexNumber).re === 'number'
}

/**
 * The shape of nested arrays (checked to be rectangular) or of a tensor. A number (or a `{ re, im }` leaf) has shape
 * `[]`.
 *
 * @example shapeOf([[1, 2, 3], [4, 5, 6]]) // [2, 3]
 */
export function shapeOf(x: NestedArray | NestedComplex | ArrayLike<number> | Tensor): number[] {
  if (isTensor(x)) return [...x.shape]
  if (typeof x === 'number' || isComplexLeaf(x)) return []
  const shape: number[] = []
  let probe: unknown = x
  while (typeof probe === 'object' && probe !== null && 'length' in probe) {
    const list = probe as ArrayLike<unknown>
    shape.push(list.length)
    if (list.length === 0) break
    probe = list[0]
  }
  // Check that every branch has the same shape, so that ragged input is an error rather than silently truncated.
  const check = (node: unknown, depth: number): void => {
    if (depth === shape.length) {
      if (typeof node !== 'number' && !isComplexLeaf(node))
        throw new ShapeError('shapeOf', 'shapeOf: ragged or non-numeric nested array')
      return
    }
    const list = node as ArrayLike<unknown>
    if (typeof node !== 'object' || node === null || list.length !== shape[depth]) {
      throw new ShapeError(
        'shapeOf',
        `shapeOf: ragged nested array (expected length ${shape[depth]} at depth ${depth})`,
      )
    }
    for (let i = 0; i < list.length; i++) check(list[i], depth + 1)
  }
  check(x, 0)
  return shape
}

/** Does a nested value contain a `{ re, im }` leaf? (Checks the first leaf only; `tensor` checks the rest.) */
function hasComplexLeaf(x: unknown): boolean {
  let probe = x
  while (typeof probe === 'object' && probe !== null && 'length' in probe && (probe as ArrayLike<unknown>).length > 0)
    probe = (probe as ArrayLike<unknown>)[0]
  return isComplexLeaf(probe)
}

/** Storage filled with one value (a number, or `{ re, im }` for complex128). */
function filled(dtype: DType, n: number, value: number | ComplexNumber): TensorData {
  const out = allocate(dtype, n)
  if (dtype === 'complex128') {
    const [re, im] = typeof value === 'number' ? [value, 0] : [value.re, value.im]
    for (let k = 0; k < n; k++) {
      out[2 * k] = re
      out[2 * k + 1] = im
    }
    return out
  }
  if (typeof value !== 'number')
    throw new DTypeError('tensor', `tensor: a complex value needs dtype complex128`, [dtype])
  return out.fill(dtype === 'bool' ? (value !== 0 ? 1 : 0) : value)
}

/**
 * A tensor from nested arrays, a flat array (with `shape`) or a typed array. Always copies.
 *
 * @param values nested `number[]…` or `{ re, im }` leaves (which make it complex128), or a flat `ArrayLike<number>`
 *   read in row-major order when `shape` is given
 * @param shape the shape to give flat values; defaults to the nested structure
 * @param dtype storage type, default `float64` (`complex128` for `{ re, im }` leaves); values are converted as a
 *   typed array converts them (int32 truncates, bool maps non-zero to 1, complex128 takes numbers as real)
 * @example tensor([[1, 2], [3, 4]]) // shape [2, 2]
 * @example tensor([1, 2, 3, 4, 5, 6], [2, 3])
 * @example tensor([{ re: 1, im: 2 }, { re: 0, im: -1 }]) // complex128, shape [2]
 */
export function tensor(
  values: NestedArray | NestedComplex | ArrayLike<number>,
  shape?: readonly number[],
  dtype?: DType,
): Tensor {
  const type: DType = dtype ?? (hasComplexLeaf(values) ? 'complex128' : 'float64')
  if (typeof values === 'number' || isComplexLeaf(values)) {
    if (shape && sizeOf(shape) !== 1)
      throw new ShapeError('tensor', `tensor: one value cannot fill shape ${showShape(shape)}`)
    return fromData(filled(type, 1, values), shape ?? [], type)
  }
  const nested = shapeOf(values as NestedArray)
  const n = sizeOf(nested)
  const flat = allocate(type, n)
  const complex = type === 'complex128'
  let k = 0
  const fill = (node: unknown): void => {
    if (typeof node === 'number') {
      if (complex) {
        flat[2 * k] = node
        flat[2 * k++ + 1] = 0
      } else flat[k++] = type === 'bool' ? (node !== 0 ? 1 : 0) : node
    } else if (isComplexLeaf(node)) {
      if (!complex)
        throw new DTypeError('tensor', `tensor: a complex value needs dtype complex128, not ${type}`, [type])
      flat[2 * k] = node.re
      flat[2 * k++ + 1] = node.im
    } else for (let i = 0; i < (node as ArrayLike<unknown>).length; i++) fill((node as ArrayLike<unknown>)[i])
  }
  fill(values)
  if (shape) {
    checkShape(shape, 'tensor')
    if (sizeOf(shape) !== n) {
      throw new ShapeError('tensor', `tensor: ${n} values do not fill shape ${showShape(shape)}`)
    }
    return fromData(flat, shape, type)
  }
  return fromData(flat, nested, type)
}

/** A scalar (rank-0) tensor; a `{ re, im }` value gives complex128. */
export function scalar(value: number | ComplexNumber, dtype?: DType): Tensor {
  return tensor(value, [], dtype)
}

/** A tensor of the given shape filled with `value` (a number, or `{ re, im }` for complex128). */
export function full(shape: readonly number[], value: number | ComplexNumber, dtype?: DType): Tensor {
  checkShape(shape, 'full')
  const type: DType = dtype ?? (typeof value === 'number' ? 'float64' : 'complex128')
  return fromData(filled(type, sizeOf(shape), value), shape, type)
}

/** A tensor of zeros. */
export function zeros(shape: readonly number[], dtype: DType = 'float64'): Tensor {
  return full(shape, 0, dtype)
}

/** A tensor of ones. */
export function ones(shape: readonly number[], dtype: DType = 'float64'): Tensor {
  return full(shape, 1, dtype)
}

/** The n×m identity-like matrix: ones on the diagonal `k` (0 main, positive above, negative below). m defaults to n. */
export function eye(n: number, m: number = n, k = 0, dtype: DType = 'float64'): Tensor {
  const data = allocate(dtype, n * m)
  const width = dtype === 'complex128' ? 2 : 1
  for (let i = 0; i < n; i++) {
    const j = i + k
    if (j >= 0 && j < m) data[width * (i * m + j)] = 1
  }
  return fromData(data, [n, m], dtype)
}

/**
 * Evenly spaced values in [start, stop) with the given step, as `np.arange`: `arange(5)` is 0…4 and
 * `arange(2, 3, 0.25)` is 2, 2.25, 2.5, 2.75. The length is ⌈(stop − start) / step⌉.
 */
export function arange(start: number, stop?: number, step = 1, dtype: DType = 'float64'): Tensor {
  if (stop === undefined) {
    stop = start
    start = 0
  }
  if (step === 0 || !Number.isFinite(step)) throw new AifnError('arange', 'arange: step must be finite and non-zero')
  const n = Math.max(0, Math.ceil((stop - start) / step))
  const data = allocate(dtype === 'complex128' ? 'float64' : dtype, n)
  for (let i = 0; i < n; i++) data[i] = start + i * step
  const t = fromData(data)
  return dtype === 'complex128' ? copy(t, dtype) : t
}

/**
 * `num` evenly spaced values from `start` to `stop`, as `np.linspace`. With `endpoint` (default) the last value is
 * exactly `stop`; without it the values stop one step short.
 */
export function linspace(start: number, stop: number, num = 50, endpoint = true): Tensor {
  if (!Number.isInteger(num) || num < 0) throw new AifnError('linspace', 'linspace: num must be a non-negative integer')
  const data = new Float64Array(num)
  const div = endpoint ? num - 1 : num
  const step = div > 0 ? (stop - start) / div : 0
  for (let i = 0; i < num; i++) data[i] = start + i * step
  if (endpoint && num > 1) data[num - 1] = stop
  return fromData(data)
}

/** A matrix from rows; every row must have the same length. */
export function fromRows(rows: readonly (readonly number[] | ArrayLike<number>)[], dtype: DType = 'float64'): Tensor {
  if (dtype === 'complex128') return copy(fromRows(rows), dtype)
  const m = rows.length
  const n = m > 0 ? rows[0].length : 0
  const data = allocate(dtype, m * n)
  for (let i = 0; i < m; i++) {
    const row = rows[i]
    if (row.length !== n) throw new ShapeError('fromRows', `fromRows: row ${i} has length ${row.length}, expected ${n}`)
    for (let j = 0; j < n; j++) data[i * n + j] = row[j]
  }
  return fromData(data, [m, n])
}

/** The elements in row-major order as a plain array; complex128 is interleaved (re₀, im₀, re₁, im₁, …). */
export function toFlat(t: Tensor): number[] {
  return Array.from(flatData(t))
}

/** Nested arrays of a shape from row-major values. */
function nest<L>(shape: readonly number[], leaf: (k: number) => L): L | L[] {
  if (shape.length === 0) return leaf(0)
  let k = 0
  type Nested = L | Nested[]
  const build = (axis: number): Nested[] => {
    const n = shape[axis]
    const out: Nested[] = new Array<Nested>(n)
    for (let i = 0; i < n; i++) out[i] = axis === shape.length - 1 ? leaf(k++) : build(axis + 1)
    return out
  }
  return build(0) as L[]
}

/**
 * The tensor as nested plain arrays (a number for a scalar tensor). A complex128 tensor gains a trailing axis of
 * length 2: each element is the pair `[re, im]`.
 */
export function toArray(t: Tensor): NestedArray {
  const flat = flatData(t)
  if (t.dtype === 'complex128') return nest([...t.shape, 2], (k) => flat[k]) as NestedArray
  return nest(t.shape, (k) => flat[k]) as NestedArray
}

/** The elements of a tensor as complex numbers `{ re, im }` in row-major order (a real tensor has im = 0). */
export function toComplexFlat(t: Tensor): ComplexNumber[] {
  const flat = flatData(t, 'complex128')
  return Array.from({ length: size(t) }, (_, k) => ({ re: flat[2 * k], im: flat[2 * k + 1] }))
}

/** The tensor as nested arrays of `{ re, im }` (one object for a scalar tensor; a real tensor has im = 0). */
export function toComplexArray(t: Tensor): NestedComplex {
  const flat = flatData(t, 'complex128')
  return nest(t.shape, (k) => ({ re: flat[2 * k], im: flat[2 * k + 1] })) as NestedComplex
}

/** A matrix as `number[][]` rows (a vector becomes one row); complex128 rows are interleaved (re, im) pairs. */
export function toRows(t: Tensor): number[][] {
  if (t.shape.length === 1) return [toFlat(t)]
  if (t.shape.length !== 2) throw new ShapeError('toRows', `toRows: expected a matrix, got shape ${showShape(t.shape)}`)
  const [m, n0] = t.shape
  const n = t.dtype === 'complex128' ? 2 * n0 : n0
  const flat = flatData(t)
  return Array.from({ length: m }, (_, i) => Array.from(flat.subarray(i * n, (i + 1) * n)))
}

/** The single element of a real tensor of size 1 (any rank). For complex128 use `complexItem`. */
export function item(t: Tensor): number {
  if (size(t) !== 1)
    throw new ShapeError('item', `item: tensor of shape ${showShape(t.shape)} has ${size(t)} elements, not 1`)
  if (t.dtype === 'complex128')
    throw new DTypeError('item', 'item: a complex128 element is not a number; use complexItem', [t.dtype])
  let value = 0
  forEachOffset(t.shape, t.strides, t.offset, (off) => {
    value = t.data[off]
  })
  return value
}

/** The single element of a tensor of size 1 as `{ re, im }` (im = 0 for a real tensor). */
export function complexItem(t: Tensor): ComplexNumber {
  if (size(t) !== 1)
    throw new ShapeError('complexItem', `complexItem: tensor of shape ${showShape(t.shape)} has ${size(t)} elements`)
  return toComplexFlat(t)[0]
}

/** A contiguous row-major copy of a tensor, optionally converted to another dtype (see `flatData` for complex). */
export function copy(t: Tensor, dtype: DType = t.dtype): Tensor {
  return fromData(flatData(t, dtype), t.shape, dtype)
}

/** The tensor converted to another dtype (a copy; int32 truncates towards zero). */
export function astype(t: Tensor, dtype: DType): Tensor {
  return copy(t, dtype)
}
