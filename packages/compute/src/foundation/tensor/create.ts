/**
 * Constructors and converters between tensors and plain arrays (the chart boundary).
 *
 * Complex values. `tensor` accepts `{ re, im }` leaves (or numbers with `dtype: 'complex128'`); `full`, `zeros`,
 * `ones` and `eye` take a dtype. On the way out, the number-typed converters give a complex element as the pair
 * (re, im): `toFlat` interleaves them ($[r_0, i_0, r_1, i_1, \dots]$, twice as many entries as elements), `toArray`
 * adds a trailing axis of length 2 (`[[re, im], …]`), `toRows` interleaves each row. `toComplexFlat` and
 * `toComplexArray` give `{ re, im }` objects instead, and `complexItem` the single element. `item` of a complex tensor
 * is an error.
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

/**
 * True for a `{ re, im }` leaf.
 *
 * @param x Any value: a leaf is a non-null object with no `length` and a numeric `re`.
 * @returns Whether `x` is a complex number leaf.
 */
function isComplexLeaf(x: unknown): x is ComplexNumber {
  return typeof x === 'object' && x !== null && !('length' in x) && typeof (x as ComplexNumber).re === 'number'
}

/**
 * The shape of nested arrays (checked to be rectangular) or of a tensor. A number (or a `{ re, im }` leaf) has shape
 * `[]`. Ragged or non-numeric nesting is a `ShapeError`.
 *
 * @param x Nested arrays of numbers or `{ re, im }` leaves, a flat array-like, a single number or leaf, or a tensor.
 * @returns The length of each level of nesting (a copy of the shape, for a tensor). An empty array stops the descent.
 *
 * @example The shape of nested arrays
 * print('matrix:', shapeOf([[1, 2, 3], [4, 5, 6]]))
 * print('number:', shapeOf(5))
 * print('tensor:', shapeOf(zeros([4, 2])))
 *
 * @example Ragged input is refused
 * try {
 *   shapeOf([[1, 2], [3]])
 * } catch (e) {
 *   print('error:', e.message)
 * }
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

/**
 * Does a nested value contain a `{ re, im }` leaf? (Checks the first leaf only; `tensor` checks the rest.)
 *
 * @param x A number, a leaf, or nested arrays; the descent follows the first element of each level.
 * @returns Whether the first leaf reached is a complex number (false when an empty array is met first).
 */
function hasComplexLeaf(x: unknown): boolean {
  let probe = x
  while (typeof probe === 'object' && probe !== null && 'length' in probe && (probe as ArrayLike<unknown>).length > 0)
    probe = (probe as ArrayLike<unknown>)[0]
  return isComplexLeaf(probe)
}

/**
 * Storage filled with one value (a number, or `{ re, im }` for complex128). A `{ re, im }` value for a real dtype is a
 * `DTypeError`.
 *
 * @param dtype The element type of the storage.
 * @param n The number of elements.
 * @param value The fill: a number (taken as real for complex128, and as 0 or 1 for bool) or a complex leaf.
 * @returns A new typed array of `n` elements, every one equal to `value`.
 */
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
 * A tensor from nested arrays, a flat array (with `shape`) or a typed array. Always copies. Ragged nesting, a `shape`
 * the values do not fill, and a `{ re, im }` leaf with a real `dtype` are errors.
 *
 * @param values Nested `number[]…` or `{ re, im }` leaves (which make it complex128), a single number or leaf (a
 *   scalar), or a flat `ArrayLike<number>` read in row-major order when `shape` is given.
 * @param shape The shape to give the values, which must hold exactly as many elements; defaults to the nested
 *   structure.
 * @param dtype Storage type, default `float64` (`complex128` for `{ re, im }` leaves); values are converted as a
 *   typed array converts them (int32 truncates, bool maps non-zero to 1, complex128 takes numbers as real).
 * @returns A new contiguous tensor.
 *
 * @example From nested arrays
 * const a = tensor([[1, 2], [3, 4]])
 * print('a =', a)
 * print('shape =', a.shape, 'dtype =', a.dtype)
 *
 * @example From a flat array and a shape
 * print(tensor([1, 2, 3, 4, 5, 6], [2, 3]))
 *
 * @example Complex leaves make a complex128 tensor
 * const z = tensor([{ re: 1, im: 2 }, { re: 0, im: -1 }])
 * print('dtype =', z.dtype, 'shape =', z.shape)
 * print('re =', realPart(z), 'im =', imagPart(z))
 *
 * @example Converted on the way in
 * print('int32:', tensor([1.7, -1.7, 2], undefined, 'int32'))
 * print('bool:', tensor([0, 2, -1], undefined, 'bool'))
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

/**
 * A scalar (rank-0) tensor; a `{ re, im }` value gives complex128.
 *
 * @param value The element: a number, or a `{ re, im }` complex number.
 * @param dtype The storage type; float64 for a number and complex128 for a complex value when left out.
 * @returns A tensor of shape `[]`.
 *
 * @example A rank-0 tensor
 * const s = scalar(3)
 * print('shape =', s.shape)
 * print('value =', item(s))
 */
export function scalar(value: number | ComplexNumber, dtype?: DType): Tensor {
  return tensor(value, [], dtype)
}

/**
 * A tensor of the given shape filled with `value` (a number, or `{ re, im }` for complex128).
 *
 * @param shape The shape, a list of non-negative integers (checked).
 * @param value The fill: a number, or a `{ re, im }` complex number (which needs complex128).
 * @param dtype The storage type; float64 for a number and complex128 for a complex value when left out.
 * @returns A new tensor of that shape with every element equal to `value`.
 *
 * @example A constant matrix
 * print(full([2, 3], 7))
 *
 * @example A constant complex vector
 * const z = full([3], { re: 1, im: -1 })
 * print('dtype =', z.dtype, 'im =', imagPart(z))
 */
export function full(shape: readonly number[], value: number | ComplexNumber, dtype?: DType): Tensor {
  checkShape(shape, 'full')
  const type: DType = dtype ?? (typeof value === 'number' ? 'float64' : 'complex128')
  return fromData(filled(type, sizeOf(shape), value), shape, type)
}

/**
 * A tensor of zeros.
 *
 * @param shape The shape, a list of non-negative integers.
 * @param dtype The storage type.
 * @returns A new tensor of that shape filled with 0.
 *
 * @example Zeros of a shape
 * print(zeros([2, 3]))
 * print('dtype:', zeros([2], 'int32').dtype)
 */
export function zeros(shape: readonly number[], dtype: DType = 'float64'): Tensor {
  return full(shape, 0, dtype)
}

/**
 * A tensor of ones.
 *
 * @param shape The shape, a list of non-negative integers.
 * @param dtype The storage type (complex128 gives $1 + 0i$).
 * @returns A new tensor of that shape filled with 1.
 *
 * @example Ones of a shape
 * print(ones([2, 2]))
 */
export function ones(shape: readonly number[], dtype: DType = 'float64'): Tensor {
  return full(shape, 1, dtype)
}

/**
 * The $n \times m$ identity-like matrix: ones on the diagonal `k` (0 main, positive above, negative below), zeros
 * elsewhere.
 *
 * @param n The number of rows.
 * @param m The number of columns; $n$ (a square matrix) when left out.
 * @param k Which diagonal holds the ones: entry $(i, i + k)$ is 1 wherever it lies inside the matrix.
 * @param dtype The storage type.
 * @returns A new $n \times m$ matrix.
 *
 * @example The identity
 * print(eye(3))
 *
 * @example A rectangular matrix with ones above the diagonal
 * print(eye(2, 3, 1))
 */
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
 * Evenly spaced values in $[\text{start}, \text{stop})$ with the given step, as `np.arange`: `arange(5)` is 0 to 4
 * and `arange(2, 3, 0.25)` is 2, 2.25, 2.5, 2.75. The length is $\lceil (\text{stop} - \text{start}) /
 * \text{step} \rceil$ (0 when that is negative); value $i$ is $\text{start} + i \cdot \text{step}$. A step that is
 * zero or not finite is an error.
 *
 * @param start The first value; with `stop` left out, it is the end instead and the values start at 0.
 * @param stop The end, not included.
 * @param step The spacing; negative counts down.
 * @param dtype The storage type; int32 truncates each value as an `Int32Array` does.
 * @returns A new vector.
 *
 * @example Integers and fractional steps
 * print('arange(5) =', arange(5))
 * print('arange(2, 3, 0.25) =', arange(2, 3, 0.25))
 * print('arange(3, 0, -1) =', arange(3, 0, -1))
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
 * exactly `stop`; without it the values stop one step short. A `num` that is not a non-negative integer is an error.
 *
 * @param start The first value.
 * @param stop The last value with `endpoint`, or the value one step past the last without it.
 * @param num The number of values (one value is just `start`).
 * @param endpoint Whether `stop` is included: the step is $(\text{stop} - \text{start}) / (\text{num} - 1)$ with it,
 *   and $(\text{stop} - \text{start}) / \text{num}$ without.
 * @returns A new float64 vector of `num` values.
 *
 * @example With and without the endpoint
 * print('endpoint:', linspace(0, 1, 5))
 * print('no endpoint:', linspace(0, 1, 5, false))
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

/**
 * A matrix from rows; every row must have the same length (`ShapeError` otherwise).
 *
 * @param rows The rows, as arrays or typed arrays of numbers; no rows gives a $0 \times 0$ matrix.
 * @param dtype The storage type; values are converted as the typed array of that dtype converts them.
 * @returns A new matrix with one row per entry of `rows`.
 *
 * @example A matrix from rows
 * print(fromRows([[1, 2, 3], new Float64Array([4, 5, 6])]))
 */
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

/**
 * The elements in row-major order as a plain array; complex128 is interleaved ($r_0, i_0, r_1, i_1, \dots$).
 *
 * @param t The tensor, of any shape, strides and dtype.
 * @returns A new array of `size(t)` numbers (twice that for complex128).
 *
 * @example Flatten a matrix and a complex vector
 * print('real:', toFlat(tensor([[1, 2], [3, 4]])))
 * print('complex:', toFlat(tensor([{ re: 1, im: -1 }, { re: 2, im: -2 }])))
 */
export function toFlat(t: Tensor): number[] {
  return Array.from(flatData(t))
}

/**
 * Nested arrays of a shape from row-major values.
 *
 * @param shape The nesting to build: one level per axis.
 * @param leaf Gives the value at row-major index $k$; called once per element, in order.
 * @returns The value of `leaf(0)` for the shape `[]`, else nested arrays of the leaves.
 */
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
 *
 * @param t The tensor to convert.
 * @returns New nested arrays matching `t`'s shape.
 *
 * @example Nested arrays out
 * print('matrix:', toArray(tensor([[1, 2], [3, 4]])))
 * print('scalar:', toArray(scalar(5)))
 * print('complex:', toArray(tensor([{ re: 1, im: 2 }])))
 */
export function toArray(t: Tensor): NestedArray {
  const flat = flatData(t)
  if (t.dtype === 'complex128') return nest([...t.shape, 2], (k) => flat[k]) as NestedArray
  return nest(t.shape, (k) => flat[k]) as NestedArray
}

/**
 * The elements of a tensor as complex numbers `{ re, im }` in row-major order (a real tensor has `im` 0).
 *
 * @param t The tensor, real or complex.
 * @returns A new array of `size(t)` objects.
 *
 * @example Complex elements as objects
 * const [a, b] = toComplexFlat(tensor([{ re: 1, im: 2 }, { re: 3, im: -4 }]))
 * print('a: re =', a.re, 'im =', a.im)
 * print('b: re =', b.re, 'im =', b.im)
 */
export function toComplexFlat(t: Tensor): ComplexNumber[] {
  const flat = flatData(t, 'complex128')
  return Array.from({ length: size(t) }, (_, k) => ({ re: flat[2 * k], im: flat[2 * k + 1] }))
}

/**
 * The tensor as nested arrays of `{ re, im }` (one object for a scalar tensor; a real tensor has `im` 0).
 *
 * @param t The tensor, real or complex.
 * @returns New nested arrays matching `t`'s shape, with an object per element.
 *
 * @example A complex matrix as nested objects
 * const rows = toComplexArray(tensor([[{ re: 1, im: 0 }, { re: 0, im: 1 }]]))
 * print('rows:', rows.length, 'columns:', rows[0].length)
 * print('entry (0, 1): re =', rows[0][1].re, 'im =', rows[0][1].im)
 */
export function toComplexArray(t: Tensor): NestedComplex {
  const flat = flatData(t, 'complex128')
  return nest(t.shape, (k) => ({ re: flat[2 * k], im: flat[2 * k + 1] })) as NestedComplex
}

/**
 * A matrix as `number[][]` rows (a vector becomes one row); complex128 rows are interleaved (re, im) pairs. Any other
 * rank is a `ShapeError`.
 *
 * @param t A matrix or a vector.
 * @returns New arrays, one per row.
 *
 * @example Rows of a matrix, and of a vector
 * print('matrix:', toRows(tensor([[1, 2], [3, 4]])))
 * print('vector:', toRows(tensor([1, 2, 3])))
 */
export function toRows(t: Tensor): number[][] {
  if (t.shape.length === 1) return [toFlat(t)]
  if (t.shape.length !== 2) throw new ShapeError('toRows', `toRows: expected a matrix, got shape ${showShape(t.shape)}`)
  const [m, n0] = t.shape
  const n = t.dtype === 'complex128' ? 2 * n0 : n0
  const flat = flatData(t)
  return Array.from({ length: m }, (_, i) => Array.from(flat.subarray(i * n, (i + 1) * n)))
}

/**
 * The single element of a real tensor of size 1 (any rank). For complex128 use `complexItem`. A tensor of any other
 * size is a `ShapeError`, and a complex one a `DTypeError`.
 *
 * @param t A real tensor with exactly one element.
 * @returns That element as a number.
 *
 * @example The value of a one-element tensor
 * print(item(scalar(3.5)))
 * print(item(tensor([[7]])))
 */
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

/**
 * The single element of a tensor of size 1 as `{ re, im }` (`im` 0 for a real tensor); other sizes are a
 * `ShapeError`.
 *
 * @param t A tensor with exactly one element, real or complex.
 * @returns That element as a complex number.
 *
 * @example The value of a one-element complex tensor
 * const z = complexItem(scalar({ re: 1, im: -2 }))
 * print('re =', z.re, 'im =', z.im)
 */
export function complexItem(t: Tensor): ComplexNumber {
  if (size(t) !== 1)
    throw new ShapeError('complexItem', `complexItem: tensor of shape ${showShape(t.shape)} has ${size(t)} elements`)
  return toComplexFlat(t)[0]
}

/**
 * A contiguous row-major copy of a tensor, optionally converted to another dtype (see `flatData` for complex).
 *
 * @param t The tensor to copy (any strides and offset).
 * @param dtype The dtype of the copy; `t`'s own when left out. Complex to a real dtype is a `DTypeError`.
 * @returns A new contiguous tensor with `t`'s shape and its own storage.
 *
 * @example A transpose copied into row-major order
 * const v = transpose(tensor([[1, 2], [3, 4]]))
 * const c = copy(v)
 * print('copy =', c)
 * print('strides: view', v.strides, 'copy', c.strides)
 */
export function copy(t: Tensor, dtype: DType = t.dtype): Tensor {
  return fromData(flatData(t, dtype), t.shape, dtype)
}

/**
 * The tensor converted to another dtype (a copy; int32 truncates towards zero).
 *
 * @param t The tensor to convert.
 * @param dtype The target dtype: bool maps non-zero to 1, complex128 adds zero imaginary parts, and complex128 to a
 *   real dtype is a `DTypeError`.
 * @returns A new contiguous tensor of that dtype.
 *
 * @example Truncation to int32 and conversion to bool
 * const x = tensor([1.7, -1.7, 0])
 * print('int32:', astype(x, 'int32'))
 * print('bool:', astype(x, 'bool'))
 */
export function astype(t: Tensor, dtype: DType): Tensor {
  return copy(t, dtype)
}
