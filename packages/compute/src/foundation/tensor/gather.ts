/**
 * Indexed reads and writes as primitives: `gather` (read flat indices), `scatterAdd` (add into flat indices) and
 * `take` (rows along the first axis, e.g. an embedding lookup). `gather` and `scatterAdd` are each other's adjoint, so
 * both are differentiable to any order.
 */

import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import { fromData, isContiguous, isTensor, sizeOf, type Tensor } from './core'
import { joinComplex, splitComplex } from './kernels'
import { toFlat } from './create'
import { batchToFront, definePrimitive, type Op } from './primitive'
import { shapeOfValue } from './structure'
import type { Value } from './trace'

/**
 * Row-major float64 data of a raw value (no copy when already contiguous float64 at offset 0). A number throws
 * `AifnError`.
 *
 * @param x The raw value, which must be a tensor.
 * @param where The caller's name, for the error message.
 * @returns The elements in row-major order: the tensor's own storage when it is contiguous float64 at offset 0 (so
 *   not to be modified), otherwise a copy.
 */
function f64(x: number | Tensor, where: string): Float64Array {
  if (typeof x === 'number') throw new AifnError(where, `${where}: expected a tensor, got a number`)
  if (x.dtype === 'float64' && isContiguous(x) && x.offset === 0 && x.data.length === sizeOf(x.shape)) {
    return x.data as Float64Array
  }
  return Float64Array.from(toFlat(x))
}

/**
 * The parameters of a gather or scatter: `indices`, the flat (row-major) index into the source of each element of the
 * gathered tensor; `shape`, the gathered tensor's shape (with `indices.length` elements); and `source`, the shape of
 * the tensor indexed.
 */
type GatherParams = { indices: Int32Array; shape: readonly number[]; source: readonly number[] }

/**
 * The dtype of a gather's or scatter's result: complex128 for a complex input, float64 otherwise.
 *
 * @param dtype The input's dtype.
 * @returns `'complex128'` or `'float64'`.
 */
const floatOrComplex = (dtype: string) => (dtype === 'complex128' ? 'complex128' : 'float64')

/**
 * Apply a real kernel to a raw value, or to the real and imaginary views of a complex one and join the results.
 *
 * @param x The raw value: a number or a tensor of any dtype.
 * @param kernel The real kernel, applied to `x` itself or to each float64 part of a complex `x`.
 * @returns The kernel's result, or for a complex `x` the complex128 tensor joined from its results on the two parts.
 */
function onParts(x: number | Tensor, kernel: (part: number | Tensor) => Tensor): Tensor {
  if (typeof x === 'number' || x.dtype !== 'complex128') return kernel(x)
  const [re, im] = splitComplex(x)
  return joinComplex(kernel(re), kernel(im!))
}

/**
 * The forward rule of `gather`: `out[k] = data[indices[k]]`, with `data` the row-major elements of `x`.
 *
 * @param x The tensor read (a number throws `AifnError`).
 * @param options The gather's parameters (`source` is not read).
 * @param options.indices The flat index into `x` of each output element.
 * @param options.shape The shape of the result.
 * @returns A float64 tensor of shape `shape`.
 */
function gatherRaw(x: number | Tensor, { indices, shape }: GatherParams): Tensor {
  const data = f64(x, 'gather')
  const out = new Float64Array(indices.length)
  for (let k = 0; k < indices.length; k++) out[k] = data[indices[k]]
  return fromData(out, shape)
}

/**
 * The forward rule of `scatterAdd`: `out[indices[k]] += data[k]` into zeros, with `data` the row-major elements of
 * `g`.
 *
 * @param g The values added, one per index (a number throws `AifnError`).
 * @param options The scatter's parameters (`shape` is not read).
 * @param options.indices The flat index into the result of each value of `g`.
 * @param options.source The shape of the result.
 * @returns A float64 tensor of shape `source`, zero where no index points.
 */
function scatterAddRaw(g: number | Tensor, { indices, source }: GatherParams): Tensor {
  const data = f64(g, 'scatterAdd')
  const out = new Float64Array(sizeOf(source))
  for (let k = 0; k < indices.length; k++) out[indices[k]] += data[k]
  return fromData(out, source)
}

/**
 * The parameters of a gather or scatter over a batch of `size` examples stacked along a new first axis: the flat
 * indices of example $b$ are shifted by $b$ times the size of one example's source.
 *
 * @param p The parameters for one example.
 * @param size The number of examples in the batch.
 * @returns The parameters for the whole batch: the shifted indices, and `shape` and `source` with a leading axis of
 *   length `size`.
 */
function batched(p: GatherParams, size: number): GatherParams {
  const n = p.indices.length
  const stride = sizeOf(p.source)
  const indices = new Int32Array(size * n)
  for (let b = 0; b < size; b++) for (let k = 0; k < n; k++) indices[b * n + k] = p.indices[k] + b * stride
  return { indices, shape: [size, ...p.shape], source: [size, ...p.source] }
}

// gather and scatterAdd are linear and each other's transpose, so each has derivatives of every order.
/** The gather primitive: linear, with `scatterAddOp` as its transpose; batched by shifting each example's indices. */
const gatherOp: Op<GatherParams> = definePrimitive<GatherParams>({
  id: 'foundation/tensor/gather',
  arity: 1,
  impl: ([x], p) => onParts(x, (part) => gatherRaw(part, p)),
  linear: 'linear',
  dtype: 'float',
  transpose: (ct, _inputs, _which, p) => scatterAddOp([ct], p),
  shape: ([x], { shape }) => ({ shape: [...shape], dtype: floatOrComplex(x.dtype), number: false }),
  batch: ([x], [axis], p, size) => [gatherOp([batchToFront(x, axis ?? 0)], batched(p, size)), 0],
  doc: { summary: 'Read elements at flat indices.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 3])], params: { indices: Int32Array.of(0, 5, 5, 2), shape: [2, 2], source: [2, 3] } },
    ],
  },
})

/** The scatter-add primitive: linear, with `gatherOp` as its transpose; batched by shifting each example's indices. */
const scatterAddOp: Op<GatherParams> = definePrimitive<GatherParams>({
  id: 'foundation/tensor/scatterAdd',
  arity: 1,
  impl: ([g], p) => onParts(g, (part) => scatterAddRaw(part, p)),
  linear: 'linear',
  dtype: 'float',
  transpose: (ct, _inputs, _which, p) => gatherOp([ct], p),
  shape: ([g], { source }) => ({ shape: [...source], dtype: floatOrComplex(g.dtype), number: false }),
  batch: ([g], [axis], p, size) => [scatterAddOp([batchToFront(g, axis ?? 0)], batched(p, size)), 0],
  doc: { summary: 'Add values into flat indices: the adjoint of gather.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 2])], params: { indices: Int32Array.of(0, 5, 5, 2), shape: [2, 2], source: [2, 3] } },
    ],
  },
})

/**
 * Throw `ShapeError` unless every index lies in $[0, n)$, $n$ the number of elements indexed.
 *
 * @param indices The flat indices to check.
 * @param size The number $n$ of elements they index.
 * @param where The caller's name, for the error message.
 */
function checkIndices(indices: Int32Array, size: number, where: string): void {
  for (let k = 0; k < indices.length; k++)
    if (indices[k] < 0 || indices[k] >= size)
      throw new ShapeError(where, `${where}: index ${indices[k]} outside [0, ${size})`)
}

/**
 * `out[k] = x.flat[indices[k]]` (row-major flat indices), reshaped to `shape` (whose size must equal
 * `indices.length`). Differentiable in `x`: the cotangent is scattered back and summed where an index repeats. An index
 * out of range, or a size that differs from `indices.length`, throws `ShapeError`.
 *
 * @param x The tensor read, in row-major order.
 * @param indices The flat index into `x` of each element of the result.
 * @param shape The shape of the result.
 * @returns The gathered values, of shape `shape`: float64, or complex128 for a complex `x`.
 *
 * @example Read entries of a matrix by flat index
 * const x = tensor([[1, 2, 3], [4, 5, 6]])
 * print('gathered =', gather(x, Int32Array.of(0, 5, 5, 2), [2, 2]))
 *
 * @example The gradient adds up where an index repeats
 * const f = (x) => sum(gather(x, Int32Array.of(0, 2, 2), [3]))
 * print('gradient =', grad(f)(tensor([1, 2, 3])))
 */
export function gather(x: Value, indices: Int32Array, shape: readonly number[]): Value {
  if (indices.length !== sizeOf(shape)) throw new ShapeError('gather', 'gather: indices and shape disagree')
  const source = shapeOfValue(x)
  checkIndices(indices, sizeOf(source), 'gather')
  return gatherOp([x], { indices, shape, source })
}

/**
 * The adjoint of `gather`: a tensor of shape `shape` holding, at each flat index, the sum of the values of `g` gathered
 * from it (`out.flat[indices[k]] += g.flat[k]`). `g` must have `indices.length` elements. Differentiable in `g`. A
 * size mismatch, or an index out of range, throws `ShapeError`.
 *
 * @param g The values to add, one per index, in row-major order.
 * @param indices The flat index into the result of each value of `g`.
 * @param shape The shape of the result.
 * @returns A tensor of shape `shape`, zero where no index points: float64, or complex128 for a complex `g`.
 *
 * @example Count how often each index occurs
 * print('counts =', scatterAdd(ones([5]), Int32Array.of(0, 2, 2, 1, 2), [3]))
 */
export function scatterAdd(g: Value, indices: Int32Array, shape: readonly number[]): Value {
  const from = shapeOfValue(g)
  if (indices.length !== sizeOf(from)) throw new ShapeError('scatterAdd', 'scatterAdd: indices and values disagree')
  checkIndices(indices, sizeOf(shape), 'scatterAdd')
  return scatterAddOp([g], { indices, shape: from, source: shape })
}

/**
 * The rows `indices` of `x` along its first axis: shape `[...indices shape, ...x.shape.slice(1)]`. An embedding lookup
 * is `take(table, ids)`. Indices are integers in `[0, x.shape[0])` (others throw `ShapeError`, as does a number `x`);
 * the gradient adds up over repeated rows.
 *
 * @param x The table, of rank $\ge 1$, whose rows along the first axis are taken.
 * @param indices The row indices: a tensor (whose shape leads the result's) or an array of numbers.
 * @returns The rows, of shape `[...indices shape, ...x.shape.slice(1)]`.
 *
 * @example An embedding lookup
 * const table = tensor([[0, 0], [1, 10], [2, 20]])
 * print('rows 2, 0, 2 =', take(table, [2, 0, 2]))
 * print('shape for a 2 × 2 grid of ids:', shapeOfValue(take(table, tensor([[1, 2], [0, 1]]))))
 */
export function take(x: Value, indices: Tensor | ArrayLike<number>): Value {
  const shape = shapeOfValue(x)
  if (shape.length === 0) throw new ShapeError('take', 'take: needs a tensor of rank ≥ 1')
  const isT = isTensor(indices)
  const ids = isT ? toFlat(indices) : Array.from(indices as ArrayLike<number>)
  const idShape = isT ? [...indices.shape] : [ids.length]
  const rows = shape[0]
  const width = sizeOf(shape.slice(1))
  const flat = new Int32Array(ids.length * width)
  ids.forEach((id, k) => {
    if (!Number.isInteger(id) || id < 0 || id >= rows)
      throw new ShapeError('take', `take: index ${id} outside [0, ${rows})`)
    for (let j = 0; j < width; j++) flat[k * width + j] = id * width + j
  })
  return gather(x, flat, [...idShape, ...shape.slice(1)])
}
