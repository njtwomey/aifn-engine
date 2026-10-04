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

/** Row-major float64 data of a raw value (no copy when already contiguous float64 at offset 0). */
function f64(x: number | Tensor, where: string): Float64Array {
  if (typeof x === 'number') throw new AifnError(where, `${where}: expected a tensor, got a number`)
  if (x.dtype === 'float64' && isContiguous(x) && x.offset === 0 && x.data.length === sizeOf(x.shape)) {
    return x.data as Float64Array
  }
  return Float64Array.from(toFlat(x))
}

type GatherParams = { indices: Int32Array; shape: readonly number[]; source: readonly number[] }

const floatOrComplex = (dtype: string) => (dtype === 'complex128' ? 'complex128' : 'float64')

/** Apply a real kernel to a raw value, or to the real and imaginary views of a complex one and join the results. */
function onParts(x: number | Tensor, kernel: (part: number | Tensor) => Tensor): Tensor {
  if (typeof x === 'number' || x.dtype !== 'complex128') return kernel(x)
  const [re, im] = splitComplex(x)
  return joinComplex(kernel(re), kernel(im!))
}

function gatherRaw(x: number | Tensor, { indices, shape }: GatherParams): Tensor {
  const data = f64(x, 'gather')
  const out = new Float64Array(indices.length)
  for (let k = 0; k < indices.length; k++) out[k] = data[indices[k]]
  return fromData(out, shape)
}

function scatterAddRaw(g: number | Tensor, { indices, source }: GatherParams): Tensor {
  const data = f64(g, 'scatterAdd')
  const out = new Float64Array(sizeOf(source))
  for (let k = 0; k < indices.length; k++) out[indices[k]] += data[k]
  return fromData(out, source)
}

/**
 * The parameters of a gather or scatter over a batch of `size` examples stacked along a new first axis: example b's
 * flat indices are shifted by b times the size of one example's source.
 */
function batched(p: GatherParams, size: number): GatherParams {
  const n = p.indices.length
  const stride = sizeOf(p.source)
  const indices = new Int32Array(size * n)
  for (let b = 0; b < size; b++) for (let k = 0; k < n; k++) indices[b * n + k] = p.indices[k] + b * stride
  return { indices, shape: [size, ...p.shape], source: [size, ...p.source] }
}

// gather and scatterAdd are linear and each other's transpose, so each has derivatives of every order.
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

function checkIndices(indices: Int32Array, size: number, where: string): void {
  for (let k = 0; k < indices.length; k++)
    if (indices[k] < 0 || indices[k] >= size)
      throw new ShapeError(where, `${where}: index ${indices[k]} outside [0, ${size})`)
}

/**
 * out[k] = x.flat[indices[k]] (row-major flat indices), reshaped to `shape` (whose size must equal indices.length).
 * Differentiable in x: the cotangent is scattered back and summed where an index repeats.
 */
export function gather(x: Value, indices: Int32Array, shape: readonly number[]): Value {
  if (indices.length !== sizeOf(shape)) throw new ShapeError('gather', 'gather: indices and shape disagree')
  const source = shapeOfValue(x)
  checkIndices(indices, sizeOf(source), 'gather')
  return gatherOp([x], { indices, shape, source })
}

/**
 * The adjoint of `gather`: a tensor of shape `shape` holding, at each flat index, the sum of the values of `g` gathered
 * from it (out.flat[indices[k]] += g.flat[k]). `g` must have indices.length elements. Differentiable in g.
 */
export function scatterAdd(g: Value, indices: Int32Array, shape: readonly number[]): Value {
  const from = shapeOfValue(g)
  if (indices.length !== sizeOf(from)) throw new ShapeError('scatterAdd', 'scatterAdd: indices and values disagree')
  checkIndices(indices, sizeOf(shape), 'scatterAdd')
  return scatterAddOp([g], { indices, shape: from, source: shape })
}

/**
 * The rows `indices` of x along its first axis: shape [...indices shape, ...x.shape.slice(1)]. An embedding lookup is
 * `take(table, ids)`. Indices are integers in [0, x.shape[0]); the gradient adds up over repeated rows.
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
