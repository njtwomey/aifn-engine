/**
 * Structural operations: reshaping, permuting, slicing, joining and broadcasting. Views share data where possible.
 *
 * The primitives are `reshape`, `broadcastTo`, `sumTo`, `permute`, `slice` (with its adjoint `scatterSlice`) and
 * `concat`; each has a derivative rule that moves the cotangent back through the inverse rearrangement (Griewank and
 * Walther, 2008, ch. 3), e.g. a slice's cotangent is scattered into zeros of the input's shape. Everything else here
 * is a composition of primitives (design K §3.4, T8), so it has no rule of its own to test: `flatten`, `squeeze` and
 * `expandDims` are reshapes, `get` sums a one-element slice, `set` is a `where` against a one-hot mask, `stack` is
 * `expandDims` and `concat`, and `diagonal`/`diag` are `gather`/`scatterAdd` at the diagonal's flat indices.
 */

import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import { flatData, fromData, normaliseAxis, promote, showShape, size, sizeOf, type Axes, type Tensor } from './core'
import { astype, zeros } from './create'
import { where } from './elementwise'
import { sumToKernel } from './kernels'
import {
  batchToFront,
  definePrimitive,
  fitTo,
  sumLike,
  type NumberResult,
  type Op,
  type Raw,
  type TensorResult,
} from './primitive'
import { avalOf, type Value } from './trace'
import { gather, scatterAdd } from './gather'
import { sum } from './reduce'
import {
  broadcastShapes,
  broadcastView,
  checkPermutation,
  concatRaw,
  flatIndex,
  permuteView,
  resolveShape,
  reshapeView,
  sliceShape,
  sliceView,
  squeezedAxes,
  type SliceSpec,
} from './views'

/**
 * A raw input as a tensor; numbers become scalar tensors.
 *
 * @param x A number or an untraced tensor; a tensor is returned as it is.
 * @returns `x` as a tensor (a new float64 scalar tensor when `x` is a number).
 */
function asTensor(x: Raw): Tensor {
  return typeof x === 'number' ? fromData(new Float64Array([x]), []) : x
}

/**
 * The shape of a value (`[]` for a number).
 *
 * @param x A number, a tensor or a traced value; for a traced value, the shape of the value it stands for.
 * @returns A new array of the axis lengths, which the caller may modify.
 *
 * @example Shapes of a matrix, a vector and a number
 * print('matrix:', shapeOfValue(tensor([[1, 2, 3], [4, 5, 6]])))
 * print('vector:', shapeOfValue(tensor([1, 2])))
 * print('number:', shapeOfValue(7))
 */
export function shapeOfValue(x: Value): number[] {
  return [...avalOf(x).shape]
}

// A worked example of a linear primitive (design K §4.2): the author writes the forward rule, the transpose, the shape
// rule and the batching rule; the vjp (= transpose) and the jvp (= reshape of the tangent) are derived.
const reshapeOp = definePrimitive<readonly number[]>({
  id: 'foundation/tensor/reshape',
  dtype: 'same',
  arity: 1,
  impl: ([x], shape) => reshapeView(asTensor(x), shape),
  linear: 'linear',
  // The adjoint of a reshape is the reshape back (to a number when the input was one).
  transpose: (ct, [x]) => (avalOf(x).number ? sum(ct) : reshape(ct, avalOf(x).shape)),
  shape: ([x], shape) => ({ shape: resolveShape(sizeOf(x.shape), shape, x.shape), dtype: x.dtype, number: false }),
  // Each example reshapes on its own: batch axis first, then the example's new shape (its -1 resolved per example).
  batch: ([x], [axis], shape, size) => {
    const example = avalOf(x).shape.filter((_, k) => k !== axis)
    return [reshape(batchToFront(x, axis ?? 0), [size, ...resolveShape(sizeOf(example), shape, example)]), 0]
  },
  doc: { summary: 'The same elements with a new shape.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 3])], params: [3, 2] },
      { inputs: [draw([2, 3])], params: [-1] },
    ],
  },
})

/**
 * The same elements with a new shape; one entry may be $-1$ (inferred). A view when the tensor is contiguous (or only
 * axes of length $1$ are added or removed), otherwise a copy. Linear and differentiable: the cotangent is reshaped
 * back. Throws `ShapeError` when the sizes do not agree.
 *
 * @param x The value to reshape: a number, a tensor or a traced value.
 * @param shape The new shape, holding as many elements as `x`; one entry may be $-1$, filled in to make the sizes
 *   agree.
 * @returns The elements of `x` in row-major order, with shape `shape`.
 *
 * @example Six elements as three rows, then two
 * const x = arange(6)
 * print('[3, 2] =', reshape(x, [3, 2]))
 * print('[2, -1] =', reshape(x, [2, -1]))
 */
export function reshape<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return reshapeOp([x], shape) as TensorResult<X>
}

/**
 * A rank-1 tensor of the elements in row-major order: `reshape(x, [-1])`.
 *
 * @param x The value to flatten: a number, a tensor or a traced value.
 * @returns A vector of `size(x)` elements (one for a number).
 *
 * @example Rows one after another
 * print('flat =', flatten(tensor([[1, 2], [3, 4]])))
 */
export function flatten<X extends Value>(x: X): TensorResult<X> {
  return reshape(x, [-1])
}

const broadcastToOp: Op<readonly number[]> = definePrimitive<readonly number[]>({
  id: 'foundation/tensor/broadcastTo',
  dtype: 'same',
  arity: 1,
  impl: ([x], shape) => broadcastView(asTensor(x), shape),
  linear: 'linear',
  // The adjoint of broadcasting sums the cotangent over the repeated axes (to a number when the input was one).
  transpose: (ct, [x]) => sumLike(ct, x),
  shape: ([x], shape) => {
    broadcastShapes(x.shape, shape)
    return { shape: [...shape], dtype: x.dtype, number: false }
  },
  // The batch axis goes first, with length-1 axes after it so the example aligns (from the right) with the target.
  batch: ([x], [axis], shape, size) => {
    const pad = Math.max(0, shape.length - (avalOf(x).shape.length - 1))
    return [broadcastToOp([batchToFront(x, axis ?? 0, pad)], [size, ...shape]), 0]
  },
  doc: { summary: 'Broadcast to a shape by NumPy rules.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([3])], params: [2, 3] },
      { inputs: [draw([2, 1])], params: [2, 3] },
    ],
  },
})

/**
 * `x` broadcast to `shape` by NumPy rules (a view with stride $0$ along repeated axes). Linear and differentiable: the
 * cotangent is summed over the repeated axes. Throws `ShapeError` when `x` does not broadcast to `shape`.
 *
 * @param x The value to broadcast: a number, a tensor or a traced value. Its shape is right-aligned with `shape`, and
 *   each of its axes has the target length or length $1$.
 * @param shape The target shape.
 * @returns `x` repeated to shape `shape`.
 *
 * @example A row repeated down three rows
 * print('row to [3, 2] =', broadcastTo(tensor([1, 2]), [3, 2]))
 * print('number to [2] =', broadcastTo(5, [2]))
 */
export function broadcastTo<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return broadcastToOp([x], shape) as TensorResult<X>
}

const sumToOp: Op<readonly number[]> = definePrimitive<readonly number[]>({
  id: 'foundation/tensor/sumTo',
  dtype: 'float',
  arity: 1,
  // As `sum` (the `float` rule): integer sums come back as float64.
  impl: ([x], shape) => {
    const r = sumToKernel(asTensor(x), shape)
    return r.dtype === 'int32' ? astype(r, 'float64') : r
  },
  linear: 'linear',
  // The adjoint of summing down is broadcasting back up (to a number when the input was one).
  transpose: (ct, [x]) => fitTo(ct, avalOf(x)),
  shape: ([x], shape) => ({
    shape: [...shape],
    dtype: x.dtype === 'complex128' ? x.dtype : 'float64',
    number: false,
  }),
  // Leading axes of an example are summed away; with the batch axis first, the target gets length-1 axes in their
  // place (summed, kept) and a reshape drops them.
  batch: ([x], [axis], shape, size) => {
    const pad = avalOf(x).shape.length - 1 - shape.length
    if (pad <= 0) return [sumToOp([batchToFront(x, axis ?? 0)], [size, ...shape]), 0]
    const kept = sumToOp([batchToFront(x, axis ?? 0)], [size, ...new Array<number>(pad).fill(1), ...shape])
    return [reshape(kept, [size, ...shape]), 0]
  },
  doc: { summary: 'Sum down to a shape: the adjoint of broadcasting.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 3])], params: [3] },
      { inputs: [draw([2, 3])], params: [2, 1] },
    ],
  },
})

/**
 * Sum `x` down to `shape`, the adjoint of broadcasting `shape` up to `x`'s shape: leading axes are summed away and
 * axes where `shape` has length $1$ are summed keeping length $1$. Integer input gives float64, as `sum` does.
 * Linear and differentiable: the cotangent is broadcast back up.
 *
 * @param x The value to sum: a number, a tensor or a traced value.
 * @param shape The shape to sum down to: one that broadcasts to `x`'s shape.
 * @returns The sums, with shape `shape`.
 *
 * @example Sum a matrix down to a row and to a column
 * const x = tensor([[1, 2, 3], [4, 5, 6]])
 * print('to [3] =', sumTo(x, [3]))
 * print('to [2, 1] =', sumTo(x, [2, 1]))
 */
export function sumTo<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return sumToOp([x], shape) as TensorResult<X>
}

/**
 * The inverse of a permutation: the order that undoes it.
 *
 * @param order A permutation of $0, \dots, n-1$, already normalised (no negative entries).
 * @returns The permutation `q` with `q[order[k]] = k` for every `k`.
 */
function inversePermutation(order: readonly number[]): number[] {
  const inverse = new Array<number>(order.length)
  order.forEach((a, k) => (inverse[a] = k))
  return inverse
}

const permuteOp: Op<readonly number[]> = definePrimitive<readonly number[]>({
  id: 'foundation/tensor/permute',
  dtype: 'same',
  arity: 1,
  impl: ([x], axes) => permuteView(asTensor(x), checkPermutation(axes, asTensor(x).shape.length)),
  linear: 'linear',
  // The adjoint of a permutation is the inverse permutation.
  transpose: (ct, [x], _which, axes) => {
    const aval = avalOf(x)
    return fitTo(permute(ct, inversePermutation(checkPermutation(axes, aval.shape.length))), aval)
  },
  shape: ([x], axes) => ({
    shape: checkPermutation(axes, x.shape.length).map((a) => x.shape[a]),
    dtype: x.dtype,
    number: false,
  }),
  // The batch axis leads the output; the example's axes keep their order after it.
  batch: ([x], [axis], axes) => {
    const b = axis ?? 0
    const order = checkPermutation(axes, avalOf(x).shape.length - 1)
    return [permuteOp([x], [b, ...order.map((a) => (a < b ? a : a + 1))]), 0]
  },
  doc: { summary: 'Reorder the axes.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [{ inputs: [draw([2, 3, 4])], params: [2, 0, 1] }],
  },
})

/**
 * A view with the axes reordered: axis $k$ of the result is axis `axes[k]` of `x`. Linear and differentiable: the
 * cotangent is permuted back by the inverse permutation. Throws `ShapeError` when `axes` is not a permutation.
 *
 * @param x The value whose axes are reordered: a tensor or a traced value.
 * @param axes A permutation of `x`'s axes, each once; negative entries count from the last axis.
 * @returns The permuted view; it shares `x`'s data.
 *
 * @example Move the last axis first
 * const x = zeros([2, 3, 4])
 * print('shape =', shapeOfValue(permute(x, [2, 0, 1])))
 */
export function permute<X extends Value>(x: X, axes: readonly number[]): TensorResult<X> {
  return permuteOp([x], axes) as TensorResult<X>
}

/**
 * A view with the axes reversed (the matrix transpose for rank $2$), or permuted by `axes` when given. A `permute`.
 *
 * @param x The value to transpose: a number, a tensor or a traced value. A vector is returned unchanged.
 * @param axes The permutation to apply, as `permute` takes it; left out, the axes are reversed.
 * @returns The transposed view; it shares `x`'s data.
 *
 * @example The transpose of a 2 by 3 matrix
 * print('Mᵀ =', transpose(tensor([[1, 2, 3], [4, 5, 6]])))
 */
export function transpose<X extends Value>(x: X, axes?: readonly number[]): TensorResult<X> {
  const rank = shapeOfValue(x).length
  return permute(x, axes ?? Array.from({ length: rank }, (_, k) => rank - 1 - k))
}

/**
 * A view without axes of length $1$: all of them, or only `axis` (each of which must have length $1$, or `ShapeError`
 * is thrown). A reshape.
 *
 * @param x The value to squeeze: a tensor or a traced value.
 * @param axis The axis or axes to remove (negative counts from the end); left out, every axis of length $1$ goes.
 * @returns `x` with those axes removed.
 *
 * @example Drop every unit axis, or just one
 * const x = zeros([1, 3, 1])
 * print('all:', shapeOfValue(squeeze(x)))
 * print('axis 0:', shapeOfValue(squeeze(x, 0)))
 */
export function squeeze<X extends Value>(x: X, axis?: Axes): TensorResult<X> {
  const shape = shapeOfValue(x)
  const drop = squeezedAxes(shape, axis)
  return reshape(
    x,
    shape.filter((_, k) => !drop.includes(k)),
  )
}

/**
 * A view with a new axis of length $1$ inserted at position `axis` of the result (negative counts from the end). A
 * reshape.
 *
 * @param x The value to expand: a number, a tensor or a traced value.
 * @param axis Where the new axis goes in the result, from $0$ to the rank of `x`; $-1$ appends it last.
 * @returns `x` with one more axis.
 *
 * @example A vector as a column and as a row
 * const v = tensor([1, 2, 3])
 * print('column:', shapeOfValue(expandDims(v, -1)))
 * print('row:', shapeOfValue(expandDims(v, 0)))
 */
export function expandDims<X extends Value>(x: X, axis: number): TensorResult<X> {
  const shape = shapeOfValue(x)
  const a = normaliseAxis(axis, shape.length + 1, 'expandDims')
  return reshape(x, [...shape.slice(0, a), 1, ...shape.slice(a)])
}

/**
 * Parameters of the `scatterSlice` primitive: `shape` is the shape of the output (that of the tensor that was sliced),
 * and `specs` the slice whose positions receive the values.
 */
type Scatter = { shape: readonly number[]; specs: readonly SliceSpec[] }

// The adjoint of slicing: zeros of the input's shape with the cotangent written into the sliced positions. Linear, and
// slicing is its transpose.
const scatterSliceOp: Op<Scatter> = definePrimitive<Scatter>({
  id: 'foundation/tensor/scatterSlice',
  arity: 1,
  impl: ([g], { shape, specs }) => {
    const src = asTensor(g)
    const complex = src.dtype === 'complex128'
    const out = zeros(shape, complex ? 'complex128' : 'float64')
    const target = sliceView(out, specs)
    const values = flatData(src, complex ? 'complex128' : 'float64')
    let k = 0
    if (complex)
      forEachTarget(target, (off) => {
        out.data[2 * off] = values[2 * k]
        out.data[2 * off + 1] = values[2 * k++ + 1]
      })
    else forEachTarget(target, (off) => (out.data[off] = values[k++]))
    return out
  },
  linear: 'linear',
  dtype: 'float',
  transpose: (ct, [g], _which, { specs }) => fitTo(slice(ct, ...specs), avalOf(g)),
  shape: ([g], { shape }) => ({
    shape: [...shape],
    dtype: g.dtype === 'complex128' ? 'complex128' : 'float64',
    number: false,
  }),
  batch: ([g], [axis], { shape, specs }, size) => [
    scatterSliceOp([batchToFront(g, axis ?? 0)], { shape: [size, ...shape], specs: [null, ...specs] }),
    0,
  ],
  doc: { summary: 'Zeros with values written into a slice: the adjoint of slicing.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [{ inputs: [draw([2, 3])], params: { shape: [4, 3], specs: [[0, 4, 2]] } }],
  },
})

/**
 * Visit the data offsets of a view in row-major order.
 *
 * @param t The view whose elements are visited; only its shape, strides and offset are read.
 * @param body Called once per element with the element's position in `t.data`, counted in elements (a complex
 *   element occupies positions `2 * off` and `2 * off + 1`).
 */
function forEachTarget(t: Tensor, body: (off: number) => void): void {
  const n = size(t)
  const index = new Array<number>(t.shape.length).fill(0)
  for (let k = 0; k < n; k++) {
    let off = t.offset
    for (let a = 0; a < index.length; a++) off += index[a] * t.strides[a]
    body(off)
    for (let a = index.length - 1; a >= 0; a--) {
      if (++index[a] < t.shape[a]) break
      index[a] = 0
    }
  }
}

const sliceOp: Op<readonly SliceSpec[]> = definePrimitive<readonly SliceSpec[]>({
  id: 'foundation/tensor/slice',
  dtype: 'same',
  arity: 1,
  impl: ([x], specs) => sliceView(asTensor(x), specs),
  linear: 'linear',
  // The adjoint scatters the cotangent into zeros of the input's shape.
  transpose: (ct, [x], _which, specs) => {
    const aval = avalOf(x)
    return fitTo(scatterSliceOp([ct], { shape: aval.shape, specs }), aval)
  },
  shape: ([x], specs) => ({ shape: sliceShape(x.shape, specs), dtype: x.dtype, number: false }),
  // With the batch axis first, a leading `null` spec keeps it whole.
  batch: ([x], [axis], specs) => [sliceOp([batchToFront(x, axis ?? 0)], [null, ...specs]), 0],
  doc: { summary: 'Basic indexing: a view of part of a tensor.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([4, 3])], params: [[1, 3], null] },
      { inputs: [draw([5])], params: [[null, null, -1]] },
      { inputs: [draw([3, 4])], params: [1] },
    ],
  },
})

/**
 * A view selecting part of a tensor, one spec per leading axis (missing trailing specs keep their axes whole), as
 * NumPy basic indexing. An integer picks a position and drops the axis, `null` keeps the axis, and
 * `[start, stop, step]` is Python's `start:stop:step` (negative values count from the end). Never copies. Linear and
 * differentiable: the cotangent is scattered into zeros of the input's shape. Throws `ShapeError` for more specs than
 * axes or an integer index out of range.
 *
 * @param x The value to slice: a tensor or a traced value.
 * @param specs One `SliceSpec` per leading axis of `x`, as separate arguments.
 * @returns The selected part of `x`, sharing its data.
 *
 * @example Rows, columns and every second column
 * const m = tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
 * print('first row =', slice(m, 0))
 * print('last column =', slice(m, null, -1))
 * print('every second column =', slice(m, null, [0, null, 2]))
 *
 * @example A step of -1 reverses
 * print('reversed =', slice(tensor([1, 2, 3, 4]), [null, null, -1]))
 */
export function slice<X extends Value>(x: X, ...specs: SliceSpec[]): TensorResult<X> {
  return sliceOp([x], specs) as TensorResult<X>
}

/**
 * A one-hot mask: $1$ at `index`, $0$ elsewhere. Throws `ShapeError` when `index` does not fit `shape`.
 *
 * @param shape The shape of the mask.
 * @param index A full multi-index into `shape` (negative entries count from the end).
 * @returns A new float64 tensor of shape `shape`.
 */
function oneHot(shape: readonly number[], index: readonly number[]): Tensor {
  const out = new Float64Array(sizeOf(shape))
  out[flatIndex(shape, index, 'oneHot')] = 1
  return fromData(out, shape)
}

/**
 * The element at a full multi-index; negative indices count from the end. `get(m, i, j)` reads `m[i, j]`. A
 * composition: the sum of the one-element slice, so its derivative scatters the cotangent back to that element.
 * Throws `ShapeError` when the index has the wrong length or is out of range.
 *
 * @param x The value to read: a tensor or a traced value.
 * @param index One integer per axis of `x`, as separate arguments.
 * @returns The element, as a number (or a traced scalar when `x` is traced).
 *
 * @example Read an element, counting from either end
 * const m = tensor([[1, 2, 3], [4, 5, 6]])
 * print('m[0, 1] =', get(m, 0, 1))
 * print('m[-1, -1] =', get(m, -1, -1))
 *
 * @example Its gradient picks out the element
 * print('gradient =', grad((x) => get(x, 1))(tensor([10, 20, 30])))
 */
export function get<X extends Value>(x: X, ...index: number[]): NumberResult<X> {
  const shape = shapeOfValue(x)
  flatIndex(shape, index, 'get')
  return sum(slice(x, ...index)) as NumberResult<X>
}

/**
 * A copy of `x` with the element at `index` replaced by `value` (a number, possibly traced); `x` is unchanged. A
 * composition: `where` against a one-hot mask, so it is differentiable in both `x` and `value`. Throws `AifnError`
 * when `value` is not a scalar, and `ShapeError` when `index` does not fit `x`.
 *
 * @param x The value to copy: a tensor or a traced value.
 * @param index A full multi-index into `x`, one integer per axis (negative entries count from the end).
 * @param value The new element: a number or a traced scalar.
 * @returns A new tensor shaped like `x`, equal to it except at `index`.
 *
 * @example Replace one element, leaving the input as it was
 * const x = tensor([[1, 2], [3, 4]])
 * print('set =', set(x, [1, 0], 9))
 * print('x =', x)
 */
export function set<X extends Value>(x: X, index: readonly number[], value: Value): TensorResult<X> {
  if (shapeOfValue(value).length !== 0) throw new AifnError('set', 'set: value must be a number')
  return where(oneHot(shapeOfValue(x), index), value, x) as TensorResult<X>
}

const concatOp: Op<number> = definePrimitive<number>({
  id: 'foundation/tensor/concat',
  dtype: 'same',
  impl: (xs, axis) => concatRaw(xs.map(asTensor), axis),
  linear: 'linear',
  // The adjoint in input i is the block of the cotangent that input i filled.
  transpose: (ct, xs, which, axis) => {
    const a = normaliseAxis(axis, shapeOfValue(xs[0]).length, 'concat')
    let start = 0
    for (let i = 0; i < which; i++) start += shapeOfValue(xs[i])[a]
    const n = shapeOfValue(xs[which])[a]
    const specs: SliceSpec[] = Array.from({ length: a + 1 }, (_, k) => (k === a ? [start, start + n] : null))
    return slice(ct, ...specs)
  },
  shape: (avals, axis) => {
    const a = normaliseAxis(axis, avals[0].shape.length, 'concat')
    const shape = [...avals[0].shape]
    shape[a] = avals.reduce((s, x) => s + x.shape[a], 0)
    return { shape, dtype: avals.map((x) => x.dtype).reduce(promote), number: false }
  },
  // Batched inputs move their batch axis first; unbatched ones are repeated along a new first axis.
  batch: (xs, axes, axis, size) => {
    const rank = avalOf(xs[axes.findIndex((b) => b !== null)]).shape.length - 1
    const a = normaliseAxis(axis, rank, 'concat')
    const moved = xs.map((x, i) => {
      const b = axes[i]
      if (b !== null) return batchToFront(x, b)
      return broadcastTo(reshape(x, [1, ...shapeOfValue(x)]), [size, ...shapeOfValue(x)])
    })
    return [concatOp(moved, a + 1), 0]
  },
  doc: { summary: 'Join tensors along an existing axis.' },
  test: {
    complex: true,
    secondOrder: true,
    cases: (draw) => [
      { inputs: [draw([2, 3]), draw([1, 3])], params: 0 },
      { inputs: [draw([2, 3]), draw([2, 2])], params: 1 },
    ],
  },
})

/**
 * Join tensors along an existing axis; the other axes must match. Copies; the dtype is the common promotion. Linear
 * and differentiable: each input's cotangent is the block of the output's cotangent that the input filled. Throws
 * `ShapeError` for an empty list, scalars, or shapes that differ off the axis.
 *
 * @param xs The values to join, in order: tensors or traced values of the same rank (at least $1$) and the same
 *   lengths on every axis but `axis`.
 * @param axis The axis to join along (negative counts from the end).
 * @returns The joined tensor, whose length on `axis` is the sum of the inputs' lengths.
 *
 * @example Stack rows, or join side by side
 * const a = tensor([[1, 2], [3, 4]])
 * const b = tensor([[5, 6]])
 * print('rows =', concat([a, b]))
 * print('columns =', concat([a, transpose(b)], 1))
 */
export function concat(xs: readonly Tensor[], axis?: number): Tensor
export function concat(xs: readonly Value[], axis?: number): Value
export function concat(xs: readonly Value[], axis = 0): Value {
  return concatOp(xs, axis)
}

/**
 * Join tensors of equal shape along a new axis at position `axis`. Copies. A composition of `expandDims` and `concat`,
 * so differentiable. Throws `ShapeError` for an empty list or shapes that differ.
 *
 * @param xs The values to join, in order: numbers, tensors or traced values, all of the same shape.
 * @param axis Where the new axis goes in the result, from $0$ to the rank of the inputs (negative counts from the
 *   end).
 * @returns A tensor with one more axis than the inputs, of length `xs.length` at `axis`.
 *
 * @example Two vectors as the rows, or the columns, of a matrix
 * const a = tensor([1, 2, 3])
 * const b = tensor([4, 5, 6])
 * print('rows =', stack([a, b]))
 * print('columns =', stack([a, b], 1))
 */
export function stack(xs: readonly Tensor[], axis?: number): Tensor
export function stack(xs: readonly Value[], axis?: number): Value
export function stack(xs: readonly Value[], axis = 0): Value {
  if (xs.length === 0) throw new ShapeError('stack', 'stack: nothing to join')
  const first = shapeOfValue(xs[0])
  for (const x of xs) {
    const s = shapeOfValue(x)
    if (s.length !== first.length || s.some((d, k) => d !== first[k])) {
      throw new ShapeError('stack', `stack: shape ${showShape(s)} differs from ${showShape(first)}`, [s, first])
    }
  }
  const a = normaliseAxis(axis, first.length + 1, 'stack')
  return concat(
    xs.map((x) => expandDims(x, a)),
    a,
  )
}

/**
 * The flat row-major indices of the main diagonal of an $m \times n$ matrix.
 *
 * @param m The number of rows.
 * @param n The number of columns.
 * @returns The $\min(m, n)$ positions $i n + i$ of the entries $(i, i)$.
 */
function diagonalIndices(m: number, n: number): Int32Array {
  return Int32Array.from({ length: Math.min(m, n) }, (_, i) => i * n + i)
}

/**
 * The main diagonal of a matrix, as a vector of length $\min(m, n)$. A `gather` at the diagonal's flat indices, so
 * differentiable. Throws `ShapeError` when `x` is not a matrix.
 *
 * @param x The $m \times n$ matrix: a tensor or a traced value.
 * @returns The entries $x_{ii}$, in order.
 *
 * @example The diagonal of a square and of a wide matrix
 * print('square:', diagonal(tensor([[1, 2], [3, 4]])))
 * print('wide:', diagonal(tensor([[1, 2, 3], [4, 5, 6]])))
 */
export function diagonal<X extends Value>(x: X): TensorResult<X> {
  const shape = shapeOfValue(x)
  if (shape.length !== 2)
    throw new ShapeError('diagonal', `diagonal: expected a matrix, got shape ${showShape(shape)}`, [shape])
  const indices = diagonalIndices(shape[0], shape[1])
  return gather(x, indices, [indices.length]) as TensorResult<X>
}

/**
 * A square matrix with `v` on its diagonal and zeros elsewhere. A `scatterAdd` at the diagonal's flat indices, so
 * differentiable. Throws `ShapeError` when `v` is not a vector.
 *
 * @param v The diagonal: a vector of $n$ values (a tensor or a traced value).
 * @returns The $n \times n$ matrix $\diag(\vvec)$.
 *
 * @example From a vector to a diagonal matrix and back
 * const D = diag(tensor([1, 2, 3]))
 * print('D =', D)
 * print('diagonal(D) =', diagonal(D))
 */
export function diag<X extends Value>(v: X): TensorResult<X> {
  const shape = shapeOfValue(v)
  if (shape.length !== 1)
    throw new ShapeError('diag', `diag: expected a vector, got shape ${showShape(shape)}`, [shape])
  const n = shape[0]
  return scatterAdd(v, diagonalIndices(n, n), [n, n]) as TensorResult<X>
}

/**
 * A batching rule that loops: apply `op` to each example (index `b` of every batched input) and stack the results
 * along a new first axis. For primitives with no cheaper rule in some cases (e.g. a convolution whose kernels are
 * batched); it is what `vmap` does for a primitive with no rule at all, written as a rule. Examples are tensors.
 *
 * @param op The primitive to apply to one example at a time.
 * @param values The primitive's inputs, batched or not.
 * @param axes One entry per input: the axis of that input along which examples run, or `null` for an input shared by
 *   every example (passed to `op` whole).
 * @param params The primitive's parameters, passed unchanged to every call.
 * @param size The number of examples, the length of every batched axis.
 * @returns The stacked results and the batch axis of the result, which is always $0$.
 *
 * @example Apply an operation row by row
 * const op = ([x, w], p) => mul(add(x, w), p)
 * const [out, axis] = batchByLoop(op, [tensor([[1, 2], [3, 4]]), tensor([10, 20])], [0, null], 2, 2)
 * print('out =', out)
 * print('batch axis =', axis)
 */
export function batchByLoop<P>(
  op: Op<P>,
  values: readonly Value[],
  axes: readonly (number | null)[],
  params: P,
  size: number,
): [Value, number] {
  const outs: Value[] = []
  for (let b = 0; b < size; b++) {
    const args = values.map((v, i) => {
      const axis = axes[i]
      if (axis === null) return v
      return slice(v, ...Array.from({ length: axis + 1 }, (_, k): SliceSpec => (k === axis ? b : null)))
    })
    outs.push(op(args, params))
  }
  return [stack(outs, 0), 0]
}
