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

/** A raw input as a tensor; numbers become scalar tensors. */
function asTensor(x: Raw): Tensor {
  return typeof x === 'number' ? fromData(new Float64Array([x]), []) : x
}

/** The shape of a value (`[]` for a number). */
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
 * The same elements with a new shape; one entry may be -1 (inferred). A view when the tensor is contiguous, otherwise
 * a copy.
 */
export function reshape<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return reshapeOp([x], shape) as TensorResult<X>
}

/** A rank-1 tensor of the elements in row-major order. */
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

/** `x` broadcast to `shape` by NumPy rules (a view with stride 0 along repeated axes). */
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
 * axes where `shape` has length 1 are summed keeping length 1.
 */
export function sumTo<X extends Value>(x: X, shape: readonly number[]): TensorResult<X> {
  return sumToOp([x], shape) as TensorResult<X>
}

/** The inverse of a permutation. */
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

/** A view with the axes reordered: axis k of the result is axis `axes[k]` of `x`. */
export function permute<X extends Value>(x: X, axes: readonly number[]): TensorResult<X> {
  return permuteOp([x], axes) as TensorResult<X>
}

/** A view with the axes reversed (the matrix transpose for rank 2), or permuted by `axes` when given. */
export function transpose<X extends Value>(x: X, axes?: readonly number[]): TensorResult<X> {
  const rank = shapeOfValue(x).length
  return permute(x, axes ?? Array.from({ length: rank }, (_, k) => rank - 1 - k))
}

/** A view without axes of length 1: all of them, or only `axis` (each of which must have length 1). A reshape. */
export function squeeze<X extends Value>(x: X, axis?: Axes): TensorResult<X> {
  const shape = shapeOfValue(x)
  const drop = squeezedAxes(shape, axis)
  return reshape(
    x,
    shape.filter((_, k) => !drop.includes(k)),
  )
}

/**
 * A view with a new axis of length 1 inserted at position `axis` of the result (negative counts from the end). A
 * reshape.
 */
export function expandDims<X extends Value>(x: X, axis: number): TensorResult<X> {
  const shape = shapeOfValue(x)
  const a = normaliseAxis(axis, shape.length + 1, 'expandDims')
  return reshape(x, [...shape.slice(0, a), 1, ...shape.slice(a)])
}

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

/** Visit the data offsets of a view in row-major order. */
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
 * `[start, stop, step]` is Python's `start:stop:step` (negative values count from the end). Never copies.
 *
 * @example slice(m, 0) // the first row of a matrix
 * @example slice(m, null, [0, null, 2]) // every second column
 * @example slice(v, [null, null, -1]) // v reversed
 */
export function slice<X extends Value>(x: X, ...specs: SliceSpec[]): TensorResult<X> {
  return sliceOp([x], specs) as TensorResult<X>
}

/** A one-hot mask: 1 at `index`, 0 elsewhere. */
function oneHot(shape: readonly number[], index: readonly number[]): Tensor {
  const out = new Float64Array(sizeOf(shape))
  out[flatIndex(shape, index, 'oneHot')] = 1
  return fromData(out, shape)
}

/**
 * The element at a full multi-index; negative indices count from the end. `get(m, i, j)` reads m[i, j]. A composition:
 * the sum of the one-element slice, so its derivative scatters the cotangent back to that element.
 */
export function get<X extends Value>(x: X, ...index: number[]): NumberResult<X> {
  const shape = shapeOfValue(x)
  flatIndex(shape, index, 'get')
  return sum(slice(x, ...index)) as NumberResult<X>
}

/**
 * A copy of `x` with the element at `index` replaced by `value` (a number, possibly traced); `x` is unchanged. A
 * composition: `where` against a one-hot mask, so it is differentiable in both `x` and `value`.
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

/** Join tensors along an existing axis; the other axes must match. Copies; the dtype is the common promotion. */
export function concat(xs: readonly Tensor[], axis?: number): Tensor
export function concat(xs: readonly Value[], axis?: number): Value
export function concat(xs: readonly Value[], axis = 0): Value {
  return concatOp(xs, axis)
}

/** Join tensors of equal shape along a new axis at position `axis`. Copies. */
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

/** The flat row-major indices of the main diagonal of an m × n matrix. */
function diagonalIndices(m: number, n: number): Int32Array {
  return Int32Array.from({ length: Math.min(m, n) }, (_, i) => i * n + i)
}

/** The main diagonal of a matrix, as a vector of length min(m, n). A `gather` at the diagonal's flat indices. */
export function diagonal<X extends Value>(x: X): TensorResult<X> {
  const shape = shapeOfValue(x)
  if (shape.length !== 2)
    throw new ShapeError('diagonal', `diagonal: expected a matrix, got shape ${showShape(shape)}`, [shape])
  const indices = diagonalIndices(shape[0], shape[1])
  return gather(x, indices, [indices.length]) as TensorResult<X>
}

/** A square matrix with `v` on its diagonal and zeros elsewhere. A `scatterAdd` at the diagonal's flat indices. */
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
