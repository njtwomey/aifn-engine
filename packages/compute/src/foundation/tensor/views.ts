/**
 * Raw indexing, views and shape manipulation on plain tensors (no tracing). The public, differentiable versions in
 * `structure.ts` call these. Views share data; functions that must copy say so.
 *
 * A view is a new shape, strides and offset over the same storage: element $(i_0, \dots, i_{r-1})$ of a view sits at
 * position $o + \sum_k s_k i_k$ of the data, with $o$ the offset and $s_k$ the strides. Slicing, permuting and
 * broadcasting change only those three, so they never copy; broadcasting repeats an element with stride $0$. Shapes
 * broadcast by NumPy's rules: they are right-aligned, and each axis has a common length or length $1$.
 */

import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import {
  allocate,
  checkShape,
  flatData,
  forEachOffset2,
  fromData,
  isContiguous,
  normaliseAxes,
  normaliseAxis,
  promote,
  rowMajorStrides,
  showShape,
  size,
  sizeOf,
  view,
  type Axes,
  type Tensor,
} from './core'

/**
 * Row-major position of a multi-index (negative entries count from the end), checked against the shape: throws
 * `ShapeError` for the wrong number of indices or one out of range.
 *
 * @param shape The shape the index addresses.
 * @param index One integer per axis of `shape`; a negative entry $-k$ counts $k$ back from the end of its axis.
 * @param where The caller's name, for error messages.
 * @returns The position $\sum_k i_k \prod_{j > k} n_j$ of the element in a contiguous row-major layout of `shape`
 *   (not counting any tensor's offset or strides).
 */
export function flatIndex(shape: readonly number[], index: readonly number[], where: string): number {
  if (index.length !== shape.length)
    throw new ShapeError(where, `${where}: ${index.length} indices for rank ${shape.length}`)
  let k = 0
  for (let axis = 0; axis < shape.length; axis++) {
    const n = shape[axis]
    const i = index[axis] < 0 ? index[axis] + n : index[axis]
    if (!Number.isInteger(i) || i < 0 || i >= n) {
      throw new ShapeError(where, `${where}: index ${index[axis]} out of range for axis ${axis} (length ${n})`)
    }
    k = k * n + i
  }
  return k
}

/**
 * One axis of a slice: an integer picks one position and drops the axis; `null` keeps the whole axis; a tuple
 * `[start, stop, step]` selects like Python's `start:stop:step`, where `null` or a missing entry means the default and
 * negative values count from the end.
 */
export type SliceSpec = number | null | readonly [start?: number | null, stop?: number | null, step?: number | null]

/**
 * A view selecting part of a tensor, as NumPy basic indexing. Shares `t`'s data; throws `ShapeError` for more specs
 * than axes or an integer index out of range, and `AifnError` for a zero or non-integer step.
 *
 * @param t The tensor to slice; not modified.
 * @param specs One `SliceSpec` per leading axis of `t`; axes past the end of the list are kept whole.
 * @returns A view of `t` with one axis per non-integer spec (and per unlisted axis).
 */
export function sliceView(t: Tensor, specs: readonly SliceSpec[]): Tensor {
  const g = sliceGeometry(t.shape, t.strides, t.offset, specs)
  return view(t, g.shape, g.strides, g.offset)
}

/**
 * The shape of a slice of a tensor of shape `shape` (for shape rules), without a tensor to slice.
 *
 * @param shape The shape of the tensor being sliced.
 * @param specs The slice, as `sliceView` takes it.
 * @returns The shape `sliceView` would give.
 */
export function sliceShape(shape: readonly number[], specs: readonly SliceSpec[]): number[] {
  return sliceGeometry(shape, new Array<number>(shape.length).fill(0), 0, specs).shape
}

/**
 * The shape, strides and offset of a slice of a layout. Out-of-range bounds of a range are clamped as Python's
 * `slice.indices` does; an integer index out of range throws `ShapeError`, a zero or non-integer step `AifnError`.
 *
 * @param inShape The shape of the layout being sliced.
 * @param inStrides Its strides, one per axis, in elements.
 * @param inOffset Its offset into the data, in elements.
 * @param specs One `SliceSpec` per leading axis; axes past the end of the list are kept whole.
 * @returns The layout of the slice: an integer spec drops its axis and moves the offset to that position, and a range
 *   keeps the axis with its length and the stride multiplied by the step.
 */
function sliceGeometry(
  inShape: readonly number[],
  inStrides: readonly number[],
  inOffset: number,
  specs: readonly SliceSpec[],
): { shape: number[]; strides: number[]; offset: number } {
  if (specs.length > inShape.length)
    throw new ShapeError('slice', `slice: ${specs.length} specs for rank ${inShape.length}`)
  const shape: number[] = []
  const strides: number[] = []
  let offset = inOffset
  for (let axis = 0; axis < inShape.length; axis++) {
    const spec = axis < specs.length ? specs[axis] : null
    const n = inShape[axis]
    const stride = inStrides[axis]
    if (typeof spec === 'number') {
      const i = spec < 0 ? spec + n : spec
      if (!Number.isInteger(i) || i < 0 || i >= n) {
        throw new ShapeError('slice', `slice: index ${spec} out of range for axis ${axis} (length ${n})`)
      }
      offset += i * stride
      continue
    }
    const [rawStart, rawStop, rawStep] = spec ?? []
    const step = rawStep ?? 1
    if (!Number.isInteger(step) || step === 0) throw new AifnError('slice', 'slice: step must be a non-zero integer')
    // Python's slice.indices: clamp to [0, n] going forwards and to [-1, n - 1] going backwards.
    const bound = (v: number | null | undefined, fallback: number): number => {
      if (v === null || v === undefined) return fallback
      const i = v < 0 ? v + n : v
      return step > 0 ? Math.min(Math.max(i, 0), n) : Math.min(Math.max(i, -1), n - 1)
    }
    const start = bound(rawStart, step > 0 ? 0 : n - 1)
    const stop = bound(rawStop, step > 0 ? n : -1)
    const length = Math.max(0, Math.ceil((stop - start) / step))
    if (length > 0) offset += start * stride
    shape.push(length)
    strides.push(stride * step)
  }
  return { shape, strides, offset }
}

/**
 * Resolve a target shape with at most one $-1$ entry for `n` elements. Throws `ShapeError` for more than one $-1$, a
 * $-1$ that cannot be filled exactly, an invalid entry, or a shape that does not hold `n` elements.
 *
 * @param n The number of elements the shape must hold.
 * @param shape The target shape; one entry may be $-1$, which is replaced by whatever length makes the sizes agree.
 * @param from The shape being reshaped, shown in error messages.
 * @returns `shape` with its $-1$ filled in.
 */
export function resolveShape(n: number, shape: readonly number[], from: readonly number[]): number[] {
  const unknown = shape.filter((d) => d === -1).length
  if (unknown > 1) throw new ShapeError('reshape', 'reshape: only one dimension may be -1')
  let target = [...shape]
  if (unknown === 1) {
    const known = sizeOf(shape.filter((d) => d !== -1))
    if (known === 0 || n % known !== 0)
      throw new ShapeError('reshape', `reshape: cannot reshape ${showShape(from)} to ${showShape(shape)}`, [
        from,
        shape,
      ])
    target = shape.map((d) => (d === -1 ? n / known : d))
  }
  checkShape(target, 'reshape')
  if (sizeOf(target) !== n) {
    throw new ShapeError(
      'reshape',
      `reshape: cannot reshape ${showShape(from)} (${n} elements) to ${showShape(shape)}`,
      [from, shape],
    )
  }
  return target
}

/**
 * The same elements, in row-major order, with a new shape: a view when the tensor is contiguous or when the new shape
 * only adds or removes axes of length $1$, otherwise a copy.
 *
 * @param t The tensor to reshape; not modified.
 * @param shape The new shape, which may hold one $-1$ (see `resolveShape`).
 * @returns A tensor of the new shape, sharing `t`'s data when it can.
 */
export function reshapeView(t: Tensor, shape: readonly number[]): Tensor {
  const target = resolveShape(size(t), shape, t.shape)
  if (isContiguous(t)) return view(t, target, rowMajorStrides(target), t.offset)
  // Adding or removing axes of length 1 (squeeze, expandDims) keeps any layout a view: the other axes keep their
  // strides, in order, and the new unit axes get stride 0.
  const kept = t.shape.flatMap((d, k) => (d === 1 ? [] : [t.strides[k]]))
  if (kept.length === target.filter((d) => d !== 1).length) {
    let j = 0
    const strides = target.map((d) => (d === 1 ? 0 : kept[j++]))
    const nonUnit = t.shape.filter((d) => d !== 1)
    if (target.filter((d) => d !== 1).every((d, k) => d === nonUnit[k])) return view(t, target, strides, t.offset)
  }
  return fromData(flatData(t), target, t.dtype)
}

/**
 * Check a permutation of the axes of a rank-`rank` tensor and normalise negative entries. Throws `ShapeError` when it
 * has the wrong length, an axis out of range or a repeated axis.
 *
 * @param axes One axis per axis of the tensor, each once; a negative entry counts from the last axis.
 * @param rank The number of axes of the tensor.
 * @returns The permutation with every entry in $0, \dots, r-1$, $r$ the rank.
 */
export function checkPermutation(axes: readonly number[], rank: number): number[] {
  if (axes.length !== rank) throw new ShapeError('permute', `permute: ${axes.length} axes for rank ${rank}`)
  const order = axes.map((a) => normaliseAxis(a, rank, 'permute'))
  if (new Set(order).size !== rank)
    throw new ShapeError('permute', `permute: [${axes.join(', ')}] is not a permutation`)
  return order
}

/**
 * A view with axis $k$ of the result being axis `order[k]` of `t` (no copy).
 *
 * @param t The tensor whose axes are reordered; not modified.
 * @param order A permutation of `t`'s axes, already checked and normalised (as `checkPermutation` returns it).
 * @returns The permuted view.
 */
export function permuteView(t: Tensor, order: readonly number[]): Tensor {
  return view(
    t,
    order.map((a) => t.shape[a]),
    order.map((a) => t.strides[a]),
    t.offset,
  )
}

/**
 * The axes `squeeze` removes: every axis of length $1$, or the listed ones (which must have length $1$, or
 * `ShapeError` is thrown).
 *
 * @param shape The shape of the tensor being squeezed.
 * @param axis The axis or axes to remove (negative entries count from the end), or undefined for every axis of length
 *   $1$.
 * @returns The axes to remove, as non-negative indices.
 */
export function squeezedAxes(shape: readonly number[], axis: Axes | undefined): number[] {
  const drop =
    axis === undefined ? shape.flatMap((d, k) => (d === 1 ? [k] : [])) : normaliseAxes(axis, shape.length, 'squeeze')
  for (const k of drop)
    if (shape[k] !== 1) throw new ShapeError('squeeze', `squeeze: axis ${k} has length ${shape[k]}, not 1`)
  return drop
}

/**
 * Join tensors along an existing axis (a copy, dtype promoted). Throws `ShapeError` for an empty list, scalars, or
 * shapes that differ off the axis.
 *
 * @param ts The tensors to join, in order: the same rank (at least 1), and the same length on every other axis. Not
 *   modified.
 * @param axis The axis to join along (negative counts from the end).
 * @returns A new contiguous tensor whose length on `axis` is the sum of the inputs' lengths, with the promoted dtype
 *   of the inputs (a real input into a complex result gets zero imaginary parts).
 */
export function concatRaw(ts: readonly Tensor[], axis: number): Tensor {
  if (ts.length === 0) throw new ShapeError('concat', 'concat: nothing to join')
  const rank = ts[0].shape.length
  if (rank === 0) throw new ShapeError('concat', 'concat: cannot join scalar tensors; use stack')
  const a = normaliseAxis(axis, rank, 'concat')
  for (const t of ts) {
    if (t.shape.length !== rank || t.shape.some((d, k) => k !== a && d !== ts[0].shape[k])) {
      throw new ShapeError(
        'concat',
        `concat: shape ${showShape(t.shape)} does not match ${showShape(ts[0].shape)} off axis ${a}`,
        [t.shape, ts[0].shape],
      )
    }
  }
  const dtype = ts.map((t) => t.dtype).reduce(promote)
  const shape = [...ts[0].shape]
  shape[a] = ts.reduce((s, t) => s + t.shape[a], 0)
  const out = allocate(dtype, sizeOf(shape))
  const outStrides = rowMajorStrides(shape)
  let start = 0
  for (const t of ts) {
    // Copy t into the block of the output that begins at `start` along axis a (two slots per complex element; a real
    // input into a complex output leaves the imaginary parts zero).
    const src = t.data
    const at = start * outStrides[a]
    if (dtype !== 'complex128')
      forEachOffset2(t.shape, outStrides, at, t.strides, t.offset, (dst, s) => (out[dst] = src[s]))
    else if (t.dtype === 'complex128')
      forEachOffset2(t.shape, outStrides, at, t.strides, t.offset, (dst, s) => {
        out[2 * dst] = src[2 * s]
        out[2 * dst + 1] = src[2 * s + 1]
      })
    else forEachOffset2(t.shape, outStrides, at, t.strides, t.offset, (dst, s) => (out[2 * dst] = src[s]))
    start += t.shape[a]
  }
  return fromData(out, shape, dtype)
}

/**
 * A view of `t` broadcast to `shape` (NumPy rules: `t`'s shape is right-aligned, and each of its axes has the target
 * length or length $1$). Repeated elements share storage through stride $0$. Throws `ShapeError` when `t` does not
 * broadcast to `shape`.
 *
 * @param t The tensor to broadcast; not modified.
 * @param shape The target shape, of at least `t`'s rank.
 * @returns A view of `t` with shape `shape`.
 */
export function broadcastView(t: Tensor, shape: readonly number[]): Tensor {
  const rank = shape.length
  const lead = rank - t.shape.length
  if (lead < 0)
    throw new ShapeError('broadcastTo', `broadcastTo: cannot broadcast ${showShape(t.shape)} to ${showShape(shape)}`, [
      t.shape,
      shape,
    ])
  const strides = new Array<number>(rank).fill(0)
  for (let k = 0; k < t.shape.length; k++) {
    const d = t.shape[k]
    if (d === shape[lead + k]) strides[lead + k] = d === 1 ? 0 : t.strides[k]
    else if (d === 1) strides[lead + k] = 0
    else
      throw new ShapeError(
        'broadcastTo',
        `broadcastTo: cannot broadcast ${showShape(t.shape)} to ${showShape(shape)}`,
        [t.shape, shape],
      )
  }
  return view(t, shape, strides, t.offset)
}

/**
 * The broadcast shape of several shapes (NumPy rules), or a `ShapeError` naming them. The shapes are right-aligned
 * (missing leading axes count as length $1$), and on each axis the lengths must agree or be $1$; the result has the
 * largest rank and, on each axis, the length that is not $1$.
 *
 * @param shapes The shapes to combine, as arrays of axis lengths (`[]` for a scalar). None gives `[]`.
 * @returns The common shape every input broadcasts to.
 *
 * @example A column and a row broadcast to a matrix
 * print('[3, 1] with [4] ->', broadcastShapes([3, 1], [4]))
 * print('[2, 3] with [] ->', broadcastShapes([2, 3], []))
 *
 * @example Incompatible shapes throw
 * try {
 *   broadcastShapes([2, 3], [4])
 * } catch (e) {
 *   print('error:', e.message)
 * }
 */
export function broadcastShapes(...shapes: (readonly number[])[]): number[] {
  const rank = Math.max(0, ...shapes.map((s) => s.length))
  const out = new Array<number>(rank).fill(1)
  for (const s of shapes) {
    for (let k = 0; k < s.length; k++) {
      const j = rank - s.length + k
      const d = s[k]
      if (d === out[j] || d === 1) continue
      if (out[j] === 1) out[j] = d
      else
        throw new ShapeError(
          'broadcast',
          `broadcast: shapes ${shapes.map(showShape).join(' and ')} are incompatible`,
          shapes,
        )
    }
  }
  return out
}
