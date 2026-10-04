/**
 * Raw indexing, views and shape manipulation on plain tensors (no tracing). The public, differentiable versions in
 * `structure.ts` call these. Views share data; functions that must copy say so.
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

/** Row-major position of a multi-index (negative entries count from the end), checked against the shape. */
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

/** A view selecting part of a tensor, as NumPy basic indexing. */
export function sliceView(t: Tensor, specs: readonly SliceSpec[]): Tensor {
  const g = sliceGeometry(t.shape, t.strides, t.offset, specs)
  return view(t, g.shape, g.strides, g.offset)
}

/** The shape of a slice of a tensor of shape `shape` (for shape rules). */
export function sliceShape(shape: readonly number[], specs: readonly SliceSpec[]): number[] {
  return sliceGeometry(shape, new Array<number>(shape.length).fill(0), 0, specs).shape
}

/** The shape, strides and offset of a slice of a layout. */
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

/** Resolve a target shape with at most one -1 entry for `n` elements. */
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

/** The same elements with a new shape: a view when the tensor is contiguous, otherwise a copy. */
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

/** Check a permutation of the axes of a rank-`rank` tensor and normalise negative entries. */
export function checkPermutation(axes: readonly number[], rank: number): number[] {
  if (axes.length !== rank) throw new ShapeError('permute', `permute: ${axes.length} axes for rank ${rank}`)
  const order = axes.map((a) => normaliseAxis(a, rank, 'permute'))
  if (new Set(order).size !== rank)
    throw new ShapeError('permute', `permute: [${axes.join(', ')}] is not a permutation`)
  return order
}

/** A view with axis k of the result being axis `order[k]` of `t`. */
export function permuteView(t: Tensor, order: readonly number[]): Tensor {
  return view(
    t,
    order.map((a) => t.shape[a]),
    order.map((a) => t.strides[a]),
    t.offset,
  )
}

/** The axes `squeeze` removes: every axis of length 1, or the listed ones (which must have length 1). */
export function squeezedAxes(shape: readonly number[], axis: Axes | undefined): number[] {
  const drop =
    axis === undefined ? shape.flatMap((d, k) => (d === 1 ? [k] : [])) : normaliseAxes(axis, shape.length, 'squeeze')
  for (const k of drop)
    if (shape[k] !== 1) throw new ShapeError('squeeze', `squeeze: axis ${k} has length ${shape[k]}, not 1`)
  return drop
}

/** Join tensors along an existing axis (a copy, dtype promoted). */
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
 * length or length 1). Repeated elements share storage through stride 0.
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

/** The broadcast shape of several shapes (NumPy rules), or an error naming them. */
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
