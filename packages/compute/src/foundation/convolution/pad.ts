/**
 * Padding, part of `aifn-compute/foundation/convolution`: a movement primitive that extends each axis by `lo` samples before
 * and `hi` after, read by a border mode. Every mode is a fixed map from output positions to input positions (or to
 * zero), so `pad` is linear; its transpose `padAdjoint` adds each output cotangent back onto the input position it was
 * read from, and the two are each other's transposes.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  add,
  definePrimitive,
  dense,
  fromData,
  mul,
  refuseComplex,
  shapeOfValue,
  type Op,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'

/**
 * How samples beyond an edge are read, with numpy.pad's names (for a b c d):
 * `constant` (0 0 | a b c d | 0 0), `reflect` (c b | a b c d | c b, the edge sample not repeated),
 * `symmetric` (b a | a b c d | d c, the edge repeated), `edge` (a a | a b c d | d d) and `wrap` (c d | a b c d | a b).
 */
export type PadMode = 'constant' | 'reflect' | 'symmetric' | 'edge' | 'wrap'

/**
 * Padding widths, as numpy.pad's `pad_width`: one number for every side of every axis, a [lo, hi] pair of numbers for
 * every axis, or a list with one entry (a number or a [lo, hi] pair) per trailing axis.
 */
export type PadWidths = number | readonly [number, number] | readonly (number | readonly [number, number])[]

type PadParams = {
  /** [lo, hi] per axis of the input. */
  readonly widths: readonly (readonly [number, number])[]
  readonly mode: PadMode
  /** The input shape (for the adjoint, whose output it is). */
  readonly shape: readonly number[]
}

/** The input position that output position `j` (of an axis padded by `lo`) reads, or −1 for a constant zero. */
export function sourceIndex(j: number, n: number, lo: number, mode: PadMode): number {
  const i = j - lo
  if (i >= 0 && i < n) return i
  switch (mode) {
    case 'constant':
      return -1
    case 'edge':
      return i < 0 ? 0 : n - 1
    case 'wrap':
      return ((i % n) + n) % n
    case 'symmetric': {
      const period = 2 * n
      const m = ((i % period) + period) % period
      return m < n ? m : period - 1 - m
    }
    case 'reflect': {
      if (n === 1) return 0
      const period = 2 * n - 2
      const m = ((i % period) + period) % period
      return m < n ? m : period - m
    }
  }
}

/** The padded shape. */
const paddedShape = (shape: readonly number[], widths: PadParams['widths']): number[] =>
  shape.map((n, d) => n + widths[d][0] + widths[d][1])

/**
 * Visit every (output offset, input offset) pair of a padding; input offset −1 for a constant zero. Per-axis maps are
 * precomputed, and the last axis is the inner loop.
 */
function padLoop(p: PadParams, visit: (out: number, src: number) => void): void {
  const rank = p.shape.length
  const out = paddedShape(p.shape, p.widths)
  const strides = new Array<number>(rank)
  for (let d = rank - 1, s = 1; d >= 0; d--) {
    strides[d] = s
    s *= p.shape[d]
  }
  // maps[d][j] = input offset contribution of output index j on axis d, or −1.
  const maps = out.map((m, d) =>
    Int32Array.from({ length: m }, (_, j) => {
      const i = sourceIndex(j, p.shape[d], p.widths[d][0], p.mode)
      return i < 0 ? -1 : i * strides[d]
    }),
  )
  if (rank === 0) {
    visit(0, 0)
    return
  }
  let o = 0
  const walk = (d: number, base: number): void => {
    const map = maps[d]
    if (d === rank - 1) {
      for (let j = 0; j < map.length; j++, o++) visit(o, base < 0 || map[j] < 0 ? -1 : base + map[j])
      return
    }
    for (let j = 0; j < map.length; j++) walk(d + 1, base < 0 || map[j] < 0 ? -1 : base + map[j])
  }
  walk(0, 0)
}

const size = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)

function tensorOf(x: Raw, what: string): Tensor {
  if (typeof x === 'number') throw new ShapeError(what, `${what}: expected a tensor, got a number`)
  return x
}

const padOp: Op<PadParams> = definePrimitive<PadParams>({
  id: 'foundation/convolution/pad',
  arity: 1,
  impl: ([x], p) => {
    const xd = dense.data(tensorOf(x, 'pad'))
    const shape = paddedShape(p.shape, p.widths)
    const y = new Float64Array(size(shape))
    padLoop(p, (o, s) => {
      if (s >= 0) y[o] = xd[s]
    })
    return fromData(y, shape)
  },
  linear: 'linear',
  transpose: (ct, _inputs, _which, p) => padAdjointOp([ct], p),
  shape: (_avals, p) => ({ shape: paddedShape(p.shape, p.widths), dtype: 'float64', number: false }),
  batch: ([x], [b], p, n) => [padOp([x], batchedParams(p, b ?? 0, n)), b ?? 0],
  dtype: 'float',
  doc: { summary: 'Pad each axis by [lo, hi] samples read by a border mode.' },
  test: {
    secondOrder: true,
    cases: (draw) =>
      (['constant', 'reflect', 'symmetric', 'edge', 'wrap'] as const).map((mode) => ({
        inputs: [draw([3, 4])],
        params: { widths: [[2, 1] as const, [3, 5] as const], mode, shape: [3, 4] },
      })),
  },
})

// The transpose of pad: each output cotangent is added onto the input position it was read from.
const padAdjointOp: Op<PadParams> = definePrimitive<PadParams>({
  id: 'foundation/convolution/padAdjoint',
  arity: 1,
  impl: ([g], p) => {
    const gd = dense.data(tensorOf(g, 'padAdjoint'))
    const gx = new Float64Array(size(p.shape))
    padLoop(p, (o, s) => {
      if (s >= 0) gx[s] += gd[o]
    })
    return fromData(gx, [...p.shape])
  },
  linear: 'linear',
  transpose: (ct, _inputs, _which, p) => padOp([ct], p),
  shape: (_avals, p) => ({ shape: [...p.shape], dtype: 'float64', number: false }),
  batch: ([g], [b], p, n) => [padAdjointOp([g], batchedParams(p, b ?? 0, n)), b ?? 0],
  dtype: 'float',
  doc: { summary: 'The adjoint of pad: cotangents added back onto the positions they were read from.' },
  test: {
    secondOrder: true,
    cases: (draw) =>
      (['constant', 'reflect', 'symmetric', 'edge', 'wrap'] as const).map((mode) => ({
        inputs: [draw([6, 12])],
        params: { widths: [[2, 1] as const, [3, 5] as const], mode, shape: [3, 4] },
      })),
  },
})

/** Params with an unpadded batch axis of length `n` inserted at `axis`. */
function batchedParams(p: PadParams, axis: number, n: number): PadParams {
  const widths = [...p.widths]
  const shape = [...p.shape]
  widths.splice(axis, 0, [0, 0])
  shape.splice(axis, 0, n)
  return { ...p, widths, shape }
}

/** Widths as one [lo, hi] pair per axis of a rank-`rank` value (a shorter list pads the trailing axes). */
export function padWidths(widths: PadWidths, rank: number): [number, number][] {
  let pairs: [number, number][]
  if (typeof widths === 'number') pairs = Array.from({ length: rank }, () => [widths, widths])
  else if (widths.length === 2 && typeof widths[0] === 'number' && typeof widths[1] === 'number')
    pairs = Array.from({ length: rank }, () => [widths[0] as number, widths[1] as number])
  else {
    const list = (widths as readonly (number | readonly [number, number])[]).map((w): [number, number] =>
      typeof w === 'number' ? [w, w] : [w[0], w[1]],
    )
    if (list.length > rank) throw new ShapeError('pad', `pad: ${list.length} widths for a rank-${rank} value`)
    pairs = [...Array.from({ length: rank - list.length }, (): [number, number] => [0, 0]), ...list]
  }
  for (const [lo, hi] of pairs)
    if (!(Number.isInteger(lo) && Number.isInteger(hi) && lo >= 0 && hi >= 0))
      throw new ShapeError('pad', 'pad: widths must be non-negative integers')
  return pairs
}

/**
 * Pad `x` by `widths` (see `PadWidths`) with a border mode, as `numpy.pad`. Linear in x for every mode, so differentiable to any order; `value` fills the constant mode (default 0).
 */
export function pad(x: Value, widths: PadWidths, mode: PadMode = 'constant', { value = 0 } = {}): Value {
  refuseComplex('pad', x, 'is real-only for now')
  const shape = shapeOfValue(x)
  const p: PadParams = { widths: padWidths(widths, shape.length), mode, shape }
  for (let d = 0; d < shape.length; d++)
    if (shape[d] === 0 && mode !== 'constant' && p.widths[d][0] + p.widths[d][1] > 0)
      throw new ShapeError('pad', `pad: cannot ${mode}-pad an empty axis`)
  const y = padOp([x], p)
  if (mode !== 'constant' || value === 0) return y
  // A nonzero fill is affine: pad with zeros, then add value on the border (a constant mask).
  const mask = padOp([fromData(new Float64Array(size(shape)).fill(1), shape)], p) as Tensor
  const border = dense.data(mask).map((v) => 1 - v)
  return add(y, mul(fromData(border, mask.shape), value))
}
