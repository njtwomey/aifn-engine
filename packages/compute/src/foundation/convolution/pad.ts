/**
 * Padding: a movement primitive that extends each axis by `lo` samples before and `hi` after, the new samples read by
 * a border mode.
 *
 * Every mode is a fixed map from output positions to input positions (or to zero), so the padding is linear in its
 * input; its transpose `padAdjoint` adds each output cotangent back onto the input position it was read from, and the
 * two are each other's transposes, so `pad` differentiates to any order and batches without a loop. The border modes
 * and the forms of the widths are those of `numpy.pad`. Real values only for now.
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
 * How samples beyond an edge are read, with `numpy.pad`'s names, shown for the signal `a b c d` padded by two on each
 * side: `constant` (`0 0 | a b c d | 0 0`, or the fill value), `reflect` (`c b | a b c d | c b`, the edge sample not
 * repeated), `symmetric` (`b a | a b c d | d c`, the edge repeated), `edge` (`a a | a b c d | d d`) and `wrap`
 * (`c d | a b c d | a b`, periodic).
 */
export type PadMode = 'constant' | 'reflect' | 'symmetric' | 'edge' | 'wrap'

/**
 * Padding widths, as `numpy.pad`'s `pad_width`: one number for every side of every axis, a `[lo, hi]` pair of numbers
 * for every axis, or a list with one entry (a number or a `[lo, hi]` pair) per trailing axis, the leading axes left
 * unpadded. A list of exactly two numbers is read as one `[lo, hi]` pair for every axis, not as one width per axis.
 * Widths are non-negative integers.
 */
export type PadWidths = number | readonly [number, number] | readonly (number | readonly [number, number])[]

/** Parameters of the `pad` primitive and its adjoint. */
type PadParams = {
  /** One `[lo, hi]` pair per axis of the input: the samples added before and after it. */
  readonly widths: readonly (readonly [number, number])[]
  /** The border mode the added samples are read by. */
  readonly mode: PadMode
  /** The input shape (for the adjoint, whose output it is). */
  readonly shape: readonly number[]
}

/**
 * The input position that an output position of one padded axis reads under a border mode.
 *
 * @param j The output position on the padded axis, from 0 to $n + \mathrm{lo} + \mathrm{hi} - 1$.
 * @param n The length of the axis before padding.
 * @param lo The number of samples added before the axis, so that output position `lo` reads input position 0.
 * @param mode The border mode that decides which input position a position beyond an edge reads.
 * @returns The input position, from 0 to $n - 1$, or $-1$ when the position is a constant (`constant` mode beyond an
 *   edge).
 */
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

/**
 * The shape of a value after padding.
 *
 * @param shape The shape before padding.
 * @param widths One `[lo, hi]` pair per axis of `shape`.
 * @returns Each axis length $n$ grown to $n + \mathrm{lo} + \mathrm{hi}$.
 */
const paddedShape = (shape: readonly number[], widths: PadParams['widths']): number[] =>
  shape.map((n, d) => n + widths[d][0] + widths[d][1])

/**
 * Visit every (output offset, input offset) pair of a padding, in row-major order of the output; the input offset is
 * $-1$ for a constant zero. Per-axis maps are precomputed, and the last axis is the inner loop.
 *
 * @param p The padding: the input shape, the widths per axis and the border mode.
 * @param visit Called once per element of the padded output with its row-major offset `out` and the row-major offset
 *   `src` of the input element it reads, or $-1$ when it reads none.
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

/**
 * The number of elements of a shape.
 *
 * @param shape The axis lengths.
 * @returns Their product (1 for a scalar).
 */
const size = (shape: readonly number[]) => shape.reduce((a, b) => a * b, 1)

/**
 * The tensor a primitive's impl was given, refusing a plain number.
 *
 * @param x The raw input of the impl.
 * @param what The caller's name for error messages.
 * @returns `x`, known to be a tensor. Throws `ShapeError` for a number.
 */
function tensorOf(x: Raw, what: string): Tensor {
  if (typeof x === 'number') throw new ShapeError(what, `${what}: expected a tensor, got a number`)
  return x
}

/**
 * The padding primitive: copies each output element from the input position its border mode reads (zero where it reads
 * none). Linear, with `padAdjoint` as its transpose; batched by inserting an unpadded axis.
 */
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

/**
 * The parameters of a padding with an unpadded batch axis inserted, for batching the primitives.
 *
 * @param p The parameters of the padding of one example.
 * @param axis Where the batch axis sits in the batched value.
 * @param n The length of the batch axis.
 * @returns `p` with widths `[0, 0]` and length `n` inserted at `axis`.
 */
function batchedParams(p: PadParams, axis: number, n: number): PadParams {
  const widths = [...p.widths]
  const shape = [...p.shape]
  widths.splice(axis, 0, [0, 0])
  shape.splice(axis, 0, n)
  return { ...p, widths, shape }
}

/**
 * Widths as one `[lo, hi]` pair per axis of a value. Throws `ShapeError` for a list longer than the rank, or for a
 * width that is not a non-negative integer.
 *
 * @param widths The widths in any form `PadWidths` allows. A list shorter than the rank pads the trailing axes, and a
 *   list of exactly two numbers is one `[lo, hi]` pair for every axis.
 * @param rank The number of axes of the value being padded.
 * @returns One `[lo, hi]` pair per axis, leading axes not covered by a list given `[0, 0]`.
 */
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
 * Pad `x` by `widths` with a border mode, as `numpy.pad`. Linear in `x` for every mode, so differentiable to any order
 * and batched by `vmap`; a nonzero `value` makes the constant mode affine (zeros are padded, then `value` added on the
 * border). Throws `ShapeError` for invalid widths, or for a non-constant mode asked to pad an empty axis. Real values
 * only.
 *
 * @param x The value to pad, of any rank.
 * @param widths How many samples to add before and after each axis, in any form of `PadWidths`: one number for every
 *   side, one `[lo, hi]` pair for every axis, or a list per trailing axis.
 * @param mode How the added samples are read from `x` (see `PadMode`).
 * @param options Options of the constant mode.
 * @param options.value The value of the added samples in `constant` mode; ignored by the other modes.
 * @returns `x` with each axis of length $n$ grown to $n + \mathrm{lo} + \mathrm{hi}$.
 *
 * @example The border modes on a short signal
 * const x = tensor([1, 2, 3, 4])
 * print('constant ', pad(x, 2))
 * print('reflect  ', pad(x, 2, 'reflect'))
 * print('symmetric', pad(x, 2, 'symmetric'))
 * print('edge     ', pad(x, 2, 'edge'))
 * print('wrap     ', pad(x, 2, 'wrap'))
 *
 * @example Pad only the last axis of a matrix, with a fill value
 * const m = tensor([[1, 2], [3, 4]])
 * print('padded =', pad(m, [[0, 1]], 'constant', { value: 9 }))
 *
 * @example The gradient adds each border read back onto its source
 * // Edge padding reads the end samples again for every sample it adds.
 * print('grad =', grad((x) => sum(pad(x, 2, 'edge')))(tensor([1, 2, 3])))
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
