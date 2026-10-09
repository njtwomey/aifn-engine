/**
 * Internal helpers for the series of a trace: each recorder's values stacked over the kept steps as a contiguous
 * float64 tensor of shape `[kept, ...valueShape]`, row $k$ holding the value recorded at the $k$-th kept step.
 *
 * `flattenRecorded` turns what a recorder returns (a number, a flat or nested array, a tensor) into a shape and
 * row-major values, `makeSeries` wraps a filled buffer as a series, `seriesData` reads one back without copying, and
 * `seriesComponents` splits a series into one line per component for drawing.
 */
import { fromData, isContiguous, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Recorded } from 'aifn-compute/foundation/contracts'

/**
 * A contiguous float64 series of the given shape over `data` (taken over, not copied).
 *
 * @param shape The series' shape, `[kept, ...valueShape]`: the number of kept steps, then the shape of one recorded
 *   value.
 * @param data The values in row-major order, as many as the product of `shape`. It becomes the tensor's storage, so
 *   later writes to it show through.
 * @returns A tensor of shape `shape` viewing `data`.
 */
export function makeSeries(shape: readonly number[], data: Float64Array): Tensor {
  return fromData(data, shape)
}

/**
 * The elements of a series in row-major order as a Float64Array: its own storage when it is contiguous float64 at
 * offset 0 (as every series the runners build is), otherwise a copy, so strided tensors are read correctly.
 *
 * @param series A series tensor, or any tensor: its elements are read in row-major order whatever its strides, offset
 *   or dtype.
 * @returns The elements as float64: the tensor's own storage when no copy is needed (do not modify it), otherwise a new
 *   array.
 */
export function seriesData(series: Tensor): Float64Array {
  const n = series.shape.reduce((a, b) => a * b, 1)
  if (series.dtype === 'float64' && series.offset === 0 && isContiguous(series) && series.data.length === n)
    return series.data as Float64Array
  return Float64Array.from(toFlat(series))
}

/**
 * Flattens one recorded value to its shape and row-major values. Numbers have shape `[]`; flat arrays and typed arrays
 * shape `[n]`; nested arrays must be rectangular; tensors are read in row-major order whatever their strides. Throws
 * `AifnError` for a flat array that mixes numbers and arrays, and `ShapeError` for a ragged nested array.
 *
 * @param value What a recorder returned for one step.
 * @param name The recorder's name, used in error messages.
 * @returns The value's shape (`[]` for a number) and its elements in row-major order.
 */
export function flattenRecorded(value: Recorded, name: string): { shape: number[]; values: number[] } {
  if (typeof value === 'number') return { shape: [], values: [value] }
  if (isTensor(value)) return { shape: [...value.shape], values: toFlat(value) }
  const list = value as ArrayLike<unknown>
  if (list.length === 0 || typeof list[0] === 'number') {
    const values = Array.from(list as ArrayLike<number>)
    if (values.some((v) => typeof v !== 'number'))
      throw new AifnError('trace', `trace: recorder "${name}" mixes numbers and arrays`)
    return { shape: [values.length], values }
  }
  const parts = Array.from(list as ArrayLike<Recorded>, (v) => flattenRecorded(v, name))
  const inner = parts[0].shape
  for (const p of parts) {
    if (p.shape.length !== inner.length || p.shape.some((d, i) => d !== inner[i]))
      throw new ShapeError('trace', `trace: recorder "${name}" returned a ragged nested array`)
  }
  return { shape: [parts.length, ...inner], values: parts.flatMap((p) => p.values) }
}

/**
 * The components of a series as separate lines against the kept steps, for drawing: a series of shape `[kept]` gives
 * one component (labelled with the empty string), `[kept, m]` gives $m$ labelled `[i]`, and `[kept, a, b]` gives
 * $ab$ labelled `[i, j]`, in row-major order.
 *
 * @param series A series of a trace (`trace.series[name]`), of shape `[kept, ...valueShape]`; any strides are read
 *   correctly.
 * @returns One entry per component of the recorded value: its `label` (its index in the value shape) and its `values`
 *   at each kept step (a new array of length `kept`).
 *
 * @example One line per coordinate of a recorded point
 * // A point rotated by a quarter turn each step, recorded as [x, y].
 * const quarterTurn = {
 *   name: 'quarter-turn',
 *   init: () => ({ t: 0, p: [1, 0] }),
 *   step: (s) => ({ t: s.t + 1, p: [-s.p[1], s.p[0]] }),
 * }
 * const tr = trace(quarterTurn, undefined, 4, { record: { p: (s) => s.p } })
 * for (const { label, values } of seriesComponents(tr.series.p)) print(label, values)
 */
export function seriesComponents(series: Tensor): { label: string; values: Float64Array }[] {
  const [kept, ...valueShape] = series.shape
  const width = valueShape.reduce((a, b) => a * b, 1)
  const data = seriesData(series)
  const out: { label: string; values: Float64Array }[] = []
  for (let c = 0; c < width; c++) {
    const values = new Float64Array(kept)
    for (let r = 0; r < kept; r++) values[r] = data[r * width + c]
    // Unravel c into the value shape for the label.
    const at: number[] = []
    let rest = c
    for (let d = valueShape.length - 1; d >= 0; d--) {
      at.unshift(rest % valueShape[d])
      rest = Math.floor(rest / valueShape[d])
    }
    out.push({ label: at.length ? `[${at.join(', ')}]` : '', values })
  }
  return out
}
