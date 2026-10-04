import { fromData, isContiguous, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { AifnError, ShapeError } from 'aifn-compute/foundation/errors'
import type { Recorded } from 'aifn-compute/foundation/contracts'

/** A contiguous float64 series of the given shape over `data` (taken over, not copied). */
export function makeSeries(shape: readonly number[], data: Float64Array): Tensor {
  return fromData(data, shape)
}

/**
 * The elements of a series in row-major order as a Float64Array: its own storage when it is contiguous float64 at
 * offset 0 (as every series the runners build is), otherwise a copy, so strided tensors are read correctly.
 */
export function seriesData(series: Tensor): Float64Array {
  const n = series.shape.reduce((a, b) => a * b, 1)
  if (series.dtype === 'float64' && series.offset === 0 && isContiguous(series) && series.data.length === n)
    return series.data as Float64Array
  return Float64Array.from(toFlat(series))
}

/**
 * Flattens one recorded value to its shape and row-major values. Numbers have shape `[]`; flat arrays and typed arrays
 * shape `[n]`; nested arrays must be rectangular; tensors are read in row-major order whatever their strides.
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
 * The components of a series as separate lines against the kept steps: a series of shape `[kept]` gives one
 * component, `[kept, m]` gives m, and `[kept, a, b]` gives a·b labelled `[i, j]`. For drawing.
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
