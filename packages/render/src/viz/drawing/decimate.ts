/** Decimation of long lines for drawing (presentation code, moved from aifn to the lab with the module tree). */

import { fromData, isTensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const axis = (v: Tensor | ArrayLike<number>) => (isTensor(v) ? toFlat(v) : Array.from(v))

/** A decimated line: the kept points and their indices in the original. */
export interface Decimated {
  x: Tensor
  y: Tensor
  /** Indices of the kept points, ascending; int32. */
  index: Tensor
}

/**
 * Largest-triangle-three-buckets downsampling (Steinarsson, 2013, "Downsampling time series for visual
 * representation", MSc thesis, University of Iceland, §4.2): keeps the first and last points and, from each of
 * `threshold − 2` buckets, the point forming the largest triangle with the previously kept point and the next
 * bucket's mean. Preserves the visual shape of a line far better than taking every k-th point. x must be ascending.
 */
export function lttb(x: Tensor | ArrayLike<number>, y: Tensor | ArrayLike<number>, threshold: number): Decimated {
  const xs = axis(x)
  const ys = axis(y)
  const n = xs.length
  if (ys.length !== n) throw new Error('lttb: x and y must have the same length')
  const keep: number[] = []
  if (threshold >= n || threshold < 3) for (let i = 0; i < n; i++) keep.push(i)
  else {
    const every = (n - 2) / (threshold - 2)
    let a = 0
    keep.push(0)
    for (let b = 0; b < threshold - 2; b++) {
      // The next bucket's average is the third vertex of the triangle.
      const nextStart = Math.floor((b + 1) * every) + 1
      const nextEnd = Math.min(Math.floor((b + 2) * every) + 1, n)
      let ax = 0
      let ay = 0
      for (let i = nextStart; i < nextEnd; i++) {
        ax += xs[i]
        ay += ys[i]
      }
      const m = nextEnd - nextStart
      ax /= m
      ay /= m
      const start = Math.floor(b * every) + 1
      const end = Math.floor((b + 1) * every) + 1
      let best = start
      let bestArea = -1
      for (let i = start; i < end; i++) {
        const area = Math.abs((xs[a] - ax) * (ys[i] - ys[a]) - (xs[a] - xs[i]) * (ay - ys[a]))
        if (area > bestArea) {
          bestArea = area
          best = i
        }
      }
      keep.push(best)
      a = best
    }
    keep.push(n - 1)
  }
  return {
    x: fromData(Float64Array.from(keep, (i) => xs[i])),
    y: fromData(Float64Array.from(keep, (i) => ys[i])),
    index: fromData(Int32Array.from(keep)),
  }
}

/**
 * Min–max decimation: split the line into `buckets` equal index ranges and keep each range's minimum and maximum (in
 * index order), plus the first and last points. Every extreme survives, so spikes are never lost; the result has at
 * most 2·buckets + 2 points.
 */
export function minMaxDecimate(
  x: Tensor | ArrayLike<number>,
  y: Tensor | ArrayLike<number>,
  buckets: number,
): Decimated {
  const xs = axis(x)
  const ys = axis(y)
  const n = xs.length
  const keep = new Set<number>([0, n - 1])
  if (2 * buckets + 2 >= n) for (let i = 0; i < n; i++) keep.add(i)
  else
    for (let b = 0; b < buckets; b++) {
      const start = Math.floor((b * n) / buckets)
      const end = Math.floor(((b + 1) * n) / buckets)
      let lo = start
      let hi = start
      for (let i = start; i < end; i++) {
        if (ys[i] < ys[lo]) lo = i
        if (ys[i] > ys[hi]) hi = i
      }
      keep.add(lo)
      keep.add(hi)
    }
  const index = [...keep].filter((i) => i >= 0 && i < n).sort((a, b) => a - b)
  return {
    x: fromData(Float64Array.from(index, (i) => xs[i])),
    y: fromData(Float64Array.from(index, (i) => ys[i])),
    index: fromData(Int32Array.from(index)),
  }
}
