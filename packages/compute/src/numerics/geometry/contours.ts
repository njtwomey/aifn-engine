/**
 * Contour lines of a field sampled on a rectangular grid, by marching squares (Lorensen and Cline, 1987, "Marching
 * cubes", SIGGRAPH, in its 2-D form), with saddles resolved by the cell-centre average, and joining of the segments
 * into polylines.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

/** A grid of values [ny, nx] (a matrix or rows), with `z[i][j]` the value at (x[j], y[i]), checked against the axes.
 *
 * @param z - 2D matrix-like data of function values.
 * @param nx - Expected number of grid columns along the x-axis.
 * @param ny - Expected number of grid rows along the y-axis.
 * @returns Flattened 64-bit float array of shape $[ny \cdot nx]$.
 */
function readGrid(z: MatrixLike, nx: number, ny: number): Float64Array {
  return dense.toMatrixF64(z, 'contours', ny, nx).data
}

/**
 * Convert a vector-like axis representation to a JavaScript array of 64-bit floats.
 *
 * @param v - Vector-like axis coordinates.
 * @returns Array of numbers representing grid coordinates.
 */
const axis = (v: VectorLike) => Array.from(dense.toF64(v, 'contours'))

/**
 * Compute the line segments where a scalar field crosses `level`, represented as an $s \times 2 \times 2$ tensor
 * (segment, endpoint, $(x, y)$ coordinate).
 *
 * Each grid square whose corners straddle `level` contributes one segment, or two segments in the case of a saddle;
 * endpoints are placed by linear interpolation along the square's edges. A corner value exactly at `level` is treated
 * as strictly below it.
 *
 * @param x - Grid coordinates along the horizontal axis, length $nx$.
 * @param y - Grid coordinates along the vertical axis, length $ny$.
 * @param z - 2D field values on the grid, shape $[ny, nx]$.
 * @param level - Iso-contour threshold value to extract.
 * @returns Tensor of line segments of shape $[s, 2, 2]$.
 * @example Extracting contour segments
 * const x = [0, 1, 2]
 * const y = [0, 1, 2]
 * const z = [
 *   [0, 1, 2],
 *   [1, 2, 3],
 *   [2, 3, 4],
 * ]
 * const segs = contourSegments(x, y, z, 1.5)
 * print('segments shape =', segs.shape)
 */
export function contourSegments(x: VectorLike, y: VectorLike, z: MatrixLike, level: number): Tensor {
  const xs = axis(x)
  const ys = axis(y)
  const nx = xs.length
  const g = readGrid(z, nx, ys.length)
  const out: number[] = []
  const cross = (xa: number, ya: number, za: number, xb: number, yb: number, zb: number) => {
    const f = za === zb ? 0.5 : (level - za) / (zb - za)
    return [xa + f * (xb - xa), ya + f * (yb - ya)]
  }
  const push = (a: number[], b: number[]) => out.push(a[0], a[1], b[0], b[1])
  for (let i = 0; i + 1 < ys.length; i++)
    for (let j = 0; j + 1 < nx; j++) {
      // Corners anticlockwise from bottom-left: (j, i), (j+1, i), (j+1, i+1), (j, i+1).
      const [x0, x1, y0, y1] = [xs[j], xs[j + 1], ys[i], ys[i + 1]]
      const [a, b, c, d] = [g[i * nx + j], g[i * nx + j + 1], g[(i + 1) * nx + j + 1], g[(i + 1) * nx + j]]
      if (![a, b, c, d].every(Number.isFinite)) continue
      const index = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (d > level ? 8 : 0)
      if (index === 0 || index === 15) continue
      const bottom = () => cross(x0, y0, a, x1, y0, b)
      const right = () => cross(x1, y0, b, x1, y1, c)
      const top = () => cross(x0, y1, d, x1, y1, c)
      const left = () => cross(x0, y0, a, x0, y1, d)
      const centreAbove = (a + b + c + d) / 4 > level
      switch (index) {
        case 1:
        case 14:
          push(left(), bottom())
          break
        case 2:
        case 13:
          push(bottom(), right())
          break
        case 3:
        case 12:
          push(left(), right())
          break
        case 4:
        case 11:
          push(right(), top())
          break
        case 6:
        case 9:
          push(bottom(), top())
          break
        case 7:
        case 8:
          push(left(), top())
          break
        case 5:
          // a and c above: joined through the centre when it is above too.
          if (centreAbove) {
            push(left(), top())
            push(bottom(), right())
          } else {
            push(left(), bottom())
            push(right(), top())
          }
          break
        case 10:
          if (centreAbove) {
            push(left(), bottom())
            push(right(), top())
          } else {
            push(left(), top())
            push(bottom(), right())
          }
          break
      }
    }
  return fromData(Float64Array.from(out), [out.length / 4, 2, 2])
}

/**
 * Trace the iso-contour at `level` as a sequence of connected polylines.
 *
 * Segments that share an endpoint (within a relative tolerance of $10^{-9}$ times the grid span) are joined into
 * continuous paths. Each line is an $m \times 2$ tensor; a closed loop repeats its initial point at the end.
 *
 * @param x - Grid coordinates along the horizontal axis, length $nx$.
 * @param y - Grid coordinates along the vertical axis, length $ny$.
 * @param z - 2D field values on the grid, shape $[ny, nx]$.
 * @param level - Iso-contour threshold value to trace.
 * @returns Array of polyline tensors, each of shape $[m, 2]$.
 * @example Tracing contour polylines
 * const x = [0, 1, 2]
 * const y = [0, 1, 2]
 * const z = [
 *   [0, 1, 2],
 *   [1, 2, 3],
 *   [2, 3, 4],
 * ]
 * const lines = contourLines(x, y, z, 1.5)
 * print('number of lines =', lines.length)
 */
export function contourLines(x: VectorLike, y: VectorLike, z: MatrixLike, level: number): Tensor[] {
  const segs = contourSegments(x, y, z, level).data as Float64Array
  const count = segs.length / 4
  const xs = axis(x)
  const ys = axis(y)
  const scale = Math.max(Math.abs(xs[xs.length - 1] - xs[0]), Math.abs(ys[ys.length - 1] - ys[0]), 1e-300) * 1e-9
  const key = (px: number, py: number) => `${Math.round(px / scale)},${Math.round(py / scale)}`
  // Every end point, keyed by its rounded position, lists the segment ends there.
  const ends = new Map<string, number[]>()
  for (let s = 0; s < count; s++)
    for (let e = 0; e < 2; e++) {
      const k = key(segs[4 * s + 2 * e], segs[4 * s + 2 * e + 1])
      const list = ends.get(k)
      if (list) list.push(2 * s + e)
      else ends.set(k, [2 * s + e])
    }
  const used = new Uint8Array(count)
  // Zero-length segments (a level passing exactly through a grid point) would branch the walk; drop them.
  for (let s = 0; s < count; s++)
    if (segs[4 * s] === segs[4 * s + 2] && segs[4 * s + 1] === segs[4 * s + 3]) used[s] = 1
  const pointOf = (end: number) => [segs[2 * end], segs[2 * end + 1]]
  const nextEnd = (end: number) => {
    const [px, py] = pointOf(end)
    for (const other of ends.get(key(px, py)) ?? []) if (!used[other >> 1]) return other
    return -1
  }
  const lines: Tensor[] = []
  for (let s = 0; s < count; s++) {
    if (used[s]) continue
    used[s] = 1
    // Walk forward from the segment's second end, then backward from its first.
    const forward: number[][] = [pointOf(2 * s), pointOf(2 * s + 1)]
    for (let end = nextEnd(2 * s + 1); end >= 0; end = nextEnd(end ^ 1)) {
      used[end >> 1] = 1
      forward.push(pointOf(end ^ 1))
    }
    const backward: number[][] = []
    for (let end = nextEnd(2 * s); end >= 0; end = nextEnd(end ^ 1)) {
      used[end >> 1] = 1
      backward.push(pointOf(end ^ 1))
    }
    const line = [...backward.reverse(), ...forward]
    lines.push(fromData(Float64Array.from(line.flat()), [line.length, 2]))
  }
  return lines
}

/**
 * Generate `count` evenly spaced iso-contour levels strictly inside the range of finite values of `z`.
 * The minimum and maximum bounds are excluded because contours at the extrema are either empty or degenerate points.
 *
 * @param z - 2D matrix-like grid of scalar values.
 * @param count - Number of interior iso-levels to generate (default 8).
 * @returns 1D tensor of contour levels of length `count`.
 * @example Generating contour levels
 * const z = [
 *   [0, 1, 2],
 *   [1, 2, 3],
 *   [2, 3, 4],
 * ]
 * const levels = contourLevels(z, 3)
 * print('levels =', levels)
 */
export function contourLevels(z: MatrixLike, count = 8): Tensor {
  const v = dense.toMatrixF64(z, 'contourLevels').data
  let lo = Infinity
  let hi = -Infinity
  for (const u of v)
    if (Number.isFinite(u)) {
      lo = Math.min(lo, u)
      hi = Math.max(hi, u)
    }
  return fromData(Float64Array.from({ length: count }, (_, k) => lo + ((hi - lo) * (k + 1)) / (count + 1)))
}
