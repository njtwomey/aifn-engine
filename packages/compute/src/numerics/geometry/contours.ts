/**
 * Contour lines of a field sampled on a rectangular grid, by marching squares (Lorensen and Cline, 1987, "Marching
 * cubes", SIGGRAPH, in its 2-D form), with saddles resolved by the cell-centre average, and joining of the segments
 * into polylines.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

/** A grid of values [ny, nx] (a matrix or rows), with `z[i][j]` the value at (x[j], y[i]), checked against the axes. */
function readGrid(z: MatrixLike, nx: number, ny: number): Float64Array {
  return dense.toMatrixF64(z, 'contours', ny, nx).data
}

const axis = (v: VectorLike) => Array.from(dense.toF64(v, 'contours'))

/**
 * The segments where a field crosses `level`, as an s × 2 × 2 tensor (segment, end, coordinate). Each grid square whose
 * corners straddle the level contributes one segment, or two at a saddle; ends are placed by linear interpolation
 * along the square's edges. A corner exactly at the level counts as below it.
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
 * The contour at `level` as polylines: segments that share an end point (to within a relative 1e-9 of the grid's
 * extent) are joined. Each line is an m × 2 tensor; a closed line repeats its first point at the end.
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
 * `count` evenly spaced levels strictly inside the range of the finite values of z (the ends are excluded, since a
 * contour at the minimum or maximum is empty or a point).
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
