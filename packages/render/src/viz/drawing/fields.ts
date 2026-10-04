/** Sampling vector and scalar fields on grids for drawing: direction and slope fields, contours, level sets and nullclines (presentation code, moved from aifn to the lab with the module tree). */

import { fromData, tensor, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import { dense } from 'aifn-compute/foundation/tensor'
import type { VectorLike } from 'aifn-compute/foundation/contracts'
import { gridAxes, sampleScalar, type Grid2, type ScalarField, type VectorField } from 'aifn-compute/dynamics/fields'

type F64 = Float64Array<ArrayBuffer>
const toF64 = (v: unknown, where: string): F64 => dense.toF64(v as VectorLike, where) as F64
/** The point (x, y) as a vector, the argument of a planar field. */
const point2 = (x: number, y: number) => tensor([x, y])

/** Line segments: `start` and `end` are k × 2 matrices of points. */
export type Segments = { start: Matrix; end: Matrix }

/** A planar vector field sampled on a grid. Matrices are ny × nx: entry [i, j] is at (x_j, y_i), as heatmaps lay out. */
export type FieldSample = { x: Vector; y: Vector; u: Matrix; v: Matrix; magnitude: Matrix }

/** Samples a planar vector field f(x, y) = (u, v) at every grid point. */
export function sampleField(f: VectorField, grid: Grid2): FieldSample {
  const { nx, ny } = grid
  const axes = gridAxes(grid)
  const xs = toFlat(axes.x)
  const ys = toFlat(axes.y)
  const u = new Float64Array(nx * ny)
  const v = new Float64Array(nx * ny)
  const m = new Float64Array(nx * ny)
  for (let i = 0; i < ny; i++)
    for (let j = 0; j < nx; j++) {
      const r = toF64(f(point2(xs[j], ys[i])), 'sampleField')
      if (r.length !== 2) throw new Error('sampleField: the field must map ℝ² to ℝ²')
      u[i * nx + j] = r[0]
      v[i * nx + j] = r[1]
      m[i * nx + j] = Math.hypot(r[0], r[1])
    }
  return {
    x: axes.x,
    y: axes.y,
    u: fromData(u, [ny, nx]),
    v: fromData(v, [ny, nx]),
    magnitude: fromData(m, [ny, nx]),
  }
}

/**
 * A direction field: at the centre of each of the nx × ny cells of the grid's rectangle, a segment along f of fixed
 * length `scale` (default 0.7) in cell units, centred on the point, so the picture shows direction only. Lengths are
 * normalised in cell units, so segments look equally long whatever the aspect of the axes. Points where f vanishes
 * (‖f‖ ≤ 1e-12) are skipped. `magnitude` gives ‖f‖ at each segment, e.g. for colouring.
 */
export function directionField(
  f: VectorField,
  grid: Grid2,
  { scale = 0.7 }: { scale?: number } = {},
): Segments & { magnitude: Vector } {
  const dx = (grid.x[1] - grid.x[0]) / grid.nx
  const dy = (grid.y[1] - grid.y[0]) / grid.ny
  const start: number[] = []
  const end: number[] = []
  const mag: number[] = []
  for (let i = 0; i < grid.ny; i++)
    for (let j = 0; j < grid.nx; j++) {
      const cx = grid.x[0] + (j + 0.5) * dx
      const cy = grid.y[0] + (i + 0.5) * dy
      const [u, v] = toF64(f(point2(cx, cy)), 'directionField')
      const gu = u / dx
      const gv = v / dy
      const m = Math.hypot(gu, gv)
      if (!(m > 1e-12)) continue
      const hx = ((gu / m) * dx * scale) / 2
      const hy = ((gv / m) * dy * scale) / 2
      start.push(cx - hx, cy - hy)
      end.push(cx + hx, cy + hy)
      mag.push(Math.hypot(u, v))
    }
  const k = mag.length
  return {
    start: fromData(Float64Array.from(start), [k, 2]),
    end: fromData(Float64Array.from(end), [k, 2]),
    magnitude: fromData(Float64Array.from(mag), [k]),
  }
}

/**
 * The slope field of a scalar ODE x′ = g(t, x): the direction field of (1, g(t, x)) on a (t, x) grid, without
 * arrowheads' worth of meaning (solutions run left to right).
 */
export function slopeField(g: (t: number, x: number) => number, grid: Grid2, options: { scale?: number } = {}) {
  return directionField(
    (p) => {
      const d = p.data
      return [1, g(d[p.offset], d[p.offset + p.strides[0]])]
    },
    grid,
    options,
  )
}

/**
 * The segments where a sampled field crosses `level`, by marching squares: `values` is ny × nx with entry [i, j] at
 * (x_j, y_i). Each cell whose corners straddle the level contributes one or two segments, their ends placed by linear
 * interpolation along the cell's edges; saddle cells are resolved by the average of the corners.
 */
export function contour(x: VectorLike, y: VectorLike, values: Matrix, level = 0): Segments {
  const xs = toF64(x, 'contour')
  const ys = toF64(y, 'contour')
  const [ny, nx] = values.shape
  if (nx !== xs.length || ny !== ys.length) throw new Error('contour: values must be ny × nx')
  const z = Float64Array.from(toFlat(values))
  const at = (i: number, j: number) => z[i * nx + j]
  const start: number[] = []
  const end: number[] = []
  const push = (p: [number, number], q: [number, number]) => {
    start.push(p[0], p[1])
    end.push(q[0], q[1])
  }
  const cross = (xa: number, ya: number, za: number, xb: number, yb: number, zb: number): [number, number] => {
    const t = za === zb ? 0.5 : (level - za) / (zb - za)
    return [xa + t * (xb - xa), ya + t * (yb - ya)]
  }
  for (let i = 0; i + 1 < ny; i++)
    for (let j = 0; j + 1 < nx; j++) {
      const [x0, x1, y0, y1] = [xs[j], xs[j + 1], ys[i], ys[i + 1]]
      const [a, b, c, d] = [at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)]
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
  const k = start.length / 2
  return { start: fromData(Float64Array.from(start), [k, 2]), end: fromData(Float64Array.from(end), [k, 2]) }
}

/** The level set {g = level} of a scalar field on a grid, as segments (sampling plus `contour`). */
export function levelSet(g: ScalarField, grid: Grid2, level = 0): Segments {
  const s = sampleScalar(g, grid)
  return contour(s.x, s.y, s.values, level)
}

/**
 * The nullclines of a planar field: for each component i, the curve where f_i(x, y) = 0 (where the flow is vertical
 * for i = 0 and horizontal for i = 1), found by marching squares on the grid. Fixed points lie where they cross.
 */
export function nullclines(f: VectorField, grid: Grid2): [Segments, Segments] {
  const s = sampleField(f, grid)
  return [contour(s.x, s.y, s.u, 0), contour(s.x, s.y, s.v, 0)]
}
