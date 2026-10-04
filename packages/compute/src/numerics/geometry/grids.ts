/** Grids for evaluating functions of two variables, part of `aifn-compute/numerics/geometry`. */

import { dense, fromData, linspace, meshgrid, type Tensor } from 'aifn-compute/foundation/tensor'
import type { VectorLike } from 'aifn-compute/foundation/contracts'

/** A coordinate argument as numbers. */
const axis = (v: VectorLike) => Array.from(dense.toF64(v, 'grid'))

/** A regular 2-D grid: its axes, every point as rows (x fastest), and its shape. */
export interface Grid2d {
  x: Tensor
  y: Tensor
  /** All grid points, (ny · nx) × 2, row i·nx + j is (x[j], y[i]). */
  points: Tensor
  shape: [number, number]
}

/** A grid of nx × ny points spanning [x0, x1] × [y0, y1], ends included. */
export function grid2d(
  xRange: readonly [number, number],
  yRange: readonly [number, number],
  nx: number,
  ny: number = nx,
): Grid2d {
  const x = linspace(xRange[0], xRange[1], nx)
  const y = linspace(yRange[0], yRange[1], ny)
  const [X, Y] = meshgrid(x, y)
  const pts = new Float64Array(nx * ny * 2)
  for (let k = 0; k < nx * ny; k++) {
    pts[2 * k] = X.data[k]
    pts[2 * k + 1] = Y.data[k]
  }
  return { x, y, points: fromData(pts, [nx * ny, 2]), shape: [ny, nx] }
}

/** Evaluate f(x, y) on a grid: z[i][j] = f(x[j], y[i]), a [ny, nx] tensor (the layout of heatmaps and contours). */
export function evaluateGrid(f: (x: number, y: number) => number, x: VectorLike, y: VectorLike): Tensor {
  const xs = axis(x)
  const ys = axis(y)
  const out = new Float64Array(xs.length * ys.length)
  for (let i = 0; i < ys.length; i++) for (let j = 0; j < xs.length; j++) out[i * xs.length + j] = f(xs[j], ys[i])
  return fromData(out, [ys.length, xs.length])
}
