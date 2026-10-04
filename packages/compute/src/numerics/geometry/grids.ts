/** Grids for evaluating functions of two variables, part of `aifn-compute/numerics/geometry`. */

import { dense, fromData, linspace, meshgrid, type Tensor } from 'aifn-compute/foundation/tensor'
import type { VectorLike } from 'aifn-compute/foundation/contracts'

/**
 * Convert a vector-like coordinate axis to a JavaScript number array.
 *
 * @param v - Vector-like coordinate values.
 * @returns Array of 64-bit float coordinates.
 */
const axis = (v: VectorLike) => Array.from(dense.toF64(v, 'grid'))

/** A regular 2-D rectangular grid: its 1D coordinate axes, flattened point array, and grid shape. */
export interface Grid2d {
  /** Horizontal coordinate axis tensor of length $nx$. */
  x: Tensor
  /** Vertical coordinate axis tensor of length $ny$. */
  y: Tensor
  /** All grid point coordinates of shape $[ny \cdot nx, 2]$, where row $i \cdot nx + j$ corresponds to $(x_j, y_i)$. */
  points: Tensor
  /** Grid dimensions $[ny, nx]$. */
  shape: [number, number]
}

/**
 * Construct a regular 2D Cartesian grid of $nx \times ny$ points spanning $[x_0, x_1] \times [y_0, y_1]$ with endpoints included.
 *
 * @param xRange - Interval $[x_0, x_1]$ along the horizontal axis.
 * @param yRange - Interval $[y_0, y_1]$ along the vertical axis.
 * @param nx - Number of grid points along the horizontal axis.
 * @param ny - Number of grid points along the vertical axis (defaults to `nx`).
 * @returns Grid object `Grid2d` containing coordinate tensors and point coordinates.
 * @example Creating a 2D grid
 * const grid = grid2d([-1, 1], [-1, 1], 5)
 * print('grid shape =', grid.shape)
 */
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

/**
 * Evaluate a bivariate scalar function $f(x, y)$ over a grid of coordinates, producing a tensor of shape $[ny, nx]$
 * with entries $z_{i,j} = f(x_j, y_i)$, matching the layout expected by heatmaps and contour plots.
 *
 * @param f - Bivariate scalar function to evaluate at coordinates $(x, y)$.
 * @param x - Grid coordinates along the horizontal axis, length $nx$.
 * @param y - Grid coordinates along the vertical axis, length $ny$.
 * @returns Evaluated 2D field values tensor of shape $[ny, nx]$.
 * @example Evaluating a function on a 2D grid
 * const x = [0, 1, 2]
 * const y = [0, 1]
 * const z = evaluateGrid((x, y) => x + 2 * y, x, y)
 * print('field shape =', z.shape)
 */
export function evaluateGrid(f: (x: number, y: number) => number, x: VectorLike, y: VectorLike): Tensor {
  const xs = axis(x)
  const ys = axis(y)
  const out = new Float64Array(xs.length * ys.length)
  for (let i = 0; i < ys.length; i++) for (let j = 0; j < xs.length; j++) out[i * xs.length + j] = f(xs[j], ys[i])
  return fromData(out, [ys.length, xs.length])
}
