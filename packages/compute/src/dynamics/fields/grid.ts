/**
 * Scalar fields sampled on rectangular grids of the plane (the grid that direction fields, contours and level sets are
 * drawn on; the drawing itself is in the lab). Sampling follows the node-centred convention of Lorensen & Cline
 * (1987), "Marching cubes", SIGGRAPH, in its two-dimensional form.
 */

import { dense, fromData, type Matrix, type Tensor, type Value, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import type { ScalarField } from './calculus'
import { ShapeError } from 'aifn-compute/foundation/errors'

type F64 = dense.F64

/**
 * A rectangular grid of the plane: `nx` equally spaced points on $[x_0, x_1]$ (`x` is $[x_0, x_1]$) and `ny` on
 * $[y_0, y_1]$ (`y` is $[y_0, y_1]$), ends included. An axis of one point sits at the interval's midpoint.
 */
export type Grid2 = { x: readonly [Scalar, Scalar]; y: readonly [Scalar, Scalar]; nx: Size; ny: Size }

/**
 * Equally spaced points on $[\mathit{lo}, \mathit{hi}]$, ends included; one point is the midpoint.
 *
 * @param lo The first point.
 * @param hi The last point.
 * @param n The number of points.
 * @returns The `n` points, in order from `lo` to `hi`.
 */
const axis = (lo: Scalar, hi: Scalar, n: Size): F64 =>
  Float64Array.from({ length: n }, (_, i) => (n === 1 ? (lo + hi) / 2 : lo + ((hi - lo) * i) / (n - 1)))

/**
 * A point of the plane as a rank-1 tensor, the form a scalar field is called with.
 *
 * @param x The first coordinate.
 * @param y The second coordinate.
 * @returns The vector $(x, y)$.
 */
const point2 = (x: Scalar, y: Scalar): Tensor => fromData(Float64Array.of(x, y), [2])

/**
 * A scalar field's value as a number: a number, or a tensor with one entry (shape $[\,]$ or $[1]$); any other shape
 * throws `ShapeError`.
 *
 * @param v The value the field returned.
 * @param where The caller's name, used in error messages.
 * @returns The value as a number.
 */
function scalarValue(v: Value, where: string): Scalar {
  if (typeof v === 'number') return v
  const t = v as Tensor
  if (t.shape.length > 1 || (t.shape.length === 1 && t.shape[0] !== 1))
    throw new ShapeError(where, `${where}: expected a scalar, got shape [${t.shape.join(', ')}]`)
  return t.data[t.offset]
}

/**
 * The coordinates of a grid along each axis.
 *
 * @param grid The grid.
 * @returns `x`, the `nx` coordinates on the first axis, and `y`, the `ny` on the second.
 *
 * @example Three points by two
 * print(gridAxes({ x: [0, 1], y: [0, 2], nx: 3, ny: 2 }))
 */
export function gridAxes(grid: Grid2): { x: Vector; y: Vector } {
  return {
    x: fromData(axis(grid.x[0], grid.x[1], grid.nx), [grid.nx]),
    y: fromData(axis(grid.y[0], grid.y[1], grid.ny), [grid.ny]),
  }
}

/**
 * Samples a scalar field $g(x, y)$ on a grid. The field is called once per grid point and must return a single
 * number (a number or a one-entry tensor), else `ShapeError` is thrown.
 *
 * @param g The scalar field on the plane, called with the point $(x, y)$ as a rank-1 tensor.
 * @param grid The grid to sample on.
 * @returns The axes `x` ($n_x$) and `y` ($n_y$), and `values` ($n_y \times n_x$), entry $[i, j]$ at $(x_j, y_i)$: one
 *   row per $y$, as an image is stored.
 *
 * @example Rows run along x
 * // g(x, y) = x + 10y on x ∈ {0, 0.5, 1}, y ∈ {0, 2}.
 * const { values } = sampleScalar((p) => add(get(p, 0), mul(10, get(p, 1))), { x: [0, 1], y: [0, 2], nx: 3, ny: 2 })
 * print(values)
 */
export function sampleScalar(g: ScalarField, grid: Grid2): { x: Vector; y: Vector; values: Matrix } {
  const { nx, ny } = grid
  const xs = axis(grid.x[0], grid.x[1], nx)
  const ys = axis(grid.y[0], grid.y[1], ny)
  const out = new Float64Array(nx * ny)
  for (let i = 0; i < ny; i++)
    for (let j = 0; j < nx; j++) out[i * nx + j] = scalarValue(g(point2(xs[j], ys[i])), 'sampleScalar')
  return { x: fromData(xs, [nx]), y: fromData(ys, [ny]), values: fromData(out, [ny, nx]) }
}
