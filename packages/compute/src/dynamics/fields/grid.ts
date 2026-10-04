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

/** A rectangular grid of the plane: nx points on [x₀, x₁] and ny points on [y₀, y₁], ends included. */
export type Grid2 = { x: readonly [Scalar, Scalar]; y: readonly [Scalar, Scalar]; nx: Size; ny: Size }

const axis = (lo: Scalar, hi: Scalar, n: Size): F64 =>
  Float64Array.from({ length: n }, (_, i) => (n === 1 ? (lo + hi) / 2 : lo + ((hi - lo) * i) / (n - 1)))

const point2 = (x: Scalar, y: Scalar): Tensor => fromData(Float64Array.of(x, y), [2])

/** A scalar field's value as a number. */
function scalarValue(v: Value, where: string): Scalar {
  if (typeof v === 'number') return v
  const t = v as Tensor
  if (t.shape.length > 1 || (t.shape.length === 1 && t.shape[0] !== 1))
    throw new ShapeError(where, `${where}: expected a scalar, got shape [${t.shape.join(', ')}]`)
  return t.data[t.offset]
}

/** The coordinates of a grid: `x` (nx) and `y` (ny). */
export function gridAxes(grid: Grid2): { x: Vector; y: Vector } {
  return {
    x: fromData(axis(grid.x[0], grid.x[1], grid.nx), [grid.nx]),
    y: fromData(axis(grid.y[0], grid.y[1], grid.ny), [grid.ny]),
  }
}

/** Samples a scalar field g(x, y) on a grid: `values` is ny × nx, entry [i, j] at (x_j, y_i). */
export function sampleScalar(g: ScalarField, grid: Grid2): { x: Vector; y: Vector; values: Matrix } {
  const { nx, ny } = grid
  const xs = axis(grid.x[0], grid.x[1], nx)
  const ys = axis(grid.y[0], grid.y[1], ny)
  const out = new Float64Array(nx * ny)
  for (let i = 0; i < ny; i++)
    for (let j = 0; j < nx; j++) out[i * nx + j] = scalarValue(g(point2(xs[j], ys[i])), 'sampleScalar')
  return { x: fromData(xs, [nx]), y: fromData(ys, [ny]), values: fromData(out, [ny, nx]) }
}
