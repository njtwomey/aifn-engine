/** Grid constructors, part of `aifn-compute/foundation/tensor`: `meshgrid` and `logspace`. */

import { fromData, isTensor, type Tensor } from './core'
import { linspace, toFlat } from './create'

/**
 * A coordinate argument as numbers (a copy of the private helper of the lab's decimation).
 *
 * @param v A tensor (read in row-major order, whatever its shape) or an array-like of numbers.
 * @returns The values as a new plain array.
 */
const axis = (v: Tensor | ArrayLike<number>) => (isTensor(v) ? toFlat(v) : Array.from(v))

/**
 * Coordinate matrices from coordinate vectors, as `numpy.meshgrid` with `indexing='xy'` (default: both outputs are
 * $n_y \times n_x$, $\Xmat$ varying along columns) or `'ij'` ($n_x \times n_y$, $\Xmat$ varying along rows).
 *
 * @param x The $n_x$ coordinates along the first axis (a tensor is flattened).
 * @param y The $n_y$ coordinates along the second axis (a tensor is flattened).
 * @param options How the outputs are indexed.
 * @param options.indexing `'xy'` for Cartesian (image) indexing, rows following `y`; `'ij'` for matrix indexing,
 *   rows following `x`.
 * @returns The pair $[\Xmat, \Ymat]$ of float64 matrices: entry by entry, the $x$ and the $y$ coordinate of a grid
 *   point.
 *
 * @example Cartesian indexing
 * const [X, Y] = meshgrid([1, 2, 3], [10, 20])
 * print('X =', X)
 * print('Y =', Y)
 *
 * @example Matrix indexing
 * const [X, Y] = meshgrid([1, 2, 3], [10, 20], { indexing: 'ij' })
 * print('X =', X)
 * print('Y =', Y)
 */
export function meshgrid(
  x: Tensor | ArrayLike<number>,
  y: Tensor | ArrayLike<number>,
  { indexing = 'xy' }: { indexing?: 'xy' | 'ij' } = {},
): [Tensor, Tensor] {
  const xs = axis(x)
  const ys = axis(y)
  const [nx, ny] = [xs.length, ys.length]
  const X = new Float64Array(nx * ny)
  const Y = new Float64Array(nx * ny)
  if (indexing === 'xy') {
    for (let i = 0; i < ny; i++)
      for (let j = 0; j < nx; j++) {
        X[i * nx + j] = xs[j]
        Y[i * nx + j] = ys[i]
      }
    return [fromData(X, [ny, nx]), fromData(Y, [ny, nx])]
  }
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++) {
      X[i * ny + j] = xs[i]
      Y[i * ny + j] = ys[j]
    }
  return [fromData(X, [nx, ny]), fromData(Y, [nx, ny])]
}

/**
 * Numbers spaced evenly on a log scale from $b^{\text{start}}$ to $b^{\text{stop}}$ (inclusive) for the base $b$, as
 * `numpy.logspace`: $b^v$ for each value $v$ of `linspace(start, stop, num)`.
 *
 * @param start The exponent of the first value.
 * @param stop The exponent of the last value.
 * @param num The number of values.
 * @param base The base $b$ the exponents apply to.
 * @returns A new float64 vector of `num` values.
 *
 * @example Decades, and powers of two
 * print('base 10:', logspace(0, 3, 4))
 * print('base 2:', logspace(0, 4, 5, 2))
 */
export function logspace(start: number, stop: number, num = 50, base = 10): Tensor {
  const t = toFlat(linspace(start, stop, num))
  return fromData(Float64Array.from(t, (v) => base ** v))
}
