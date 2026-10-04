/** Grid constructors, part of `aifn-compute/foundation/tensor`: `meshgrid` and `logspace`. */

import { fromData, isTensor, type Tensor } from './core'
import { linspace, toFlat } from './create'

/** A coordinate argument as numbers (a copy of the private helper of the lab's decimation). */
const axis = (v: Tensor | ArrayLike<number>) => (isTensor(v) ? toFlat(v) : Array.from(v))

/**
 * Coordinate matrices from coordinate vectors, as `numpy.meshgrid` with `indexing='xy'` (default: both outputs are
 * [ny, nx], X varying along columns) or `'ij'` ([nx, ny]).
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

/** Numbers spaced evenly on a log scale from 10^start to 10^stop (inclusive), as `numpy.logspace`. */
export function logspace(start: number, stop: number, num = 50, base = 10): Tensor {
  const t = toFlat(linspace(start, stop, num))
  return fromData(Float64Array.from(t, (v) => base ** v))
}
