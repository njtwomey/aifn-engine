/**
 * Random projections (Johnson and Lindenstrauss, 1984; Dasgupta and Gupta, 2003): a $k \times d$ matrix $\mathbf{R}$ of independent random
 * entries scaled so that $\expect[\|\mathbf{R}\mathbf{x}\|^2] = \|\mathbf{x}\|^2$, which maps $n$ points to $k$ dimensions while keeping every pairwise distance within
 * a factor $1 \pm \varepsilon$ with probability at least $1/n$ once $k \ge 4 \ln n / (\varepsilon^2/2 - \varepsilon^3/3)$.
 *
 * - `gaussian`: entries $\mathcal{N}(0, 1/k)$, as scikit-learn's `GaussianRandomProjection`.
 * - `sparse`: entries $\pm 1/\sqrt{s k}$ with probability $s/2$ each and 0 otherwise, density $s$ (Achlioptas, 2003, $s = 1/3$; Li,
 *   Hastie and Church, 2006, $s = 1/\sqrt{d}$ by default), as scikit-learn's `SparseRandomProjection`.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The distribution of the projection's entries. */
export type ProjectionKind = 'gaussian' | 'sparse'

/** Options of `randomProjectionMatrix` and `randomProjection`. */
export type RandomProjectionOptions = {
  /** Entry distribution (default `'gaussian'`). */
  kind?: ProjectionKind
  /** For `sparse`: the share of non-zero entries $s \in (0, 1]$ (default $1/\sqrt{d}$; $1/3$ is Achlioptas's database-friendly case). */
  density?: number
}

/**
 * The target dimension $k = \lfloor 4 \ln n / (\varepsilon^2/2 - \varepsilon^3/3) \rfloor$ of the
 * Johnson–Lindenstrauss bound, as scikit-learn's `johnson_lindenstrauss_min_dim` (which truncates; the bound itself
 * asks for $k$ at least the unrounded value). At that $k$ a random projection of $n$ points keeps every squared
 * pairwise distance within a factor $1 \pm \varepsilon$ with probability at least $1/n$ (Dasgupta and Gupta, 2003,
 * Thm. 2.1). Does not depend on the input dimension.
 *
 * @param samples Number of sample points $n \ge 1$.
 * @param epsilon Maximum relative distortion $\varepsilon \in (0, 1)$.
 * @returns The integer target dimension $k$.
 *
 * @example Calculate minimum dimension for 1000 points
 * print('dim =', johnsonLindenstraussDimension(1000, 0.5))
 */
export function johnsonLindenstraussDimension(samples: Size, epsilon: number): Size {
  if (!(Number.isInteger(samples) && samples >= 1))
    throw new DomainError(
      'johnsonLindenstraussDimension',
      `johnsonLindenstraussDimension: samples must be a positive integer, got ${samples}`,
    )
  if (!(epsilon > 0 && epsilon < 1))
    throw new DomainError(
      'johnsonLindenstraussDimension',
      `johnsonLindenstraussDimension: epsilon must lie in (0, 1), got ${epsilon}`,
    )
  const denominator = epsilon ** 2 / 2 - epsilon ** 3 / 3
  return Math.floor((4 * Math.log(samples)) / denominator)
}

/**
 * The distortion the Johnson–Lindenstrauss bound guarantees for $n$ points at target dimension $k$: the smallest
 * $\varepsilon \in (0, 1)$ with `johnsonLindenstraussDimension(n, ε)` $\le k$, by bisection on the continuous bound
 * $4 \ln n / (\varepsilon^2/2 - \varepsilon^3/3) \le k$. Returns NaN when even $\varepsilon \to 1$ needs more than $k$
 * dimensions ($k < 24 \ln n$).
 *
 * @param samples Number of sample points $n \ge 1$.
 * @param targetDim Target projection dimension $k \ge 1$.
 * @returns The guaranteed relative distortion $\varepsilon \in (0, 1)$, or NaN.
 *
 * @example Calculate distortion bound for 100 points in 500 dimensions
 * print('eps =', johnsonLindenstraussEpsilon(100, 500) < 0.5)
 */
export function johnsonLindenstraussEpsilon(samples: Size, targetDim: Size): number {
  if (!(Number.isInteger(samples) && samples >= 1))
    throw new DomainError(
      'johnsonLindenstraussEpsilon',
      `johnsonLindenstraussEpsilon: samples must be a positive integer, got ${samples}`,
    )
  const need = (e: number) => (4 * Math.log(samples)) / (e ** 2 / 2 - e ** 3 / 3)
  if (samples === 1) return 0
  if (need(1 - 1e-12) > targetDim) return NaN
  let lo = 1e-12
  let hi = 1 - 1e-12
  for (let i = 0; i < 200 && hi - lo > 1e-14; i++) {
    const mid = (lo + hi) / 2
    if (need(mid) > targetDim) lo = mid
    else hi = mid
  }
  return hi
}

/**
 * A random projection matrix $\Rmat$ ($k \times d$) with $\expect[\Rmat^\top\Rmat] = \Imat$, drawn from `stream`.
 *
 * @param inputDim Dimension $d$ of input data.
 * @param targetDim Dimension $k$ of projected data.
 * @param stream Random stream used to draw matrix entries.
 * @param options Distribution kind (`'gaussian'` or `'sparse'`) and sparsity density.
 * @returns A rank-2 tensor of shape $[k, d]$ representing the projection matrix.
 *
 * @example Generate a Gaussian random projection matrix
 * const R = randomProjectionMatrix(10, 3, stream('proj'))
 * print('shape =', R.shape)
 */
export function randomProjectionMatrix(
  inputDim: Size,
  targetDim: Size,
  stream: Stream,
  options: RandomProjectionOptions = {},
): Tensor {
  const { kind = 'gaussian' } = options
  const d = inputDim
  const k = targetDim
  if (!(Number.isInteger(k) && k >= 1))
    throw new DomainError('randomProjectionMatrix', `randomProjectionMatrix: target dimension ${k} < 1`)
  if (kind === 'gaussian') {
    const z = toFlat(normal(stream, 0, 1 / Math.sqrt(k), { shape: [k * d] }))
    return fromData(Float64Array.from(z), [k, d])
  }
  const s = options.density ?? 1 / Math.sqrt(d)
  if (!(s > 0 && s <= 1))
    throw new DomainError('randomProjectionMatrix', `randomProjectionMatrix: density must lie in (0, 1], got ${s}`)
  const u = toFlat(uniform(stream, 0, 1, { shape: [k * d] }))
  const value = 1 / Math.sqrt(s * k)
  return fromData(
    Float64Array.from(u, (v) => (v < s / 2 ? -value : v < s ? value : 0)),
    [k, d],
  )
}

/**
 * Project the rows of $\Xmat$ ($n \times d$) to `targetDim` dimensions: $\Xmat\Rmat^\top$ ($n \times k$), with the
 * matrix $\Rmat$ ($k \times d$) used.
 *
 * @param X Input data matrix of shape $[n, d]$.
 * @param targetDim Target dimension $k$.
 * @param stream Random stream used for drawing projection entries.
 * @param options Projection configuration options.
 * @returns An object containing projected tensor `projected` and the projection `matrix`.
 *
 * @example Project high-dimensional points
 * const X = [[1, 2, 3, 4], [5, 6, 7, 8]]
 * const res = randomProjection(X, 2, stream('proj'))
 * print('shape =', res.projected.shape)
 */
export function randomProjection(
  X: MatrixLike,
  targetDim: Size,
  stream: Stream,
  options: RandomProjectionOptions = {},
): { projected: Tensor; matrix: Tensor } {
  const { data, m: n, n: d } = dense.toMatrixF64(X, 'randomProjection')
  const R = randomProjectionMatrix(d, targetDim, stream, options)
  const Rt = dense.transpose(dense.data(R), targetDim, d)
  return { projected: fromData(dense.matMul(data, Rt, n, d, targetDim), [n, targetDim]), matrix: R }
}

/**
 * The distortion of a map on pairwise distances: for every pair $i < j$ of rows, the ratio
 * $\|y_i - y_j\|^2 / \|x_i - x_j\|^2$ of squared distances after ($\Ymat$ $[n, k]$) and before ($\Xmat$ $[n, d]$),
 * skipping coincident rows; with the largest $|\text{ratio} - 1|$.
 *
 * @param X Original data matrix of shape $[n, d]$.
 * @param Y Projected data matrix of shape $[n, k]$.
 * @returns An object containing array `ratios` of squared distance ratios and maximum distortion `maxDistortion`.
 *
 * @example Measure pairwise distance distortion
 * const X = [[0, 0], [1, 1], [2, 2]]
 * const Y = [[0], [1.414], [2.828]]
 * const d = distanceDistortion(X, Y)
 * print('distortion =', d.maxDistortion < 0.01)
 */
export function distanceDistortion(X: MatrixLike, Y: MatrixLike): { ratios: Float64Array; maxDistortion: number } {
  const x = dense.toMatrixF64(X, 'distanceDistortion')
  const y = dense.toMatrixF64(Y, 'distanceDistortion')
  if (x.m !== y.m) throw new ShapeError('distanceDistortion', `distanceDistortion: ${x.m} rows before and ${y.m} after`)
  const n = x.m
  const ratios: number[] = []
  let worst = 0
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let a = 0
      for (let c = 0; c < x.n; c++) a += (x.data[i * x.n + c] - x.data[j * x.n + c]) ** 2
      if (a === 0) continue
      let b = 0
      for (let c = 0; c < y.n; c++) b += (y.data[i * y.n + c] - y.data[j * y.n + c]) ** 2
      const r = b / a
      ratios.push(r)
      worst = Math.max(worst, Math.abs(r - 1))
    }
  return { ratios: Float64Array.from(ratios), maxDistortion: worst }
}
