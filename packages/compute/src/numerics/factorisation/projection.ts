/**
 * Random projections (Johnson and Lindenstrauss, 1984; Dasgupta and Gupta, 2003): a k × d matrix R of independent random
 * entries scaled so that E‖Rx‖² = ‖x‖², which maps n points to k dimensions while keeping every pairwise distance within
 * a factor 1 ± ε with probability at least 1/n once k ≥ 4 ln n / (ε²/2 − ε³/3).
 *
 * - `gaussian`: entries N(0, 1/k), as scikit-learn's `GaussianRandomProjection`.
 * - `sparse`: entries ±1/√(s k) with probability s/2 each and 0 otherwise, density s (Achlioptas, 2003, s = 1/3; Li,
 *   Hastie and Church, 2006, s = 1/√d by default), as scikit-learn's `SparseRandomProjection`.
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { normal, uniform, type Stream } from 'aifn-compute/foundation/random'
import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The distribution of the projection's entries. */
export type ProjectionKind = 'gaussian' | 'sparse'

/** Options of `randomProjectionMatrix` and `randomProjection`. */
export type RandomProjectionOptions = {
  /** Entry distribution (default `gaussian`). */
  kind?: ProjectionKind
  /** For `sparse`: the share of non-zero entries s ∈ (0, 1] (default 1/√d; 1/3 is Achlioptas's database-friendly case). */
  density?: number
}

/**
 * The target dimension k = ⌊4 ln n / (ε²/2 − ε³/3)⌋ of the Johnson–Lindenstrauss bound, as scikit-learn's
 * `johnson_lindenstrauss_min_dim` (which truncates; the bound itself asks for k at least the unrounded value). At that k
 * a random projection of n points keeps every squared pairwise distance within a factor 1 ± ε with probability at
 * least 1/n (Dasgupta and Gupta, 2003, Thm. 2.1: each pair fails with probability at most 2/n², so all n(n − 1)/2
 * pairs hold with probability at least 1/n, and O(n) independent draws find such a map). It does not depend on the
 * input dimension.
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
 * The distortion the Johnson–Lindenstrauss bound guarantees for n points at target dimension k: the smallest ε ∈ (0, 1)
 * with `johnsonLindenstraussDimension(n, ε)` ≤ k, by bisection on the continuous bound 4 ln n / (ε²/2 − ε³/3) ≤ k.
 * NaN when even ε → 1 needs more than k dimensions (k < 24 ln n).
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

/** A random projection matrix R [k, d] with E[RᵀR] = I, drawn from `stream`. */
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

/** Project the rows of X [n, d] to `targetDim` dimensions: XRᵀ [n, k], with the matrix R [k, d] used. */
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
 * The distortion of a map on pairwise distances: for every pair i < j of rows, the ratio ‖yᵢ − yⱼ‖² / ‖xᵢ − xⱼ‖² of
 * squared distances after (Y [n, k]) and before (X [n, d]), skipping coincident rows; with the largest |ratio − 1|.
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
