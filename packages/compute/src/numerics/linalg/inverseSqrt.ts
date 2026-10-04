/**
 * The symmetric inverse square root S^{−1/2} = V Λ^{−1/2} Vᵀ of a symmetric positive-definite matrix, from its
 * eigendecomposition S = V Λ Vᵀ: the whitening map of a covariance (CCA's C_xx^{−1/2}) and FastICA's symmetric
 * decorrelation (WWᵀ)^{−1/2} W (Hyvärinen and Oja, 2000).
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { eigh } from './eigh'

/** Options of `symmetricInverseSqrt`. */
export type InverseSqrtOptions = {
  /**
   * Eigenvalues below `floor` are raised to it rather than refused, as scikit-learn's FastICA clips at the smallest
   * normal double. Omitted: an eigenvalue at or below 1e−12 times the largest throws `NumericalError` ('singular').
   */
  floor?: number
}

/**
 * S^{−1/2} [d, d] of a symmetric positive-definite S [d, d] (only its lower triangle is read), by `eigh`. Throws
 * `ShapeError` for a non-square matrix and `NumericalError` ('singular') for a singular or indefinite one unless a
 * `floor` is given.
 */
export function symmetricInverseSqrt(S: MatrixLike, options: InverseSqrtOptions = {}): Tensor {
  const op = 'symmetricInverseSqrt'
  const { data, m, n } = dense.toMatrixF64(S, op)
  if (m !== n) throw new ShapeError(op, `${op}: expected a square matrix, got [${m}, ${n}]`, [[m, n]])
  const { floor } = options
  if (floor !== undefined && !(floor > 0)) throw new DomainError(op, `${op}: floor must be positive, got ${floor}`)
  const e = eigh(fromData(data, [n, n]))
  const values = dense.data(e.values)
  const V = dense.data(e.vectors)
  const top = Math.max(values[0] ?? 0, 0)
  const out = new Float64Array(n * n)
  for (let c = 0; c < n; c++) {
    let lambda = values[c]
    if (floor !== undefined) lambda = Math.max(lambda, floor)
    else if (!(lambda > top * 1e-12))
      throw new NumericalError(op, `${op}: the matrix is singular or not positive definite`, 'singular')
    const w = 1 / Math.sqrt(lambda)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] += V[i * n + c] * w * V[j * n + c]
  }
  return fromData(out, [n, n])
}
