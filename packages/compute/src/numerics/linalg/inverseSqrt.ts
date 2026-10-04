/**
 * The symmetric inverse square root $\Smat^{-1/2} = \Vmat \Lambdamat^{-1/2} \Vmat^\top$ of a symmetric
 * positive-definite matrix, from its eigendecomposition $\Smat = \Vmat \Lambdamat \Vmat^\top$: the whitening map of a
 * covariance (CCA's $\Cmat_{xx}^{-1/2}$) and FastICA's symmetric decorrelation $(\Wmat\Wmat^\top)^{-1/2} \Wmat$
 * (Hyvärinen and Oja, 2000).
 */

import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { DomainError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { eigh } from './eigh'

/** Options of `symmetricInverseSqrt`. */
export type InverseSqrtOptions = {
  /**
   * Eigenvalues below `floor` are raised to it rather than refused, as scikit-learn's FastICA clips at the smallest
   * normal double. Omitted: an eigenvalue at or below $10^{-12}$ times the largest throws `NumericalError`
   * ('singular').
   */
  floor?: number
}

/**
 * $\Smat^{-1/2}$ $[d, d]$ of a symmetric positive-definite $\Smat$ $[d, d]$ (only its lower triangle is read), by
 * `eigh`. Throws `ShapeError` for a non-square matrix and `NumericalError` ('singular') for a singular or indefinite
 * one unless a `floor` is given.
 *
 * @param S The matrix $\Smat$ ($d \times d$, symmetric positive definite), as a tensor or nested arrays. Only its lower
 *   triangle is read, and it is not modified.
 * @param options How to treat eigenvalues that are too small to invert (default: throw `NumericalError`). A `floor`
 *   that is not positive throws `DomainError`.
 * @returns $\Smat^{-1/2}$ as a new symmetric $d \times d$ float64 tensor.
 *
 * @example Whiten with the inverse square root
 * const S = tensor([[4, 1], [1, 3]])
 * const W = symmetricInverseSqrt(S)
 * print('S^(-1/2) =', W)
 * print('W S W =', matmul(matmul(W, S), W))
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
