/**
 * Sparse coding of many signals at once: every column of a matrix $\Ymat$ ($m \times n$) coded over one dictionary
 * $\Dmat$ ($m \times k$) by a chosen method, giving the code matrix $\Xmat$ ($k \times n$) with $\Ymat \approx
 * \Dmat\Xmat$, as scikit-learn's `sparse_encode` (which takes signals and atoms as rows instead).
 */

import type { MatrixLike, Size } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { readDictionary } from './atoms'
import { basisPursuit, basisPursuitDenoising, iterativeHardThresholding } from './convex'
import { matchingPursuit, orthogonalMatchingPursuit, type SparseApproximation } from './pursuit'

/**
 * How to code each signal, by method and that method's options:
 *
 * - `omp`: `orthogonalMatchingPursuit`, to `sparsity` atoms or the `tolerance`.
 * - `mp`: `matchingPursuit`, to the `tolerance` or `maxSteps` steps.
 * - `bp`: `basisPursuit`, an exact representation of least $\ell_1$ norm (NaN for a signal outside the range).
 * - `lasso`: `basisPursuitDenoising` with penalty `lambda`.
 * - `iht`: `iterativeHardThresholding` to `sparsity` non-zeros.
 */
export type SparseCoder =
  | { method: 'omp'; sparsity?: Size; tolerance?: number }
  | { method: 'mp'; maxSteps?: Size; tolerance?: number }
  | { method: 'bp' }
  | { method: 'lasso'; lambda: number; maxSteps?: Size; tolerance?: number }
  | { method: 'iht'; sparsity: Size; maxSteps?: Size; tolerance?: number }

/** The result of `sparseCode`. */
export type SparseCodes = {
  /** The codes $\Xmat$, $k \times n$: column $i$ codes signal $i$. */
  X: Tensor
  /** $\lVert \yvec_i - \Dmat\xvec_i \rVert$ for every signal $i$, $n$ values. */
  residualNorms: Tensor
}

/**
 * Code one signal by the chosen method.
 *
 * @param D The dictionary, $m \times k$.
 * @param y The signal, $m$ values.
 * @param coder The method and its options.
 * @returns The method's sparse approximation of the signal.
 */
function codeOne(D: Tensor, y: Float64Array, coder: SparseCoder): SparseApproximation {
  switch (coder.method) {
    case 'omp':
      return orthogonalMatchingPursuit(D, y, coder)
    case 'mp':
      return matchingPursuit(D, y, coder)
    case 'bp':
      return basisPursuit(D, y)
    case 'lasso':
      return basisPursuitDenoising(D, y, coder)
    case 'iht':
      return iterativeHardThresholding(D, y, coder)
    default:
      throw new DomainError('sparseCode', `sparseCode: unknown method ${(coder as { method: string }).method}`)
  }
}

/**
 * Code every column of $\Ymat$ over the dictionary $\Dmat$ by one method, independently: column $i$ of the result is
 * the code of column $i$ of $\Ymat$.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param Y The signals $\Ymat$, $m \times n$, one signal per column.
 * @param coder The method and its options (default orthogonal matching pursuit to $\min(m, k)$ atoms).
 * @returns The codes $\Xmat$ ($k \times n$) and each signal's residual norm.
 *
 * @example Code three signals with two atoms each
 * const s = Math.SQRT1_2
 * const D = [[1, 0, 0, s], [0, 1, 0, s], [0, 0, 1, 0]]
 * const Y = [[1, 0, 2], [1, 2, 0], [0, 3, 1]]
 * const { X, residualNorms } = sparseCode(D, Y, { method: 'omp', sparsity: 2 })
 * print('X =', X)
 * print('residual norms =', residualNorms)
 *
 * @example The same signals by the lasso
 * const s = Math.SQRT1_2
 * const D = [[1, 0, 0, s], [0, 1, 0, s], [0, 0, 1, 0]]
 * const Y = [[1, 0, 2], [1, 2, 0], [0, 3, 1]]
 * print('X =', sparseCode(D, Y, { method: 'lasso', lambda: 0.2 }).X)
 */
export function sparseCode(D: MatrixLike, Y: MatrixLike, coder: SparseCoder = { method: 'omp' }): SparseCodes {
  const d = readDictionary(D, 'sparseCode')
  const y = dense.toMatrixF64(Y, 'sparseCode')
  if (y.m !== d.m)
    throw new ShapeError(
      'sparseCode',
      `sparseCode: the signals have ${y.m} rows but the dictionary's atoms have ${d.m}`,
    )
  const dict = dense.mat(d.data, d.m, d.k)
  const X = new Float64Array(d.k * y.n)
  const residualNorms = new Float64Array(y.n)
  for (let i = 0; i < y.n; i++) {
    const signal = Float64Array.from({ length: y.m }, (_, r) => y.data[r * y.n + i])
    const code = codeOne(dict, signal, coder)
    const x = dense.data(code.x)
    for (let j = 0; j < d.k; j++) X[j * y.n + i] = x[j]
    residualNorms[i] = code.residualNorm
  }
  return { X: dense.mat(X, d.k, y.n), residualNorms: dense.vec(residualNorms) }
}
