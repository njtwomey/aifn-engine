/**
 * The continuous Lyapunov and discrete Stein equations, solved directly as a Kronecker-product linear system, the
 * method for small n (Golub & Van Loan, 2013, "Matrix Computations", 4th ed., §7.6.3; Bartels & Stewart, 1972, is the
 * method for large n).
 */

import { add, eye, mul, neg, reshape, sub, transpose, type Matrix } from 'aifn-compute/foundation/tensor'
import type { MatrixLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { asMatrix } from './dense'
import { luFactor, luSolve } from './lu'
import { kron } from './products'

/** The solution of a Lyapunov equation, or `singular` when the equation has no unique solution. */
export type LyapunovSolution = { X: Matrix | null; singular: boolean }

/**
 * Solves the continuous Lyapunov equation A X + X Aᵀ + Q = 0 (the Gramian convention; scipy's
 * `solve_continuous_lyapunov(A, −Q)`), or with `discrete` the Stein equation X = A X Aᵀ + Q. Solved directly as the
 * n²×n² linear system (I ⊗ A + A ⊗ I) vec X = −vec Q, or (I − A ⊗ A) vec X = vec Q, which suits the small n here.
 * Unique iff no two eigenvalues of A sum to 0 (continuous) or multiply to 1 (discrete); otherwise `singular`.
 */
export function lyapunov(
  a: MatrixLike,
  q: MatrixLike,
  { discrete = false }: { discrete?: boolean } = {},
): LyapunovSolution {
  const A = asMatrix(a, 'lyapunov A')
  const Q = asMatrix(q, 'lyapunov Q')
  const n = A.shape[0]
  if (A.shape[1] !== n || Q.shape[0] !== n || Q.shape[1] !== n)
    throw new ShapeError('lyapunov', 'lyapunov: A and Q must be square and the same size')
  const I = eye(n)
  // Row-major vec: vec(A X) = (A ⊗ I) vec X and vec(X Aᵀ) = (I ⊗ A) vec X.
  const K = discrete ? sub(eye(n * n), kron(A, A)) : add(kron(A, I), kron(I, A))
  const f = luFactor(K)
  if (f.singular) return { X: null, singular: true }
  const X = reshape(luSolve(f, reshape(discrete ? Q : neg(Q), [n * n])), [n, n])
  return { X: mul(0.5, add(X, transpose(X))), singular: false }
}
