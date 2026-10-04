/**
 * The continuous Lyapunov and discrete Stein equations, solved directly as a Kronecker-product linear system, the
 * method for small $n$ (Golub & Van Loan, 2013, "Matrix Computations", 4th ed., §7.6.3; Bartels & Stewart, 1972, is the
 * method for large $n$).
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
 * Solves the continuous Lyapunov equation $\Amat\Xmat + \Xmat\Amat^\top + \Qmat = 0$ (the Gramian convention; scipy's
 * `solve_continuous_lyapunov(A, −Q)`), or with `discrete` the Stein equation
 * $\Xmat = \Amat\Xmat\Amat^\top + \Qmat$. Solved directly as the $n^2 \times n^2$ linear system
 * $(\Imat \otimes \Amat + \Amat \otimes \Imat) \operatorname{vec} \Xmat = -\operatorname{vec} \Qmat$, or
 * $(\Imat - \Amat \otimes \Amat) \operatorname{vec} \Xmat = \operatorname{vec} \Qmat$, which suits the small $n$ here.
 * Unique iff no two eigenvalues of $\Amat$ sum to 0 (continuous) or multiply to 1 (discrete); otherwise `singular`.
 *
 * @param a The coefficient matrix $\Amat$ ($n \times n$), as a tensor or nested arrays.
 * @param q The constant term $\Qmat$ ($n \times n$), as a tensor or nested arrays. It enters with a plus sign, so for a
 *   stable $\Amat$ and positive semi-definite $\Qmat$ the solution is positive semi-definite.
 * @param options Which of the two equations to solve.
 * @param options.discrete True solves the discrete (Stein) equation; false (the default) the continuous Lyapunov
 *   equation.
 * @returns `X`, the solution ($n \times n$, symmetrised as $(\Xmat + \Xmat^\top)/2$), with `singular` false; or `X`
 *   null and `singular` true when the linear system has no unique solution.
 *
 * @example Solve the continuous Lyapunov equation
 * // A X + X Aᵀ + Q = 0 for a stable A.
 * const A = tensor([[-1, 0], [0, -2]])
 * const Q = tensor([[1, 0], [0, 1]])
 * const { X, singular } = lyapunov(A, Q)
 * print('X =', X)
 * print('singular =', singular)
 *
 * @example The discrete (Stein) equation
 * const { X } = lyapunov(tensor([[0.5, 0], [0, 0.25]]), tensor([[1, 0], [0, 1]]), { discrete: true })
 * print('X =', X)
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
