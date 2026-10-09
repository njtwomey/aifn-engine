/**
 * Pole placement by Ackermann's formula, part of `aifn-compute/dynamics/control` (Ackermann, 1972, "Der Entwurf
 * linearer Regelungssysteme im Zustandsraum", Regelungstechnik 20; Kailath, 1980, "Linear Systems", §3.2).
 *
 * For a single-input plant $\xvec' = \Amat\xvec + \bvec u$ the gain that places the poles of $\Amat - \bvec\Kmat$ is
 * unique, and Ackermann's formula writes it down from the controllability matrix. The result reports whether the pair
 * was controllable, how well conditioned the controllability matrix was, and the poles actually obtained, so that a
 * caller can check the placement rather than trust it.
 */

import { eig, solveDense, svd } from 'aifn-compute/numerics/linalg'
import { dense, fromData, toFlat, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar } from 'aifn-compute/foundation/contracts'
import { polyFromRoots, type ComplexLike } from 'aifn-compute/numerics/polynomial'
import type { StateFeedbackPlant } from './lqr'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The result of `ackermann`. */
export type PolePlacement = {
  /**
   * The gain $\Kmat$ ($1 \times n$) whose closed loop $\Amat - \bvec\Kmat$ has the requested poles, or null when
   * $(\Amat, \bvec)$ is not controllable (or its controllability matrix could not be solved with).
   */
  K: Matrix | null
  /** The desired characteristic polynomial's coefficients $[1, \alpha_1, \dots, \alpha_n]$, highest power first. */
  characteristic: Vector
  /** The eigenvalues of $\Amat - \bvec\Kmat$ actually obtained, complex128 (empty when `K` is null). */
  closedLoop: Tensor
  /** Whether the controllability matrix has full rank. */
  controllable: boolean
  /**
   * The condition number $\sigma_{\max} / \sigma_{\min}$ of the controllability matrix; large means the gain is
   * sensitive to rounding and to errors in the model.
   */
  conditioning: Scalar
}

/**
 * Single-input pole placement by Ackermann's formula (Ackermann, 1972):
 * $\Kmat = [0 \; \cdots \; 0 \; 1] \, \Ccal^{-1} \varphi(\Amat)$, where
 * $\Ccal = [\bvec, \Amat\bvec, \dots, \Amat^{n-1}\bvec]$ is the controllability matrix and
 * $\varphi(s) = \prod_i (s - p_i)$ the desired characteristic polynomial, so that $u = -\Kmat\xvec$ gives
 * $\operatorname{eig}(\Amat - \bvec\Kmat) = \{p_i\}$. Numerically poor for large $n$ or nearly uncontrollable pairs (it
 * inverts $\Ccal$): the conditioning is reported, and the achieved poles are returned for checking. An uncontrollable
 * pair (rank of $\Ccal$ below $n$) is reported with `K` null, not thrown.
 *
 * @param plant The plant $\xvec' = \Amat\xvec + \bvec u$: `A` square ($n \times n$) and `B` a single column
 *   ($n \times 1$). A non-square `A` throws `ShapeError`; more than one input throws `DomainError`.
 * @param desired The $n$ closed-loop poles $p_i$ wanted: real numbers or complex numbers, complex ones in conjugate
 *   pairs (the gain is real). The wrong number of poles throws `ShapeError`.
 * @returns The gain `K` with the characteristic polynomial, the poles obtained, whether the pair is controllable and
 *   the conditioning of $\Ccal$.
 *
 * @example Place the poles of a double integrator
 * // x'' = u: the closed loop's characteristic polynomial s² + 3s + 2 needs K = [2, 3].
 * const plant = { A: [[0, 1], [0, 0]], B: [[0], [1]] }
 * const { K, characteristic, closedLoop } = ackermann(plant, [-1, -2])
 * print('K =', K)
 * print('characteristic =', characteristic)
 * print('poles obtained =', closedLoop)
 *
 * @example An uncontrollable pair is reported, not thrown
 * // Both states see the same input and have the same dynamics: only their sum can be steered.
 * const { K, controllable } = ackermann({ A: [[1, 0], [0, 1]], B: [[1], [1]] }, [-1, -2])
 * print('controllable =', controllable)
 * print('K =', K)
 */
export function ackermann(plant: StateFeedbackPlant, desired: ComplexLike): PolePlacement {
  const { data: A, m: n, n: nA } = dense.toMatrixF64(plant.A, 'ackermann A')
  if (n !== nA) throw new ShapeError('ackermann', 'ackermann: A must be square')
  const { data: b, n: inputs } = dense.toMatrixF64(plant.B, 'ackermann B', n)
  if (inputs !== 1) throw new DomainError('ackermann', 'ackermann: single-input systems only')
  const phi = polyFromRoots(desired, { real: true })
  const alpha = toFlat(phi)
  if (alpha.length !== n + 1) throw new ShapeError('ackermann', `ackermann: need ${n} poles, got ${alpha.length - 1}`)
  // 𝒞 (n×n, row-major): column j is Aʲb.
  const C = new Float64Array(n * n)
  let col: ArrayLike<number> = b
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) C[i * n + j] = col[i]
    col = dense.matVec(A, col, n, n)
  }
  const S = toFlat(svd(fromData(C, [n, n])).S)
  const conditioning = S[0] / S[S.length - 1]
  const rank = S.filter((s) => s > n * Number.EPSILON * S[0]).length
  const empty = fromData(new Float64Array(0), [0], 'complex128')
  const failed = { K: null, characteristic: phi, closedLoop: empty, controllable: false }
  if (rank < n) return { ...failed, conditioning }
  // φ(A) = Aⁿ + α₁Aⁿ⁻¹ + … + αₙI by Horner's rule.
  let P: ArrayLike<number> = new Float64Array(n * n)
  const I = dense.identity(n)
  for (const c of alpha) P = dense.axpy(c, I, dense.matMul(P, A, n, n, n))
  // Row vector eₙᵀ 𝒞⁻¹: solve 𝒞ᵀ w = eₙ.
  const Ct = new Float64Array(n * n)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Ct[i * n + j] = C[j * n + i]
  const en = new Float64Array(n)
  en[n - 1] = 1
  const w = solveDense(Ct, en, n).x
  if (!w) return { ...failed, conditioning }
  const K = dense.matMul(w, P, 1, n, n)
  const Acl = dense.sub(A, dense.matMul(b, K, n, 1, n))
  const cl = eig(fromData(Acl, [n, n]), { vectors: false })
  return {
    K: fromData(K, [1, n]),
    characteristic: phi,
    closedLoop: cl.values,
    controllable: true,
    conditioning,
  }
}
