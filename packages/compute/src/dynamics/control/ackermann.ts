/**
 * Pole placement by Ackermann's formula, part of `aifn-compute/dynamics/control` (Ackermann, 1972, "Der Entwurf linearer
 * Regelungssysteme im Zustandsraum", Regelungstechnik 20; Kailath, 1980, "Linear Systems", §3.2).
 */

import { eig, solveDense, svd } from 'aifn-compute/numerics/linalg'
import { dense, fromData, toFlat, type Matrix, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import type { Scalar } from 'aifn-compute/foundation/contracts'
import { polyFromRoots, type ComplexLike } from 'aifn-compute/numerics/polynomial'
import type { StateFeedbackPlant } from './lqr'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The result of `ackermann`. */
export type PolePlacement = {
  /** The gain K (1×n) with eig(A − BK) = the requested poles, or null when (A, b) is not controllable. */
  K: Matrix | null
  /** The desired characteristic polynomial [1, α₁, …, αₙ]. */
  characteristic: Vector
  /** The eigenvalues of A − BK actually obtained, complex128 (empty when K is null). */
  closedLoop: Tensor
  /** Whether the controllability matrix has full rank. */
  controllable: boolean
  /** The condition number σ_max/σ_min of the controllability matrix; large means the gain is sensitive. */
  conditioning: Scalar
}

/**
 * Single-input pole placement by Ackermann's formula (Ackermann, 1972): K = [0 … 0 1] 𝒞⁻¹ φ(A), where
 * 𝒞 = [b, Ab, …, Aⁿ⁻¹b] is the controllability matrix and φ(s) = Πᵢ(s − pᵢ) the desired characteristic polynomial,
 * so that u = −Kx gives eig(A − bK) = {pᵢ}. Numerically poor for large n or nearly uncontrollable pairs (it inverts
 * 𝒞): the conditioning is reported, and the achieved poles are returned for checking.
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
