/**
 * `aifn-compute/dynamics/control`: optimal state feedback. The linear-quadratic regulator (`lqr`, continuous time, by
 * Kleinman iteration or the Hamiltonian matrix sign function) and its discrete-time form (`dlqr`, by the Riccati
 * recursion or doubling), built on the Riccati solvers of `aifn-compute/numerics/linalg` (Kalman, 1960, "Contributions to the
 * theory of optimal control", Bol. Soc. Mat. Mexicana 5; Anderson & Moore, 1990, "Optimal Control: Linear Quadratic
 * Methods", §2–3).
 */

import { eig } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Matrix, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size } from 'aifn-compute/foundation/contracts'
import {
  kleinmanIteration,
  riccatiDoubling,
  riccatiMatrixSign,
  riccatiRecursion,
  type RiccatiFailure,
  type RiccatiOptions,
  type RiccatiState,
} from 'aifn-compute/numerics/linalg'

/** The plant of a state-feedback problem: x′ = Ax + Bu (or x_{k+1} = Ax_k + Bu_k), A n×n and B n×m. */
export type StateFeedbackPlant = { A: MatrixLike; B: MatrixLike }

/** The solution of an LQR problem. */
export type LqrResult = {
  /** The stabilising Riccati solution P (the optimal cost is x₀ᵀPx₀). */
  P: Matrix
  /** The optimal gain: u = −Kx. */
  K: Matrix
  /** Eigenvalues of A − BK, complex128 [n]. */
  closedLoop: Tensor
  /** The relative Riccati residual at P. */
  residual: Scalar
  /** Steps the Riccati solver took. */
  steps: Size
  converged: boolean
  /** Why the solver stopped early, or null. */
  failure: RiccatiFailure | null
}

/** The eigenvalues of A − BK as a complex128 vector, empty when anything is not finite. */
export function closedLoopPoles(A: MatrixLike, B: MatrixLike, K: MatrixLike): Tensor {
  const a = dense.toMatrixF64(A, 'closedLoop A')
  const b = dense.toMatrixF64(B, 'closedLoop B', a.m)
  const k = dense.toMatrixF64(K, 'closedLoop K', b.n, a.n)
  const acl = dense.sub(a.data, dense.matMul(b.data, k.data, a.m, b.n, a.n))
  if (!dense.allFinite(acl)) {
    return fromData(new Float64Array(0), [0], 'complex128')
  }
  return eig(fromData(acl, [a.m, a.n]), { vectors: false }).values
}

function finishLqr(plant: StateFeedbackPlant, s: RiccatiState): LqrResult {
  return {
    P: s.P,
    K: s.K,
    closedLoop: closedLoopPoles(plant.A, plant.B, s.K),
    residual: s.residual,
    steps: s.t,
    converged: s.converged,
    failure: s.failure,
  }
}

/**
 * The continuous-time LQR: minimise ∫₀^∞ (xᵀQx + uᵀRu) dt subject to x′ = Ax + Bu. The optimum is u = −Kx with
 * K = R⁻¹BᵀP and P the stabilising solution of the CARE (Kalman, 1960). Solved by `kleinmanIteration` (default) or
 * `riccatiMatrixSign`, for at most `maxSteps` (default 100) steps.
 */
export function lqr(
  plant: StateFeedbackPlant,
  Q: MatrixLike,
  R: MatrixLike,
  options: RiccatiOptions & { method?: 'kleinman' | 'sign'; maxSteps?: Size } = {},
): LqrResult {
  const prob = { A: plant.A, B: plant.B, Q, R }
  const maxSteps = options.maxSteps ?? 100
  const s: RiccatiState =
    options.method === 'sign'
      ? run(riccatiMatrixSign(prob, options), undefined, maxSteps)
      : run(kleinmanIteration(prob, options), undefined, maxSteps)
  return finishLqr(plant, s)
}

/**
 * The discrete-time LQR: minimise Σ (x_kᵀQx_k + u_kᵀRu_k) subject to x_{k+1} = Ax_k + Bu_k; u = −Kx with
 * K = (R + BᵀPB)⁻¹BᵀPA and P the stabilising DARE solution. Solved by `riccatiDoubling` (default) or
 * `riccatiRecursion`, for at most `maxSteps` steps (default 100 for doubling, 10 000 for the recursion).
 */
export function dlqr(
  plant: StateFeedbackPlant,
  Q: MatrixLike,
  R: MatrixLike,
  options: RiccatiOptions & { method?: 'doubling' | 'recursion'; maxSteps?: Size } = {},
): LqrResult {
  const prob = { A: plant.A, B: plant.B, Q, R }
  const s: RiccatiState =
    options.method === 'recursion'
      ? run(riccatiRecursion(prob, options), undefined, options.maxSteps ?? 10000)
      : run(riccatiDoubling(prob, options), undefined, options.maxSteps ?? 100)
  return finishLqr(plant, s)
}
