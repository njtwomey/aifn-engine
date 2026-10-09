/**
 * The linear-quadratic regulator, optimal state feedback $\uvec = -\Kmat\xvec$ for a linear plant: `lqr` in
 * continuous time (by Kleinman iteration or the Hamiltonian matrix sign function) and `dlqr` in discrete time (by the
 * Riccati recursion or doubling), with `closedLoopPoles` to check the result (Kalman, 1960, "Contributions to the
 * theory of optimal control", Bol. Soc. Mat. Mexicana 5; Anderson & Moore, 1990, "Optimal Control: Linear Quadratic
 * Methods", §2–3).
 *
 * Both solve an algebraic Riccati equation with the step algorithms of `aifn-compute/numerics/linalg`, run to
 * convergence here. A solver that does not converge is reported in the result (`converged`, `failure`), not thrown.
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

/**
 * The plant of a state-feedback problem: $\xvec' = \Amat\xvec + \Bmat\uvec$ (or
 * $\xvec_{k+1} = \Amat\xvec_k + \Bmat\uvec_k$), with `A` the $n \times n$ dynamics and `B` the $n \times m$ input
 * matrix.
 */
export type StateFeedbackPlant = { A: MatrixLike; B: MatrixLike }

/** The solution of an LQR problem. */
export type LqrResult = {
  /** The stabilising Riccati solution $\Pmat$ (the optimal cost from $\xvec_0$ is $\xvec_0^\top\Pmat\xvec_0$). */
  P: Matrix
  /** The optimal gain $\Kmat$ ($m \times n$): $\uvec = -\Kmat\xvec$. */
  K: Matrix
  /** Eigenvalues of $\Amat - \Bmat\Kmat$, a complex128 vector of $n$ (empty when the gain is not finite). */
  closedLoop: Tensor
  /** The Riccati residual at $\Pmat$, relative to $1 + \max \lvert P_{ij} \rvert$. */
  residual: Scalar
  /** Steps the Riccati solver took. */
  steps: Size
  /** Whether the solver met its tolerance within the allowed steps. */
  converged: boolean
  /** Why the solver stopped early, or null. */
  failure: RiccatiFailure | null
}

/**
 * The eigenvalues of the closed loop $\Amat - \Bmat\Kmat$ under $\uvec = -\Kmat\xvec$, as a complex128 vector; empty
 * when an entry of $\Amat - \Bmat\Kmat$ is not finite. Shapes that do not agree throw `ShapeError`.
 *
 * @param A The plant's dynamics matrix $\Amat$ ($n \times n$).
 * @param B The plant's input matrix $\Bmat$ ($n \times m$).
 * @param K The feedback gain $\Kmat$ ($m \times n$).
 * @returns The $n$ closed-loop poles (stable in continuous time when every real part is negative; in discrete time
 *   when every modulus is below 1).
 *
 * @example The poles of a double integrator under a chosen gain
 * // A − BK = [[0, 1], [−2, −3]], whose characteristic polynomial s² + 3s + 2 has roots −1 and −2.
 * print('poles =', closedLoopPoles([[0, 1], [0, 0]], [[0], [1]], [[2, 3]]))
 */
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

/**
 * Assemble an `LqrResult` from a Riccati solver's final state, adding the closed-loop poles.
 *
 * @param plant The plant the problem was solved for, used for the closed-loop poles.
 * @param s The solver's state after the run.
 * @returns The solution, gain, poles and the solver's report.
 */
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
 * The continuous-time LQR: minimise $\int_0^\infty (\xvec^\top\Qmat\xvec + \uvec^\top\Rmat\uvec) \, dt$ subject to
 * $\xvec' = \Amat\xvec + \Bmat\uvec$. The optimum is $\uvec = -\Kmat\xvec$ with $\Kmat = \Rmat^{-1}\Bmat^\top\Pmat$ and
 * $\Pmat$ the stabilising solution of the continuous algebraic Riccati equation (CARE)
 * $\Amat^\top\Pmat + \Pmat\Amat - \Pmat\Bmat\Rmat^{-1}\Bmat^\top\Pmat + \Qmat = 0$ (Kalman, 1960). Solved by
 * `kleinmanIteration` (default) or `riccatiMatrixSign`, for at most `maxSteps` (default 100) steps. A solver that
 * stops short is reported by `converged` and `failure`, not thrown.
 *
 * @param plant The plant: `A` ($n \times n$) and `B` ($n \times m$).
 * @param Q The state weight $\Qmat$ ($n \times n$, symmetric positive semi-definite).
 * @param R The input weight $\Rmat$ ($m \times m$, symmetric positive definite).
 * @param options `method`, the solver: `'kleinman'` (default) or `'sign'`; `maxSteps`, the most solver steps to run
 *   (default 100); and `tolerance`, the solver's stopping tolerance on the relative residual and on the change in
 *   $\Pmat$ (default $10^{-12}$).
 * @returns The Riccati solution `P`, the gain `K`, the closed-loop poles and the solver's report.
 *
 * @example An unstable scalar plant
 * // x' = x + u with Q = R = 1: the CARE 2P − P² + 1 = 0 gives P = K = 1 + √2.
 * const { P, K, closedLoop, converged } = lqr({ A: [[1]], B: [[1]] }, [[1]], [[1]])
 * print('P =', P)
 * print('K =', K)
 * print('closed-loop pole =', closedLoop)
 * print('converged =', converged)
 *
 * @example A double integrator
 * // x'' = u with Q = I and R = 1: the gain is K = [1, √3], and both solvers agree.
 * const plant = { A: [[0, 1], [0, 0]], B: [[0], [1]] }
 * const Q = [[1, 0], [0, 1]]
 * print('Kleinman K =', lqr(plant, Q, [[1]]).K)
 * print('sign K =', lqr(plant, Q, [[1]], { method: 'sign' }).K)
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
 * The discrete-time LQR: minimise $\sum_k (\xvec_k^\top\Qmat\xvec_k + \uvec_k^\top\Rmat\uvec_k)$ subject to
 * $\xvec_{k+1} = \Amat\xvec_k + \Bmat\uvec_k$. The optimum is $\uvec_k = -\Kmat\xvec_k$ with
 * $\Kmat = (\Rmat + \Bmat^\top\Pmat\Bmat)^{-1}\Bmat^\top\Pmat\Amat$ and $\Pmat$ the stabilising solution of the
 * discrete algebraic Riccati equation (DARE). Solved by `riccatiDoubling` (default) or `riccatiRecursion`, for at most
 * `maxSteps` steps (default 100 for doubling, 10 000 for the recursion). A solver that stops short is reported by
 * `converged` and `failure`, not thrown.
 *
 * @param plant The plant: `A` ($n \times n$) and `B` ($n \times m$).
 * @param Q The state weight $\Qmat$ ($n \times n$, symmetric positive semi-definite).
 * @param R The input weight $\Rmat$ ($m \times m$, symmetric positive definite).
 * @param options `method`, the solver: `'doubling'` (default) or `'recursion'`; `maxSteps`, the most solver steps to
 *   run (default 100 for doubling, 10 000 for the recursion); and `tolerance`, the solver's stopping tolerance on the
 *   relative residual and on the change in $\Pmat$ (default $10^{-12}$).
 * @returns The Riccati solution `P`, the gain `K`, the closed-loop poles and the solver's report.
 *
 * @example A scalar integrator
 * // x_{k+1} = x_k + u_k with Q = R = 1: the DARE gives P² = P + 1, the golden ratio, and K = P / (1 + P).
 * const { P, K, closedLoop } = dlqr({ A: [[1]], B: [[1]] }, [[1]], [[1]])
 * print('P =', P)
 * print('K =', K)
 * print('closed-loop pole =', closedLoop)
 *
 * @example Doubling and the plain recursion agree
 * const plant = { A: [[1, 1], [0, 1]], B: [[0], [1]] }
 * const Q = [[1, 0], [0, 1]]
 * const doubling = dlqr(plant, Q, [[1]])
 * const recursion = dlqr(plant, Q, [[1]], { method: 'recursion' })
 * print('doubling K =', doubling.K, 'in', doubling.steps, 'steps')
 * print('recursion K =', recursion.K, 'in', recursion.steps, 'steps')
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
