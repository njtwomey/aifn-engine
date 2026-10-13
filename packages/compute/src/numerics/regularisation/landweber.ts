/**
 * Landweber iteration, regularisation by early stopping: gradient descent on $\tfrac12\lVert \Amat\xvec - \bvec
 * \rVert^2$ with a fixed step $\omega$,
 *
 * $$\xvec_{k+1} = \xvec_k + \omega\,\Amat^\top(\bvec - \Amat\xvec_k), \qquad 0 < \omega < 2/s_1^2.$$
 *
 * From $\xvec_0 = \zeros$ the $k$-th iterate is a spectral filter like Tikhonov's,
 * $\xvec_k = \sum_i \varphi_i^{(k)} (\beta_i/s_i)\vvec_i$ with $\varphi_i^{(k)} = 1 - (1 - \omega s_i^2)^k$, so the
 * number of steps plays the part of the weight: roughly $k \approx 1/(\omega\lambda)$. Run long enough it converges
 * to the minimum-norm least-squares solution, and on noisy data the error first falls and then rises
 * (semi-convergence).
 * Landweber (1951), "An iteration formula for Fredholm integral equations of the first kind", Amer. J. Math. 73(3);
 * Hansen (1998), §6.1.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { svd } from 'aifn-compute/numerics/linalg'

/** The problem of `landweberSteps`. */
export type LandweberProblem = {
  /** The matrix $\Amat$, $m \times n$. Copied once. */
  A: MatrixLike
  /** The right-hand side $\bvec$, $m$ values. Copied once. */
  b: VectorLike
  /**
   * The step $\omega > 0$ (default $1/s_1^2$, $s_1$ the largest singular value of $\Amat$). The iteration converges
   * for $0 < \omega < 2/s_1^2$; a larger step makes it diverge, which is flagged.
   */
  step?: number
  /**
   * Stop as converged when $\lVert \Amat^\top(\bvec - \Amat\xvec) \rVert$ is at most `tolerance` times
   * $\lVert \Amat^\top\bvec \rVert$ (default 1e-10): the least-squares solution is reached.
   */
  tolerance?: number
}

/** One state of `landweberSteps`. */
export interface LandweberState extends Status {
  /** The iterate $\xvec_k$, $n$ values. */
  x: Tensor
  /** $\lVert \Amat\xvec_k - \bvec \rVert$. */
  residualNorm: number
  /** $\lVert \xvec_k \rVert$. */
  solutionNorm: number
  /** $\lVert \Amat^\top(\bvec - \Amat\xvec_k) \rVert$, the norm of the (negative) gradient. */
  gradientNorm: number
  /** The step $\omega$. */
  step: number
  /**
   * The filter factors $1 - (1 - \omega s_i^2)^k$ of the singular values $s_i$ (descending), $\min(m, n)$ values:
   * the iterate's spectral filter when started from $\zeros$.
   */
  filterFactors: Tensor
  /** True once the gradient norm is within the tolerance. */
  converged: boolean
  /** True when the iterate is no longer finite or the residual grew (the step is too large). */
  diverged: boolean
}

/**
 * Landweber iteration as a step-through algorithm: each step is $\xvec \leftarrow \xvec +
 * \omega\Amat^\top(\bvec - \Amat\xvec)$, and each state carries the residual and solution norms and the equivalent
 * filter factors $1 - (1 - \omega s_i^2)^k$, to set against Tikhonov's $s_i^2/(s_i^2 + \lambda)$. The residual norm
 * never increases for $0 < \omega < 2/s_1^2$; a state where it does, or whose iterate is not finite, is flagged
 * `diverged`. `init` takes `{ x0 }` (default zeros). Throws `DomainError` for a step that is not positive or a zero
 * $\Amat$ and `ShapeError` for mismatched shapes.
 *
 * @param problem The matrix, right-hand side, step and tolerance, as `LandweberProblem`.
 * @returns The algorithm: `init` takes `{ x0 }`, the start of $n$ values (default zeros), and each `step` is one
 *   Landweber update. It is done when converged or diverged.
 *
 * @example Few steps damp the small singular values, as a large Tikhonov weight does
 * const A = tensor([[1, 0], [0, 0.1]])
 * const b = tensor([1, 0.1])
 * for (const k of [1, 10, 100]) {
 *   const s = run(landweberSteps({ A, b }), undefined, k)
 *   print('k =', k, ' x =', s.x, ' filter =', s.filterFactors)
 * }
 *
 * @example Too large a step diverges
 * const A = tensor([[1, 0], [0, 0.1]])
 * const s = run(landweberSteps({ A, b: tensor([1, 1]), step: 2.5 }), undefined, 20)
 * print('diverged =', s.diverged, ' after', s.t, 'steps')
 */
export function landweberSteps(problem: LandweberProblem): Algorithm<{ x0?: VectorLike } | void, LandweberState> {
  const where = 'landweberSteps'
  const { data: a, m, n } = dense.toMatrixF64(problem.A, where)
  const b = dense.toF64(problem.b, where)
  if (b.length !== m) throw new ShapeError(where, `${where}: A has ${m} rows but b has ${b.length} entries`)
  const s = dense.data(svd(dense.mat(a, m, n)).S)
  if (!(s[0] > 0)) throw new DomainError(where, `${where}: A is zero`)
  const omega = problem.step ?? 1 / (s[0] * s[0])
  if (!(omega > 0 && Number.isFinite(omega)))
    throw new DomainError(where, `${where}: step must be positive and finite, got ${omega}`)
  const bNorm = dense.norm(b)
  const scale = dense.norm(dense.matTVec(a, b, m, n))
  const tol = (problem.tolerance ?? 1e-10) * (scale > 0 ? scale : 1)
  const state = (t: number, x: dense.F64, previous: number): LandweberState => {
    const r = dense.sub(b, dense.matVec(a, x, m, n))
    const g = dense.matTVec(a, r, m, n)
    const residualNorm = dense.norm(r)
    const gradientNorm = dense.norm(g)
    const finite = dense.allFinite(x) && Number.isFinite(residualNorm)
    const grew = residualNorm > previous * (1 + 1e-8) + 1e-14 * bNorm
    return {
      t,
      x: dense.vec(x),
      residualNorm,
      solutionNorm: dense.norm(x),
      gradientNorm,
      step: omega,
      filterFactors: dense.vec(s.map((si) => 1 - (1 - omega * si * si) ** t)),
      converged: finite && !grew && gradientNorm <= tol,
      diverged: !finite || grew,
    }
  }
  return {
    name: 'landweber',
    init: (start) => {
      const x = start && start.x0 !== undefined ? dense.toF64(start.x0, where) : new Float64Array(n)
      if (x.length !== n) throw new ShapeError(where, `${where}: x0 has ${x.length} entries, A has ${n} columns`)
      return state(0, x, Infinity)
    },
    step: (st) => {
      if (st.diverged) return { ...st, t: st.t + 1 }
      const x = dense.data(st.x)
      const g = dense.matTVec(a, dense.sub(b, dense.matVec(a, x, m, n)), m, n)
      return state(st.t + 1, dense.axpy(omega, g, x), st.residualNorm)
    },
    done: (st) => st.converged || st.diverged,
  }
}
