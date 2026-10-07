/**
 * Sparse approximation by optimisation: basis pursuit, basis pursuit denoising (the lasso) and iterative hard
 * thresholding, each a thin layer over a solver of `aifn-compute/optim`.
 *
 * - Basis pursuit (Chen, Donoho and Saunders, 1998) is the convex relaxation of the sparsest exact representation,
 *   $\min \lVert \xvec \rVert_1$ subject to $\Dmat\xvec = \yvec$. Splitting $\xvec = \uvec - \vvec$ with
 *   $\uvec, \vvec \ge 0$ makes it the linear program $\min \mathbf{1}^\top(\uvec + \vvec)$ subject to
 *   $\Dmat(\uvec - \vvec) = \yvec$, solved by `linprog`.
 * - Basis pursuit denoising, the lasso in Lagrangian form (Tibshirani, 1996), trades the fit against sparsity:
 *   $\min_\xvec \frac{1}{2}\lVert \yvec - \Dmat\xvec \rVert^2 + \lambda \lVert \xvec \rVert_1$. It is solved by
 *   proximal gradient descent with soft thresholding (`proximalGradient` with `proxL1`; FISTA by default, ISTA
 *   without acceleration), with step $1/L$ for $L = \sigma_{\max}(\Dmat)^2$, the Lipschitz constant of the gradient.
 *   For $\lambda \ge \lVert \Dmat^\top\yvec \rVert_\infty$ the solution is $\xvec = \mathbf{0}$.
 * - Iterative hard thresholding (Blumensath and Davies, 2009) attacks the non-convex problem $\min \lVert \yvec -
 *   \Dmat\xvec \rVert^2$ subject to at most $s$ non-zeros directly: the same gradient step, followed by keeping the
 *   $s$ largest entries (`hardThreshold`) instead of soft thresholding.
 */

import type { Index, MatrixLike, ObjectiveFn, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { svd } from 'aifn-compute/numerics/linalg'
import { linprog, type LinearProgramStatus, type LinprogOptions } from 'aifn-compute/optim/programming'
import { proximalGradient, proxL1, type Prox, type ProximalGradientState } from 'aifn-compute/optim/proximal'
import { hardThreshold, readDictionary, readSignal, type Dictionary } from './atoms'
import type { SparseApproximation } from './pursuit'

type F64 = dense.F64

/** The result of `basisPursuit`: a sparse approximation, with the linear program's outcome. */
export type BasisPursuitResult = SparseApproximation & {
  /** The linear program's status: `optimal`, or `infeasible` when $\yvec$ is not in the range of $\Dmat$. */
  status: LinearProgramStatus
  /** $\lVert \xvec \rVert_1$ at the solution; NaN unless `status` is `optimal`. */
  norm1: number
}

/** Options of `basisPursuitDenoisingSteps` and `basisPursuitDenoising`. */
export type BasisPursuitDenoisingOptions = {
  /** The weight $\lambda > 0$ of the $\ell_1$ penalty: larger gives sparser coefficients and a looser fit. */
  lambda: number
  /** FISTA's extrapolation (default true); false runs plain ISTA. */
  accelerated?: boolean
  /** Stop when the gradient mapping's norm is at most this. Default $10^{-8}$. */
  tolerance?: number
}

/** Options of `iterativeHardThresholdingSteps` and `iterativeHardThresholding`. */
export type HardThresholdingOptions = {
  /** The number $s$ of non-zero coefficients to keep, from 1 to $k$. */
  sparsity: Size
  /** Stop when the gradient mapping's norm is at most this. Default $10^{-8}$. */
  tolerance?: number
}

/**
 * The least-squares part $f(\xvec) = \frac{1}{2}\lVert \yvec - \Dmat\xvec \rVert^2$ of the objectives, with its
 * gradient $\Dmat^\top(\Dmat\xvec - \yvec)$, and the step $1/L$ that makes the proximal gradient step a descent step.
 *
 * @param d The dictionary.
 * @param signal The signal $\yvec$, $m$ values.
 * @returns The objective function and the Lipschitz constant $L = \sigma_{\max}(\Dmat)^2$ of its gradient.
 */
function leastSquares(d: Dictionary, signal: F64): { f: ObjectiveFn; lipschitz: number } {
  const f: ObjectiveFn = (x) => {
    const r = dense.sub(dense.matVec(d.data, dense.data(x), d.m, d.k), signal)
    return { value: 0.5 * dense.dot(r, r), grad: dense.matTVec(d.data, r, d.m, d.k) }
  }
  const sigma = svd(dense.mat(d.data, d.m, d.k)).S.data[0] ?? 0
  return { f, lipschitz: Math.max(sigma * sigma, Number.MIN_VALUE) }
}

/**
 * A proximal gradient method started from $\xvec = \mathbf{0}$, so it needs no start.
 *
 * @param name The algorithm's name.
 * @param k The number of coefficients.
 * @param alg The proximal gradient method, which starts from `{ x0 }`.
 * @returns The same algorithm with the start fixed at zero.
 */
function fromZero(
  name: string,
  k: Size,
  alg: Algorithm<{ x0: VectorLike }, ProximalGradientState>,
): Algorithm<void, ProximalGradientState> {
  return { ...alg, name, init: (_start, stream) => alg.init({ x0: new Float64Array(k) }, stream) }
}

/**
 * The approximation a final proximal gradient state describes.
 *
 * @param d The dictionary.
 * @param signal The signal $\yvec$.
 * @param x The coefficients.
 * @param steps The steps taken.
 * @returns The coefficients, the residual and its norm, the non-zero entries and the steps.
 */
function approximation(d: Dictionary, signal: F64, x: F64, steps: Size): SparseApproximation {
  const residual = dense.sub(signal, dense.matVec(d.data, x, d.m, d.k))
  const support: Index[] = []
  x.forEach((v, j) => v !== 0 && support.push(j))
  return { x: dense.vec(x), residual: dense.vec(residual), residualNorm: dense.norm(residual), support, steps }
}

/**
 * Basis pursuit (Chen, Donoho and Saunders, 1998): the representation $\yvec = \Dmat\xvec$ of least $\ell_1$ norm,
 * as a linear program solved by `linprog`. When $\yvec = \Dmat\xvec^\star$ for a sparse enough $\xvec^\star$ (fewer
 * than $\frac{1}{2}(1 + 1/\mu)$ non-zeros, $\mu$ the `mutualCoherence`), the solution is $\xvec^\star$ (Donoho and
 * Elad, 2003). A signal outside the range of $\Dmat$ has no exact representation, and `status` reports it
 * `infeasible`.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$ to represent, $m$ values.
 * @param options Passed to `linprog`: the method (simplex by default), its tolerance and step budget.
 * @returns The coefficients (NaN unless `status` is `optimal`), the residual, the support (the entries above
 *   $10^{-9}$ times the largest), $\lVert \xvec \rVert_1$, the linear program's status and its steps.
 *
 * @example One diagonal atom rather than two spikes
 * // y = e1 + e2 is also sqrt(2) times atom 3, which has the smaller l1 norm.
 * const s = Math.SQRT1_2
 * const D = [[1, 0, 0, s], [0, 1, 0, s], [0, 0, 1, 0]]
 * const res = basisPursuit(D, [1, 1, 0])
 * print('status =', res.status, ' support =', res.support)
 * print('x =', res.x, ' l1 norm =', res.norm1)
 *
 * @example A signal outside the range is reported
 * const res = basisPursuit([[1, 2], [2, 4]], [1, 0])
 * print('status =', res.status)
 */
export function basisPursuit(D: MatrixLike, y: VectorLike, options: LinprogOptions = {}): BasisPursuitResult {
  const d = readDictionary(D, 'basisPursuit')
  const signal = readSignal(y, d.m, 'basisPursuit')
  // x = u − v with u, v ≥ 0 (linprog's default bounds): A_eq = [D, −D], c = 1.
  const A_eq = Array.from({ length: d.m }, (_, i) => {
    const row = d.data.subarray(i * d.k, (i + 1) * d.k)
    return [...row, ...row.map((v) => -v)]
  })
  const lp = linprog({ c: new Float64Array(2 * d.k).fill(1), A_eq, b_eq: signal }, options)
  const uv = dense.data(lp.x)
  const x = Float64Array.from({ length: d.k }, (_, j) => uv[j] - uv[d.k + j])
  const result = approximation(d, signal, x, lp.steps)
  const cutoff = 1e-9 * dense.maxAbs(x)
  return {
    ...result,
    support: result.support.filter((j) => Math.abs(x[j]) > cutoff),
    status: lp.status,
    norm1: lp.status === 'optimal' ? x.reduce((a, v) => a + Math.abs(v), 0) : NaN,
  }
}

/**
 * Basis pursuit denoising, the lasso, as a step-through algorithm: proximal gradient descent on
 * $\frac{1}{2}\lVert \yvec - \Dmat\xvec \rVert^2 + \lambda \lVert \xvec \rVert_1$ from $\xvec = \mathbf{0}$, with
 * soft thresholding as the proximal step and step size $1/\sigma_{\max}(\Dmat)^2$. FISTA (Beck and Teboulle, 2009)
 * by default, so the objective's gap to its minimum falls as $O(1/t^2)$; ISTA ($O(1/t)$) without acceleration.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The penalty $\lambda$, the acceleration and the tolerance.
 * @returns The algorithm; its state is that of `proximalGradient`, with the coefficients in `x` and the objective in
 *   `value`.
 *
 * @example FISTA against ISTA
 * const D = [[1, 0, 0.6], [0, 1, 0.8]]
 * const y = [0.9, 1.3]
 * for (const accelerated of [true, false]) {
 *   const s = run(basisPursuitDenoisingSteps(D, y, { lambda: 0.1, accelerated }), undefined, 25)
 *   print(accelerated ? 'FISTA' : 'ISTA', 'after 25 steps: objective =', s.value)
 * }
 */
export function basisPursuitDenoisingSteps(
  D: MatrixLike,
  y: VectorLike,
  options: BasisPursuitDenoisingOptions,
): Algorithm<void, ProximalGradientState> {
  const { lambda, accelerated = true, tolerance = 1e-8 } = options
  if (!(lambda > 0))
    throw new DomainError('basisPursuitDenoising', `basisPursuitDenoising: lambda must be positive, got ${lambda}`)
  const d = readDictionary(D, 'basisPursuitDenoising')
  const { f, lipschitz } = leastSquares(d, readSignal(y, d.m, 'basisPursuitDenoising'))
  const alg = proximalGradient(f, proxL1(lambda), { stepSize: 1 / lipschitz, accelerated, tolerance })
  return fromZero('basisPursuitDenoising', d.k, alg)
}

/**
 * Basis pursuit denoising, the lasso, run to the tolerance or for at most `maxSteps` steps: the one-call form of
 * `basisPursuitDenoisingSteps`. The larger $\lambda$, the fewer non-zeros; from
 * $\lambda = \lVert \Dmat^\top\yvec \rVert_\infty$ up, none.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The penalty $\lambda$, the acceleration, the tolerance and the step budget.
 * @param options.maxSteps The most steps to take. Default 1000.
 * @returns The coefficients, the residual, its norm, the non-zero entries, the steps taken, the objective's value
 *   and whether the tolerance was met.
 *
 * @example More penalty, fewer atoms
 * const D = [[1, 0, 0.6], [0, 1, 0.8]]
 * for (const lambda of [0.01, 0.3, 1, 2]) {
 *   const res = basisPursuitDenoising(D, [0.9, 1.3], { lambda })
 *   print(`lambda = ${lambda}: x =`, res.x, ' support =', res.support)
 * }
 */
export function basisPursuitDenoising(
  D: MatrixLike,
  y: VectorLike,
  options: BasisPursuitDenoisingOptions & { maxSteps?: Size },
): SparseApproximation & { objective: number; converged: boolean } {
  const s = run(basisPursuitDenoisingSteps(D, y, options), undefined, options.maxSteps ?? 1000)
  const d = readDictionary(D, 'basisPursuitDenoising')
  const result = approximation(d, readSignal(y, d.m, 'basisPursuitDenoising'), dense.data(s.x), s.t)
  return { ...result, objective: s.value, converged: s.converged }
}

/**
 * Iterative hard thresholding (Blumensath and Davies, 2009) as a step-through algorithm: from $\xvec = \mathbf{0}$,
 * $\xvec \leftarrow H_s(\xvec + \eta\Dmat^\top(\yvec - \Dmat\xvec))$ with $\eta = 1/\sigma_{\max}(\Dmat)^2$ and
 * $H_s$ the `hardThreshold` to $s$ entries. Every iterate has at most $s$ non-zeros and the squared error never
 * increases; the limit is a local minimiser of the non-convex problem, which is the sparse solution when $\Dmat$ is
 * close enough to orthogonal on sparse vectors (the restricted isometry property).
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The sparsity $s$ and the tolerance.
 * @returns The algorithm; its state is that of `proximalGradient`, with the coefficients in `x` and the squared error
 *   (halved) in `value`.
 *
 * @example Every iterate keeps two entries
 * const D = [[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.7]]
 * const alg = iterativeHardThresholdingSteps(D, [0, 2, -3], { sparsity: 2 })
 * for (const t of [1, 5, 50]) print(`after ${t} steps: x =`, run(alg, undefined, t).x)
 */
export function iterativeHardThresholdingSteps(
  D: MatrixLike,
  y: VectorLike,
  options: HardThresholdingOptions,
): Algorithm<void, ProximalGradientState> {
  const { sparsity: s, tolerance = 1e-8 } = options
  const d = readDictionary(D, 'iterativeHardThresholding')
  if (!(Number.isInteger(s) && s >= 1 && s <= d.k))
    throw new DomainError(
      'iterativeHardThresholding',
      `iterativeHardThresholding: sparsity must be an integer from 1 to ${d.k}, got ${s}`,
    )
  const { f, lipschitz } = leastSquares(d, readSignal(y, d.m, 'iterativeHardThresholding'))
  // The indicator of at most s non-zeros, whose "prox" (a projection onto a non-convex set) is hard thresholding.
  const g: Prox = {
    name: 'hard-threshold',
    value: (x) => (dense.data(x).reduce((n, v) => n + (v !== 0 ? 1 : 0), 0) <= s ? 0 : Infinity),
    prox: (v) => hardThreshold(v, s),
  }
  const alg = proximalGradient(f, g, { stepSize: 1 / lipschitz, tolerance })
  return fromZero('iterativeHardThresholding', d.k, alg)
}

/**
 * Iterative hard thresholding run to the tolerance or for at most `maxSteps` steps: the one-call form of
 * `iterativeHardThresholdingSteps`.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The sparsity $s$, the tolerance and the step budget.
 * @param options.maxSteps The most steps to take. Default 1000.
 * @returns The coefficients (at most $s$ non-zeros), the residual, its norm, the non-zero entries, the steps taken
 *   and whether the tolerance was met.
 *
 * @example Recover a 2-sparse vector
 * const D = [[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.7]]
 * const res = iterativeHardThresholding(D, [0, 2, -3], { sparsity: 2 })
 * print('x =', res.x, ' residual norm =', res.residualNorm)
 *
 * @example Caught in a local minimum
 * // y = 2 d1 + 3 d2 exactly, but d3 correlates with y best, so the first step keeps d3 and the method settles on a
 * // worse pair.
 * const D = [[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.7]]
 * const res = iterativeHardThresholding(D, [0, 2, 3], { sparsity: 2 })
 * print('x =', res.x, ' residual norm =', res.residualNorm)
 */
export function iterativeHardThresholding(
  D: MatrixLike,
  y: VectorLike,
  options: HardThresholdingOptions & { maxSteps?: Size },
): SparseApproximation & { converged: boolean } {
  const s = run(iterativeHardThresholdingSteps(D, y, options), undefined, options.maxSteps ?? 1000)
  const d = readDictionary(D, 'iterativeHardThresholding')
  const result = approximation(d, readSignal(y, d.m, 'iterativeHardThresholding'), dense.data(s.x), s.t)
  return { ...result, converged: s.converged }
}
