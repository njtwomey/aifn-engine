/**
 * Greedy sparse approximation of a signal $\yvec \approx \Dmat\xvec$ over a dictionary $\Dmat$: matching pursuit
 * (Mallat and Zhang, 1993) and orthogonal matching pursuit (Pati, Rezaiifar and Krishnaprasad, 1993), each as a
 * step-through algorithm that adds one atom per step.
 *
 * Both keep a residual $\rvec = \yvec - \Dmat\xvec$, starting from $\xvec = \mathbf{0}$, and at each step pick the atom
 * most correlated with it, $j = \arg\max_j |\dvec_j^\top \rvec| / \lVert \dvec_j \rVert$. They differ in what follows:
 *
 * - Matching pursuit adds the projection onto that atom alone, $x_j \leftarrow x_j + \dvec_j^\top \rvec / \lVert
 *   \dvec_j \rVert^2$. An atom may be picked again, and the residual shrinks but is not orthogonal to the atoms used.
 * - Orthogonal matching pursuit adds the atom to a support $S$ and refits every coefficient on it by least squares,
 *   $\xvec_S = \arg\min \lVert \yvec - \Dmat_S \xvec_S \rVert$ (by `lstsq`), so the residual is orthogonal to every
 *   chosen atom, no atom is picked twice, and after $s$ steps $\xvec$ has exactly $s$ non-zeros.
 *
 * Both stop when $\lVert \rvec \rVert \le \tau \lVert \yvec \rVert$ for the `tolerance` $\tau$; orthogonal matching
 * pursuit also stops once it has `sparsity` atoms.
 */

import type { Index, MatrixLike, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor, type Vector } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { lstsq } from 'aifn-compute/numerics/linalg'
import { atomNorms, readDictionary, readSignal, type Dictionary } from './atoms'

type F64 = dense.F64

/** Options of the greedy pursuits. */
export type PursuitOptions = {
  /**
   * Orthogonal matching pursuit only: the number of atoms $s$ to select, at most $\min(m, k)$. Default $\min(m, k)$,
   * so only the tolerance stops it early.
   */
  sparsity?: Size
  /** Stop when $\lVert \rvec \rVert \le \tau \lVert \yvec \rVert$ for this $\tau$. Default $10^{-10}$. */
  tolerance?: number
}

/** The state of `matchingPursuitSteps` and `orthogonalMatchingPursuitSteps`. */
export interface PursuitState extends Status {
  /** Steps taken: atoms selected. */
  t: Size
  /** The coefficients $\xvec$, $k$ values. */
  x: Vector
  /** The residual $\rvec = \yvec - \Dmat\xvec$, $m$ values. */
  residual: Vector
  /** $\lVert \rvec \rVert$. */
  residualNorm: number
  /** The atoms selected so far, in the order they were picked (matching pursuit may list one more than once). */
  support: Index[]
  /** The normalised correlations $\dvec_j^\top \rvec / \lVert \dvec_j \rVert$ of the residual with every atom. */
  correlations: Vector
  /** Set when $\lVert \rvec \rVert \le \tau \lVert \yvec \rVert$. */
  converged: boolean
  /** Set when no atom is left that correlates with the residual, so another step would not change it. */
  stalled: boolean
}

/** A sparse approximation of a signal: the result of the one-call pursuits and solvers of this module. */
export type SparseApproximation = {
  /** The coefficients $\xvec$, $k$ values. */
  x: Tensor
  /** The residual $\yvec - \Dmat\xvec$, $m$ values. */
  residual: Tensor
  /** $\lVert \yvec - \Dmat\xvec \rVert$. */
  residualNorm: number
  /** The indices of the non-zero coefficients, ascending. */
  support: Index[]
  /** Steps taken by the solver. */
  steps: Size
}

/**
 * The normalised correlation of the residual with every atom.
 *
 * @param d The dictionary.
 * @param norms The norm of every atom; a zero atom gets correlation $0$.
 * @param r The residual, $m$ values.
 * @returns $\dvec_j^\top \rvec / \lVert \dvec_j \rVert$ for every atom $j$.
 */
function correlate(d: Dictionary, norms: F64, r: F64): F64 {
  const c = dense.matTVec(d.data, r, d.m, d.k)
  for (let j = 0; j < d.k; j++) c[j] = norms[j] > 0 ? c[j] / norms[j] : 0
  return c
}

/**
 * The atom most correlated with the residual among those allowed.
 *
 * @param c The normalised correlations.
 * @param skip Atoms that may not be picked (the support of orthogonal matching pursuit), or null.
 * @returns The index of the atom, or $-1$ when every allowed correlation is zero.
 */
function pick(c: F64, skip: ReadonlySet<Index> | null): Index {
  let best = -1
  let value = 0
  for (let j = 0; j < c.length; j++) {
    if (skip?.has(j)) continue
    if (Math.abs(c[j]) > value) {
      value = Math.abs(c[j])
      best = j
    }
  }
  return best
}

/**
 * The shared skeleton of the two pursuits: reading the inputs, the initial state and the stopping tests.
 *
 * @param name The algorithm's name, for its trace and error messages.
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$, $m$ values.
 * @param options The tolerance, and for orthogonal matching pursuit the sparsity.
 * @param orthogonal Whether to refit on the support by least squares (orthogonal matching pursuit) or not.
 * @returns The step-through algorithm.
 */
function pursuit(
  name: string,
  D: MatrixLike,
  y: VectorLike,
  options: PursuitOptions,
  orthogonal: boolean,
): Algorithm<void, PursuitState> {
  const d = readDictionary(D, name)
  const signal = readSignal(y, d.m, name)
  const norms = atomNorms(d)
  const { tolerance = 1e-10 } = options
  const sparsity = options.sparsity ?? Math.min(d.m, d.k)
  if (!(Number.isInteger(sparsity) && sparsity >= 1 && sparsity <= Math.min(d.m, d.k)))
    throw new DomainError(name, `${name}: sparsity must be an integer from 1 to ${Math.min(d.m, d.k)}, got ${sparsity}`)
  const target = tolerance * dense.norm(signal)
  const state = (t: Size, x: F64, r: F64, support: Index[]): PursuitState => {
    const residualNorm = dense.norm(r)
    const correlations = correlate(d, norms, r)
    const left = pick(correlations, orthogonal ? new Set(support) : null)
    return {
      t,
      x: dense.vec(x),
      residual: dense.vec(r),
      residualNorm,
      support,
      correlations: dense.vec(correlations),
      converged: residualNorm <= target,
      stalled: residualNorm > target && left < 0,
    }
  }
  return {
    name,
    init: () => state(0, new Float64Array(d.k), signal, []),
    step: (s) => {
      const c = dense.data(s.correlations)
      const j = pick(c, orthogonal ? new Set(s.support) : null)
      const support = [...s.support, j]
      if (!orthogonal) {
        const x = Float64Array.from(dense.data(s.x))
        const alpha = c[j] / norms[j]
        x[j] += alpha
        const r = Float64Array.from(dense.data(s.residual))
        for (let i = 0; i < d.m; i++) r[i] -= alpha * d.data[i * d.k + j]
        return state(s.t + 1, x, r, support)
      }
      const sub = new Float64Array(d.m * support.length)
      for (let i = 0; i < d.m; i++)
        for (let a = 0; a < support.length; a++) sub[i * support.length + a] = d.data[i * d.k + support[a]]
      const coef = dense.data(lstsq(dense.mat(sub, d.m, support.length), dense.vec(signal)).x)
      const x = new Float64Array(d.k)
      support.forEach((atom, a) => (x[atom] = coef[a]))
      const r = dense.sub(signal, dense.matVec(sub, coef, d.m, support.length))
      return state(s.t + 1, x, r, support)
    },
    done: (s) => s.stalled || (orthogonal && s.support.length >= sparsity),
  }
}

/**
 * Matching pursuit (Mallat and Zhang, 1993) as a step-through algorithm: each step picks the atom most correlated
 * with the residual and moves the residual's projection onto it into the coefficients. The residual norm never
 * increases, and for a dictionary that spans $\reals^m$ it tends to zero, but an atom may be picked many times. It
 * stops when the residual is within the tolerance, or stalls when the residual is orthogonal to every atom.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns; any non-zero norms.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The tolerance; `sparsity` is ignored (it bounds orthogonal matching pursuit only), so bound the steps
 *   with `run`'s step budget.
 * @returns The algorithm; its state holds the coefficients, the residual and the atoms picked.
 *
 * @example Watch the residual shrink
 * const s = Math.SQRT1_2
 * const D = [[1, 0, s], [0, 1, s]]
 * const alg = matchingPursuitSteps(D, [2, 1.5])
 * for (const t of [1, 2, 3]) print(`after ${t} steps: residual norm =`, run(alg, undefined, t).residualNorm)
 */
export function matchingPursuitSteps(
  D: MatrixLike,
  y: VectorLike,
  options: PursuitOptions = {},
): Algorithm<void, PursuitState> {
  return pursuit('matchingPursuit', D, y, { tolerance: options.tolerance }, false)
}

/**
 * Orthogonal matching pursuit (Pati, Rezaiifar and Krishnaprasad, 1993; Tropp, 2004) as a step-through algorithm:
 * each step adds the atom most correlated with the residual to the support and refits all the support's coefficients
 * by least squares. After $s$ steps $\xvec$ has $s$ non-zeros and the residual is orthogonal to them. If $\yvec =
 * \Dmat\xvec^\star$ with fewer than $\frac{1}{2}(1 + 1/\mu)$ non-zeros ($\mu$ the `mutualCoherence`), it finds
 * $\xvec^\star$ in that many steps.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns; any non-zero norms.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The sparsity $s$ at which it stops, and the tolerance.
 * @returns The algorithm; its state holds the coefficients, the residual and the support in the order chosen.
 *
 * @example Recover a 2-sparse vector one atom at a time
 * const s = Math.SQRT1_2
 * const D = [[1, 0, 0, s], [0, 1, 0, s], [0, 0, 1, 0]]
 * const y = [0, 2, 3] // 3 times atom 2 plus 2 times atom 1
 * const one = run(orthogonalMatchingPursuitSteps(D, y, { sparsity: 2 }), undefined, 1)
 * print('first atom:', one.support, ' residual =', one.residual)
 * const two = run(orthogonalMatchingPursuitSteps(D, y, { sparsity: 2 }), undefined, 2)
 * print('support:', two.support, ' x =', two.x, ' converged:', two.converged)
 */
export function orthogonalMatchingPursuitSteps(
  D: MatrixLike,
  y: VectorLike,
  options: PursuitOptions = {},
): Algorithm<void, PursuitState> {
  return pursuit('orthogonalMatchingPursuit', D, y, options, true)
}

/**
 * The one-call result of a pursuit's final state.
 *
 * @param s The final state.
 * @returns The coefficients, the residual and its norm, the support (ascending, without repeats) and the steps.
 */
function finish(s: PursuitState): SparseApproximation {
  return {
    x: s.x,
    residual: s.residual,
    residualNorm: s.residualNorm,
    support: [...new Set(s.support)].sort((a, b) => a - b),
    steps: s.t,
  }
}

/**
 * Matching pursuit run to the tolerance or for at most `maxSteps` steps: the one-call form of
 * `matchingPursuitSteps`.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The tolerance, and the step budget.
 * @param options.maxSteps The most steps to take. Default 100.
 * @returns The coefficients, the residual, its norm, the atoms used and the steps taken.
 *
 * @example Approximate a signal over spikes and a diagonal atom
 * const s = Math.SQRT1_2
 * const D = [[1, 0, s], [0, 1, s]]
 * const res = matchingPursuit(D, [2, 1.5], { maxSteps: 20 })
 * print('x =', res.x)
 * print('residual norm =', res.residualNorm, 'after', res.steps, 'steps')
 */
export function matchingPursuit(
  D: MatrixLike,
  y: VectorLike,
  options: PursuitOptions & { maxSteps?: Size } = {},
): SparseApproximation {
  return finish(run(matchingPursuitSteps(D, y, options), undefined, options.maxSteps ?? 100))
}

/**
 * Orthogonal matching pursuit run until it has `sparsity` atoms or the residual is within the tolerance: the one-call
 * form of `orthogonalMatchingPursuitSteps`, as scikit-learn's `orthogonal_mp` for one signal.
 *
 * @param D The dictionary $\Dmat$, $m \times k$, atoms as columns.
 * @param y The signal $\yvec$ to approximate, $m$ values.
 * @param options The sparsity $s$ and the tolerance.
 * @returns The coefficients (at most $s$ non-zeros), the residual, its norm, the support and the steps taken.
 *
 * @example Recover a sparse vector exactly
 * const s = Math.SQRT1_2
 * const D = [[1, 0, 0, s], [0, 1, 0, s], [0, 0, 1, 0]]
 * const res = orthogonalMatchingPursuit(D, [0, 2, 3], { sparsity: 2 })
 * print('x =', res.x)
 * print('support =', res.support, ' residual norm =', res.residualNorm)
 */
export function orthogonalMatchingPursuit(
  D: MatrixLike,
  y: VectorLike,
  options: PursuitOptions = {},
): SparseApproximation {
  return finish(run(orthogonalMatchingPursuitSteps(D, y, options), undefined, Number.MAX_SAFE_INTEGER))
}
