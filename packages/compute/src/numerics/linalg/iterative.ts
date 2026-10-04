/**
 * Stationary iterative solvers for $\Amat\xvec = \bvec$ and the power iteration for a dominant eigenpair, as
 * traceable algorithms (Saad, 2003, §4.1; Golub and Van Loan, 2013, §11.2 and §8.2.1). Each step is one sweep or one
 * product, so a figure can follow the iterate and its residual. Conjugate gradient is `linearConjugateGradient` in
 * `aifn-compute/optim/first-order`.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { factorDense, solveFactored } from './solveDense'

type F64 = Float64Array<ArrayBuffer>

/** Options of the stationary solvers. */
export type StationaryOptions = {
  /**
   * Stop when $\lVert \bvec - \Amat\xvec \rVert$ is at most `tolerance` times $\lVert \bvec \rVert$. Default
   * 1e-10.
   */
  tolerance?: number
}

/** One state of `jacobiSteps` and `gaussSeidelSteps`. */
export interface StationaryState extends Status {
  x: Tensor
  /** The residual $\bvec - \Amat\xvec$. */
  residual: Tensor
  residualNorm: number
  converged: boolean
  /** True when the iterate stopped being finite (the iteration matrix has spectral radius above 1). */
  diverged: boolean
}

/**
 * The frame shared by the stationary methods: `sweep` maps $\xvec$ to the next iterate, and each state carries the
 * residual and its norm. $\Amat$ must be square with a non-zero diagonal. `init` takes `{ x0 }` (default zeros).
 *
 * @param name The name of the method (`'jacobi'`, `'gauss-seidel'`, `'sor'`): the name of the algorithm returned,
 *   and the prefix of error messages.
 * @param A The matrix $\Amat$, $n \times n$ with no zero on its diagonal (else `DomainError`). Copied once.
 * @param b The right-hand side $\bvec$, $n$ values (else `DomainError`). Copied once.
 * @param options The stopping tolerance on the residual, relative to $\lVert \bvec \rVert$.
 * @param sweep One iteration of the method. It is given `a`, the row-major data of $\Amat$ ($n^2$ values); `b`, the
 *   right-hand side; `x`, the current iterate; and `n`. It must return the next iterate as a new array and leave
 *   all three arrays unmodified.
 * @returns The algorithm: `init` takes `{ x0 }`, the starting iterate of $n$ values (default zeros), and each `step`
 *   is one sweep over all $n$ unknowns. It is done when converged or diverged.
 */
function stationary(
  name: string,
  A: MatrixLike,
  b: VectorLike,
  options: StationaryOptions,
  sweep: (a: F64, b: F64, x: F64, n: number) => F64,
): Algorithm<{ x0?: VectorLike } | void, StationaryState> {
  const { data: a, n } = dense.toMatrixF64(A, name)
  const bb = dense.toF64(b, name)
  if (bb.length !== n) throw new DomainError(name, `${name}: A is ${n}×${n} but b has ${bb.length} entries`)
  for (let i = 0; i < n; i++)
    if (a[i * n + i] === 0) throw new DomainError(name, `${name}: A has a zero diagonal entry at ${i}`)
  const bNorm = dense.norm(bb)
  const tol = (options.tolerance ?? 1e-10) * (bNorm > 0 ? bNorm : 1)
  const state = (t: number, x: F64): StationaryState => {
    const r = dense.sub(bb, dense.matVec(a, x, n, n))
    const residualNorm = dense.norm(r)
    const finite = x.every(Number.isFinite)
    return {
      t,
      x: dense.vec(x),
      residual: dense.vec(r),
      residualNorm,
      converged: finite && residualNorm <= tol,
      diverged: !finite,
    }
  }
  return {
    name,
    init: (start) => state(0, start && start.x0 !== undefined ? dense.toF64(start.x0, name) : new Float64Array(n)),
    step: (s) => (s.diverged ? { ...s, t: s.t + 1 } : state(s.t + 1, sweep(a, bb, dense.data(s.x), n))),
    done: (s) => s.converged || s.diverged,
  }
}

/**
 * Jacobi's method: $x_i \leftarrow (b_i - \sum_{j \ne i} a_{ij} x_j) / a_{ii}$ for every $i$ from the previous
 * iterate, i.e. $\xvec \leftarrow \Dmat^{-1}(\bvec - (\Lmat + \Umat)\xvec)$. It converges for every $\xvec_0$
 * exactly when the spectral radius of $\Dmat^{-1}(\Lmat + \Umat)$ is below 1, e.g. for a strictly diagonally dominant
 * $\Amat$. `init` takes `{ x0 }` (default zeros).
 *
 * @param A The matrix $\Amat$ of the system, $n \times n$ with no zero on its diagonal (else `DomainError`). Not
 *   modified.
 * @param b The right-hand side $\bvec$, $n$ values. Not modified.
 * @param options The stopping tolerance on the residual, relative to $\lVert \bvec \rVert$ (default 1e-10).
 * @returns The algorithm: `init` takes `{ x0 }`, the starting iterate of $n$ values (default zeros), and each `step`
 *   is one sweep over all $n$ unknowns. It is done when converged or diverged.
 *
 * @example Jacobi iteration on a diagonally dominant system
 * const A = tensor([[4, 1], [1, 3]])
 * const b = tensor([1, 2])
 * const state = run(jacobiSteps(A, b), undefined, 50)
 * print('x =', state.x)
 * print('residual norm =', state.residualNorm)
 * print('converged =', state.converged)
 */
export function jacobiSteps(
  A: MatrixLike,
  b: VectorLike,
  options: StationaryOptions = {},
): Algorithm<{ x0?: VectorLike } | void, StationaryState> {
  return stationary('jacobi', A, b, options, (a, bb, x, n) => {
    const next = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let s = bb[i]
      for (let j = 0; j < n; j++) if (j !== i) s -= a[i * n + j] * x[j]
      next[i] = s / a[i * n + i]
    }
    return next
  })
}

/**
 * Gauss–Seidel, or successive over-relaxation (SOR) with `omega` $\ne 1$: the update of $x_i$ uses the components
 * already updated in this sweep, $x_i \leftarrow (1 - \omega) x_i + \omega r_i / a_{ii}$ with
 * $r_i = b_i - \sum_{j < i} a_{ij} x_j^{\text{new}} - \sum_{j > i} a_{ij} x_j$.
 * It converges for a symmetric positive definite $\Amat$ when $0 < \omega < 2$ (Ostrowski–Reich), and for a strictly
 * diagonally dominant $\Amat$ at $\omega = 1$. `init` takes `{ x0 }` (default zeros).
 *
 * @param A The matrix $\Amat$ of the system, $n \times n$ with no zero on its diagonal (else `DomainError`). Not
 *   modified.
 * @param b The right-hand side $\bvec$, $n$ values. Not modified.
 * @param options The stopping `tolerance` on the residual, relative to $\lVert \bvec \rVert$ (default 1e-10), and
 *   `omega`, the relaxation factor $\omega$ strictly between 0 and 2 (else `DomainError`): 1 (the default) is
 *   Gauss–Seidel, above 1 over-relaxes and below 1 under-relaxes.
 * @returns The algorithm: `init` takes `{ x0 }`, the starting iterate of $n$ values (default zeros), and each `step`
 *   is one sweep over all $n$ unknowns. It is done when converged or diverged.
 *
 * @example Gauss–Seidel takes fewer sweeps than Jacobi
 * const A = tensor([[4, 1], [1, 3]])
 * const b = tensor([1, 2])
 * const after = (alg) => run(alg, undefined, 8).residualNorm
 * print('Jacobi, 8 sweeps:', after(jacobiSteps(A, b)))
 * print('Gauss–Seidel, 8 sweeps:', after(gaussSeidelSteps(A, b)))
 */
export function gaussSeidelSteps(
  A: MatrixLike,
  b: VectorLike,
  options: StationaryOptions & { omega?: number } = {},
): Algorithm<{ x0?: VectorLike } | void, StationaryState> {
  const omega = options.omega ?? 1
  if (!(omega > 0 && omega < 2)) throw new DomainError('gaussSeidelSteps', 'gaussSeidelSteps: omega must be in (0, 2)')
  return stationary(omega === 1 ? 'gauss-seidel' : 'sor', A, b, options, (a, bb, x, n) => {
    const next = Float64Array.from(x)
    for (let i = 0; i < n; i++) {
      let s = bb[i]
      for (let j = 0; j < n; j++) if (j !== i) s -= a[i * n + j] * next[j]
      next[i] = (1 - omega) * next[i] + (omega * s) / a[i * n + i]
    }
    return next
  })
}

/** The result of `solveStationary`. */
export type StationarySolution = { x: Tensor; residualNorm: number; steps: number; converged: boolean }

/**
 * Solves $\Amat\xvec = \bvec$ by Jacobi (`method: 'jacobi'`) or Gauss–Seidel/SOR (the default; `omega`), running at
 * most `maxSteps` sweeps (default 1000): `jacobiSteps` or `gaussSeidelSteps` run to the end.
 *
 * @param A The matrix $\Amat$ of the system, $n \times n$ with no zero on its diagonal (else `DomainError`). Not
 *   modified.
 * @param b The right-hand side $\bvec$, $n$ values. Not modified.
 * @param options Which method to run and when to stop: `method` (`'gauss-seidel'` by default, or `'jacobi'`),
 *   `omega` (the SOR relaxation factor in $(0, 2)$, default 1; ignored by Jacobi), `tolerance` (on the residual,
 *   relative to $\lVert \bvec \rVert$; default 1e-10) and `maxSteps` (the most sweeps, default 1000). The iteration
 *   starts from zeros.
 * @returns The last iterate `x` ($n$ values), its `residualNorm` $\lVert \bvec - \Amat\xvec \rVert$, the number of
 *   sweeps `steps`, and whether it `converged`. A run that diverged or ran out of sweeps returns `converged: false`
 *   and does not throw.
 *
 * @example Run a stationary method to convergence
 * const A = tensor([[4, 1], [1, 3]])
 * const b = tensor([1, 2])
 * const { x, steps, converged } = solveStationary(A, b, { method: 'gauss-seidel' })
 * print('x =', x)
 * print('steps =', steps)
 * print('converged =', converged)
 */
export function solveStationary(
  A: MatrixLike,
  b: VectorLike,
  options: StationaryOptions & { method?: 'jacobi' | 'gauss-seidel'; omega?: number; maxSteps?: number } = {},
): StationarySolution {
  const alg = options.method === 'jacobi' ? jacobiSteps(A, b, options) : gaussSeidelSteps(A, b, options)
  const s = run(alg, undefined, options.maxSteps ?? 1000)
  return { x: s.x, residualNorm: s.residualNorm, steps: s.t, converged: s.converged }
}

// ── Power iteration ──────────────────────────────────────────────────────────────────────────────────────────────────

/** One state of `powerIterationSteps`. */
export interface PowerIterationState extends Status {
  /** The unit iterate $\vvec_k$. */
  vector: Tensor
  /** The Rayleigh quotient $\vvec_k^\top\Amat\vvec_k$, the eigenvalue estimate. */
  value: number
  /** $\lVert \Amat\vvec_k - \lambda_k \vvec_k \rVert$, the eigen-residual. */
  residualNorm: number
  converged: boolean
  /** True when the iterate is zero or not finite. */
  diverged: boolean
}

/**
 * The power iteration $\vvec \leftarrow \Amat\vvec / \lVert \Amat\vvec \rVert$ for the eigenvalue of largest
 * magnitude of a square matrix and its eigenvector, with the Rayleigh quotient $\lambda = \vvec^\top\Amat\vvec$ as
 * the estimate; the error falls like $\lvert \lambda_2 / \lambda_1 \rvert^k$ when
 * $\lvert \lambda_1 \rvert > \lvert \lambda_2 \rvert$. With `shift` $\sigma$ it iterates
 * $\vvec \leftarrow (\Amat - \sigma\Imat)^{-1}\vvec$, normalised, instead (inverse iteration, one LU factor
 * reused), which finds the eigenvalue nearest $\sigma$. Stops when $\lVert \Amat\vvec - \lambda\vvec \rVert$ is at
 * most `tolerance` (default 1e-10) times $\max(1, \lvert \lambda \rvert)$. `init` takes `{ v0 }` (default all ones).
 *
 * @param A The square matrix $\Amat$ ($n \times n$, else `DomainError`). Not modified.
 * @param options Whether to run inverse iteration, and when to stop.
 * @param options.shift The shift $\sigma$. When given, inverse iteration finds the eigenvalue nearest $\sigma$; a
 *   shift that makes $\Amat - \sigma\Imat$ singular is nudged by a relative $10^{-10}$. When omitted, the plain
 *   power iteration finds the eigenvalue of largest magnitude.
 * @param options.tolerance The bound on the eigen-residual, relative to $\max(1, \lvert \lambda \rvert)$ (default
 *   1e-10).
 * @returns The algorithm: `init` takes `{ v0 }`, the starting vector of $n$ values (default all ones; it is
 *   normalised), and each `step` is one product (or one solve, with a shift) followed by normalisation. It is done
 *   when converged or diverged.
 *
 * @example The dominant eigenpair by power iteration
 * const A = tensor([[2, 1], [1, 2]])
 * const state = run(powerIterationSteps(A), undefined, 100)
 * print('eigenvalue =', state.value)
 * print('eigenvector =', state.vector)
 */
export function powerIterationSteps(
  A: MatrixLike,
  options: { shift?: number; tolerance?: number } = {},
): Algorithm<{ v0?: VectorLike } | void, PowerIterationState> {
  const where = 'powerIterationSteps'
  const { data: a, m, n } = dense.toMatrixF64(A, where)
  if (m !== n) throw new DomainError(where, `${where}: A must be square, got ${m}×${n}`)
  const tolerance = options.tolerance ?? 1e-10
  const { shift } = options
  let apply = (v: F64): F64 => dense.matVec(a, v, n, n)
  if (shift !== undefined) {
    // A shift equal to an eigenvalue makes A − σI singular: nudge it by a relative 1e-10.
    const shifted = (s: number) =>
      factorDense(
        a.map((x, k) => (k % (n + 1) === 0 ? x - s : x)),
        n,
      )
    let f = shifted(shift)
    if (f.singular) f = shifted(shift + 1e-10 * Math.max(1, Math.abs(shift)))
    apply = (v) => solveFactored(f, v) ?? new Float64Array(n).fill(NaN)
  }
  const state = (t: number, w: F64): PowerIterationState => {
    const size = dense.norm(w)
    const v = size > 0 && Number.isFinite(size) ? dense.scale(1 / size, w) : new Float64Array(n).fill(NaN)
    const Av = dense.matVec(a, v, n, n)
    const value = dense.dot(v, Av)
    const residualNorm = dense.norm(dense.axpy(-value, v, Av))
    const finite = v.every(Number.isFinite)
    return {
      t,
      vector: dense.vec(v),
      value,
      residualNorm,
      converged: finite && residualNorm <= tolerance * Math.max(1, Math.abs(value)),
      diverged: !finite,
    }
  }
  return {
    name: shift === undefined ? 'power-iteration' : 'inverse-iteration',
    init: (start) =>
      state(0, start && start.v0 !== undefined ? dense.toF64(start.v0, where) : new Float64Array(n).fill(1)),
    step: (s) => (s.diverged ? { ...s, t: s.t + 1 } : state(s.t + 1, apply(dense.data(s.vector)))),
    done: (s) => s.converged || s.diverged,
  }
}
