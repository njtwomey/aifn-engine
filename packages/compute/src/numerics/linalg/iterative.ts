/**
 * Stationary iterative solvers for Ax = b and the power iteration for a dominant eigenpair, as traceable algorithms
 * (Saad, 2003, §4.1; Golub and Van Loan, 2013, §11.2 and §8.2.1). Each step is one sweep or one product, so a figure
 * can follow the iterate and its residual. Conjugate gradient is `linearConjugateGradient` in `aifn-compute/optim/first-order`.
 */

import type { MatrixLike, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { dense, type Tensor } from 'aifn-compute/foundation/tensor'
import { run, type Algorithm } from 'aifn-compute/foundation/trace'
import { factorDense, solveFactored } from './solveDense'

type F64 = Float64Array<ArrayBuffer>

/** Options of the stationary solvers. */
export type StationaryOptions = {
  /** Stop when ‖b − Ax‖ ≤ tolerance · ‖b‖. Default 1e-10. */
  tolerance?: number
}

/** One state of `jacobiSteps` and `gaussSeidelSteps`. */
export interface StationaryState extends Status {
  x: Tensor
  /** The residual b − Ax. */
  residual: Tensor
  residualNorm: number
  converged: boolean
  /** True when the iterate stopped being finite (the iteration matrix has spectral radius above 1). */
  diverged: boolean
}

/**
 * The frame shared by the stationary methods: `sweep` maps x to the next iterate, and each state carries the residual
 * and its norm. A must be square with a non-zero diagonal. `init` takes `{ x0 }` (default zeros).
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
 * Jacobi's method: x_i ← (b_i − Σ_{j≠i} a_ij x_j) / a_ii for every i from the previous iterate, i.e.
 * x ← D⁻¹(b − (L + U)x). It converges for every x₀ exactly when the spectral radius of D⁻¹(L + U) is below 1, e.g.
 * for a strictly diagonally dominant A. `init` takes `{ x0 }` (default zeros).
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
 * Gauss–Seidel, or successive over-relaxation (SOR) with `omega` ≠ 1: the update of x_i uses the components already
 * updated in this sweep, x_i ← (1 − ω)x_i + ω(b_i − Σ_{j<i} a_ij x_j^{new} − Σ_{j>i} a_ij x_j) / a_ii. It converges
 * for a symmetric positive definite A when 0 < ω < 2 (Ostrowski–Reich), and for a strictly diagonally dominant A at
 * ω = 1.
 * `init` takes `{ x0 }` (default zeros).
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
 * Solves Ax = b by Jacobi (`method: 'jacobi'`) or Gauss–Seidel/SOR (the default; `omega`), running at most `maxSteps`
 * sweeps (default 1000): `jacobiSteps` or `gaussSeidelSteps` run to the end.
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
  /** The unit iterate v_k. */
  vector: Tensor
  /** The Rayleigh quotient v_kᵀAv_k, the eigenvalue estimate. */
  value: number
  /** ‖Av_k − λ_k v_k‖, the eigen-residual. */
  residualNorm: number
  converged: boolean
  /** True when the iterate is zero or not finite. */
  diverged: boolean
}

/**
 * The power iteration v ← Av / ‖Av‖ for the eigenvalue of largest magnitude of a square matrix and its eigenvector,
 * with the Rayleigh quotient λ = vᵀAv as the estimate; the error falls like |λ₂/λ₁|^k when |λ₁| > |λ₂|. With `shift`
 * σ it iterates v ← (A − σI)⁻¹v / ‖·‖ instead (inverse iteration, one LU factor reused), which finds the eigenvalue
 * nearest σ. Stops when ‖Av − λv‖ ≤ tolerance (default 1e-10) · max(1, |λ|). `init` takes `{ v0 }` (default all
 * ones).
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
