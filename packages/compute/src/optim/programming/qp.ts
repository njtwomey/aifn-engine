/**
 * Convex quadratic programming: minimise $\frac{1}{2}\xvec^\top\Qmat\xvec + \cvec^\top\xvec$ subject to
 * $\Amat\xvec \le \bvec$ and $\Emat\xvec = \evec$, by the primal active-set method (Nocedal and Wright, 2006,
 * "Numerical Optimization", Algorithm 16.3) and a primal–dual interior-point method with Mehrotra's
 * predictor–corrector (ibid., §16.6); and box-constrained QP by projected gradient with subspace Newton steps (Moré
 * and Toraldo, 1991, "On the solution of large quadratic programming problems with bound constraints", SIAM J.
 * Optimization 1(1)). Multipliers follow the Lagrangian $L = f(\xvec) + \lambdavec^\top(\Amat\xvec - \bvec)$
 * $+ \nuvec^\top(\Emat\xvec - \evec)$, with $f$ the objective and $\lambdavec \ge \zeros$.
 */

import { dense, fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { eigh } from 'aifn-compute/numerics/linalg'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { run } from 'aifn-compute/foundation/trace'
import type { MatrixLike, Scalar, Size, Status, VectorLike } from 'aifn-compute/foundation/contracts'
import type { RunOptions } from '../options'
import {
  checkFinite,
  intTensor,
  matTVec,
  matVec,
  matrix,
  readMatrix,
  readVector,
  solve,
  vector,
  type Mat,
} from './input'
import { simplexSolve } from './simplex'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A convex quadratic program: minimise $\frac{1}{2}\xvec^\top\Qmat\xvec + \cvec^\top\xvec$ subject to
 * $\Amat\xvec \le \bvec$ and $\Emat\xvec = \evec$ (variables otherwise free). Absent constraints are empty.
 */
export interface QuadraticProgram {
  /**
   * Symmetric Hessian $\Qmat$, $n \times n$, positive semi-definite on the null space of $\Emat$ (the problem is then
   * convex). The interior point stops `nonconvex` otherwise; the active set stops `nonconvex` when a working set
   * exposes negative curvature.
   */
  Q: MatrixLike
  /** Linear term $\cvec$, length $n$. */
  c: VectorLike
  /** Inequality constraint matrix $\Amat$ of $\Amat\xvec \le \bvec$, $m \times n$. */
  A?: MatrixLike
  /** Inequality right-hand side $\bvec$, length $m$. */
  b?: VectorLike
  /** Equality constraint matrix $\Emat$ of $\Emat\xvec = \evec$, $p \times n$. */
  E?: MatrixLike
  /** Equality right-hand side $\evec$, length $p$. */
  e?: VectorLike
}

/** A quadratic program read into dense arrays. */
export interface ParsedQP {
  /** The number of variables. */
  n: number
  /** The Hessian, $n \times n$. */
  Q: Mat
  /** The linear term, length $n$. */
  c: Float64Array
  /** The inequality matrix, $m \times n$ ($0 \times n$ when there is none). */
  A: Mat
  /** The inequality right-hand side, length $m$. */
  b: Float64Array
  /** The equality matrix, $p \times n$ ($0 \times n$ when there is none). */
  E: Mat
  /** The equality right-hand side, length $p$. */
  e: Float64Array
}

/**
 * Read and validate a quadratic program. Throws `ShapeError` when a size disagrees with the length of `c` (or `Q` is
 * not square), and `DomainError` for a non-finite entry. Symmetry and convexity are not checked here.
 *
 * @param problem The quadratic program, with its matrices and vectors as tensors or plain arrays.
 * @returns The program as dense arrays.
 */
export function parseQP(problem: QuadraticProgram): ParsedQP {
  const c = readVector(problem.c, 'quadratic program: c')
  const n = c.length
  const Q = readMatrix(problem.Q, 'quadratic program: Q', n)
  if (Q.m !== n) throw new ShapeError('quadratic program', `quadratic program: Q must be ${n}×${n}`)
  const A = readMatrix(problem.A, 'quadratic program: A', n)
  const b = readVector(problem.b, 'quadratic program: b', A.m)
  const E = readMatrix(problem.E, 'quadratic program: E', n)
  const e = readVector(problem.e, 'quadratic program: e', E.m)
  for (const [v, name] of [
    [Q.a, 'Q'],
    [c, 'c'],
    [A.a, 'A'],
    [b, 'b'],
    [E.a, 'E'],
    [e, 'e'],
  ] as const)
    checkFinite(v, `quadratic program: ${name}`)
  return { n, Q, c, A, b, E, e }
}

/**
 * The objective $\frac{1}{2}\xvec^\top\Qmat\xvec + \cvec^\top\xvec$ of a quadratic program at a point.
 *
 * @param qp The parsed quadratic program.
 * @param x The point, $n$ values.
 * @returns The objective.
 */
const objectiveOf = (qp: ParsedQP, x: ArrayLike<number>) => 0.5 * dense.dot(x, matVec(qp.Q, x)) + dense.dot(qp.c, x)

/** The Karush–Kuhn–Tucker residuals of a point and multipliers of a quadratic program. */
export interface KKTReport {
  /** The point $\xvec$, length $n$. */
  x: Tensor
  /** $\frac{1}{2}\xvec^\top\Qmat\xvec + \cvec^\top\xvec$. */
  objective: Scalar
  /** Multipliers $\lambdavec$ of $\Amat\xvec \le \bvec$ (length $m$; $\ge 0$ at an optimum). */
  lambda: Tensor
  /** Multipliers $\nuvec$ of $\Emat\xvec = \evec$ (length $p$). */
  nu: Tensor
  /** Slacks $\bvec - \Amat\xvec$ (length $m$). */
  slack: Tensor
  /**
   * $\lVert \Qmat\xvec + \cvec + \Amat^\top\lambdavec + \Emat^\top\nuvec \rVert_\infty$: stationarity of the
   * Lagrangian.
   */
  stationarity: Scalar
  /** Largest violation of $\Amat\xvec \le \bvec$ or $\Emat\xvec = \evec$. */
  primalInfeasibility: Scalar
  /** Largest negative multiplier, as a positive number (0 when $\lambdavec \ge \zeros$). */
  dualInfeasibility: Scalar
  /** $\max_i \lvert \lambda_i s_i \rvert$ ($s_i$ the slack): complementary slackness. */
  complementarity: Scalar
  /** 1 for each inequality whose slack is at most `tolerance` in absolute value, else 0; int32. */
  active: Tensor
}

/**
 * The KKT report of a point `x` with multipliers `lambda` (inequalities) and `nu` (equalities): stationarity, primal
 * and dual feasibility and complementary slackness (Nocedal and Wright, 2006, Theorem 12.1). All four are zero at an
 * optimum of a convex QP. Throws `ShapeError` when a length disagrees with the problem.
 *
 * @param problem The quadratic program, or its parsed form (as `parseQP` returns it, which is not parsed again).
 * @param x The point $\xvec$, $n$ values.
 * @param lambda The multipliers $\lambdavec$ of the inequalities, $m$ values.
 * @param nu The multipliers $\nuvec$ of the equalities, $p$ values.
 * @param tolerance The largest absolute slack at which an inequality is reported `active`. It decides nothing else.
 * @returns The report.
 *
 * @example Certify the minimiser of a small QP
 * // Minimise (x - 1)^2 + (y - 2.5)^2 over a pentagon (Nocedal and Wright, Example 16.4). At (1.4, 1.7) only the first
 * // constraint, -x + 2y <= 2, binds; its multiplier is 0.8.
 * const A = [[-1, 2], [1, 2], [1, -2], [-1, 0], [0, -1]]
 * const problem = { Q: [[2, 0], [0, 2]], c: [-2, -5], A, b: [2, 6, 2, 0, 0] }
 * const r = kktReport(problem, [1.4, 1.7], [0.8, 0, 0, 0, 0], [])
 * print('stationarity =', r.stationarity, ' complementarity =', r.complementarity)
 * print('primal infeasibility =', r.primalInfeasibility, ' dual infeasibility =', r.dualInfeasibility)
 * print('active =', r.active)
 *
 * @example A wrong multiplier shows in the stationarity residual
 * const A = [[-1, 2], [1, 2], [1, -2], [-1, 0], [0, -1]]
 * const problem = { Q: [[2, 0], [0, 2]], c: [-2, -5], A, b: [2, 6, 2, 0, 0] }
 * print('stationarity =', kktReport(problem, [1.4, 1.7], [0.5, 0, 0, 0, 0], []).stationarity)
 */
export function kktReport(
  problem: QuadraticProgram | ParsedQP,
  x: VectorLike,
  lambda: VectorLike,
  nu: VectorLike,
  tolerance = 1e-7,
): KKTReport {
  const qp = 'Q' in problem && 'n' in problem ? (problem as ParsedQP) : parseQP(problem as QuadraticProgram)
  const xs = readVector(x, 'kktReport: x', qp.n)
  const l = readVector(lambda, 'kktReport: lambda', qp.A.m)
  const v = readVector(nu, 'kktReport: nu', qp.E.m)
  const g = matVec(qp.Q, xs)
  const Al = matTVec(qp.A, l)
  const Ev = matTVec(qp.E, v)
  for (let j = 0; j < qp.n; j++) g[j] += qp.c[j] + Al[j] + Ev[j]
  const Ax = matVec(qp.A, xs)
  const slack = qp.b.map((bi, i) => bi - Ax[i])
  const Ex = matVec(qp.E, xs)
  let primal = 0
  let dual = 0
  let comp = 0
  for (let i = 0; i < qp.A.m; i++) {
    primal = Math.max(primal, -slack[i])
    dual = Math.max(dual, -l[i])
    comp = Math.max(comp, Math.abs(l[i] * slack[i]))
  }
  for (let i = 0; i < qp.E.m; i++) primal = Math.max(primal, Math.abs(Ex[i] - qp.e[i]))
  return {
    x: vector(xs),
    objective: objectiveOf(qp, xs),
    lambda: vector(l),
    nu: vector(v),
    slack: vector(slack),
    stationarity: dense.maxAbs(g),
    primalInfeasibility: primal,
    dualInfeasibility: dual,
    complementarity: comp,
    active: intTensor(Array.from(slack, (s) => (Math.abs(s) <= tolerance ? 1 : 0))),
  }
}

/**
 * Outcome of a QP solver: `running`, `optimal`, `infeasible` (active set: no feasible start exists), `singular` (a
 * linear system could not be solved), `nonconvex` or `diverged` (interior point: the iterates grew past $10^{12}$ or
 * became non-finite). `nonconvex`: $\Qmat$ has negative curvature where the method needs it to have none, so a
 * stationary point need not be a minimiser. The active set checks the null space of its working set at each step; the
 * interior point checks the null space of the equalities $\Emat$ before its first step.
 */
export type QuadraticProgramStatus = 'running' | 'optimal' | 'infeasible' | 'singular' | 'nonconvex' | 'diverged'

// ---------------------------------------------------------------------------------------------------------------------
// Primal active-set method.

/** Options for `activeSet`. */
export interface ActiveSetOptions {
  /** Tolerance for zero steps, multipliers and active constraints (default 1e-10). */
  tolerance?: Scalar
}

/**
 * The start of `activeSet`: `x0`, a feasible point (not checked); default a vertex of the feasible set found by the
 * simplex method.
 */
export type ActiveSetStart = { x0?: VectorLike }

/**
 * What an active-set step did: `start` (the initial state), `step` (moved to the working-set minimiser), `add` (stopped
 * at a blocking constraint and added it), `drop` (removed the constraint with the most negative multiplier) or
 * `optimal`.
 */
export type ActiveSetEvent = 'start' | 'step' | 'add' | 'drop' | 'optimal'

/**
 * One iterate of the primal active-set method: `converged` at an optimum, `terminated` when the problem is
 * infeasible, an equality-QP solve is singular, or a working set exposes negative curvature.
 */
export interface ActiveSetState extends Status {
  /** The current feasible point, length $n$ (NaN when the problem is infeasible). */
  x: Tensor
  /** The working set: indices of inequalities held as equalities, int32, ascending. */
  working: Tensor
  /**
   * The step direction $\pvec$ computed by the last step (zero when $\xvec$ minimises over the working set).
   */
  p: Tensor
  /** The step length taken along $\pvec$ (1 unless a constraint blocks the step). */
  alpha: Scalar
  /** Multipliers of the inequalities (length $m$; zero outside the working set) from the last equality-QP solve. */
  lambda: Tensor
  /** Multipliers of the equalities (length $p$). */
  nu: Tensor
  /** The constraint added (blocking) or dropped (most negative multiplier) by the last step, or $-1$. */
  changed: number
  /** What the step that made this state did. */
  event: ActiveSetEvent
  /** The objective at `x`. */
  objective: Scalar
  /** Whether the run goes on, or how it ended. */
  status: QuadraticProgramStatus
  /** The problem as parsed. */
  problem: ParsedQP
  /** True when `status` is `optimal`. */
  converged: boolean
  /** True when `status` is `infeasible`, `singular` or `nonconvex`. */
  terminated: boolean
}

/**
 * The Status flags of a QP solver state from its outcome.
 *
 * @param status The outcome.
 * @returns `converged` (when `optimal`) and `terminated` (when `infeasible`, `singular` or `nonconvex`).
 */
const flags = (status: QuadraticProgramStatus) => ({
  converged: status === 'optimal',
  terminated: status === 'infeasible' || status === 'singular' || status === 'nonconvex',
})

/**
 * The rows of the equalities and of the working set, $[\Emat; \Amat_W]$, as a dense matrix.
 *
 * @param qp The parsed quadratic program.
 * @param working The indices of the inequalities in the working set, in the order their rows are stacked.
 * @returns The $(p + \lvert W \rvert) \times n$ matrix.
 */
function workingRows(qp: ParsedQP, working: ArrayLike<number>): Mat {
  const k = qp.E.m + working.length
  const a = new Float64Array(k * qp.n)
  a.set(qp.E.a)
  for (let r = 0; r < working.length; r++)
    a.set(qp.A.a.subarray(working[r] * qp.n, (working[r] + 1) * qp.n), (qp.E.m + r) * qp.n)
  return { m: k, n: qp.n, a }
}

/**
 * Solve the equality-constrained QP: minimise $\frac{1}{2}\pvec^\top\Qmat\pvec + \gvec^\top\pvec$ subject to
 * $\Mmat\pvec = \zeros$, through its KKT system $\Qmat\pvec + \Mmat^\top\muvec = -\gvec$, $\Mmat\pvec = \zeros$.
 *
 * @param qp The parsed quadratic program, whose $\Qmat$ is used.
 * @param M The constraint rows $\Mmat$ (equalities, then the working set), $k \times n$.
 * @param g The gradient $\gvec = \Qmat\xvec + \cvec$ at the current point, $n$ values.
 * @returns `p`, the step to the minimiser; `mu`, the $k$ multipliers of $\Mmat$'s rows; and `singular`, true when the
 *   KKT matrix is singular (both are then zero).
 */
function equalityQP(qp: ParsedQP, M: Mat, g: Float64Array) {
  const n = qp.n
  const size = n + M.m
  const K = new Float64Array(size * size)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * size + j] = qp.Q.a[i * n + j]
  for (let r = 0; r < M.m; r++)
    for (let j = 0; j < n; j++) {
      K[(n + r) * size + j] = M.a[r * n + j]
      K[j * size + n + r] = M.a[r * n + j]
    }
  const rhs = new Float64Array(size)
  for (let j = 0; j < n; j++) rhs[j] = -g[j]
  const { x: sol, singular } = solve(K, size, rhs)
  return { p: sol.subarray(0, n), mu: sol.subarray(n), singular }
}

/**
 * Whether $\Qmat$ has negative curvature on the null space of $\Mmat$: the reduced Hessian $\Zmat^\top\Qmat\Zmat$
 * ($\Zmat$ an orthonormal basis of that null space) has an eigenvalue below $-$`tol` times
 * $1 + \max_{ij} \lvert Q_{ij} \rvert$. $\Zmat$ is rank-revealing, the eigenvectors of $\Mmat^\top\Mmat$ with
 * eigenvalues at most $10^{-12}$ of the largest, so dependent rows of $\Mmat$ (repeated or combined equalities) do not
 * shrink the null space. Zero curvature is left to the KKT solve, which reports it as singular.
 *
 * @param qp The parsed quadratic program, whose $\Qmat$ is used.
 * @param M The constraint rows $\Mmat$, $k \times n$ (with $k = 0$ the whole space is checked).
 * @param tol The tolerance on the smallest eigenvalue, relative to $\Qmat$'s scale.
 * @returns True when the reduced Hessian has a negative eigenvalue; false also when the null space is trivial.
 */
function negativeCurvature(qp: ParsedQP, M: Mat, tol: number): boolean {
  const n = qp.n
  const k = M.m
  let Z: Float64Array
  let r: number
  if (k === 0) {
    Z = Float64Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0))
    r = n
  } else {
    // MᵀM (n × n); its eigenvectors with negligible eigenvalues span null(M) whatever M's rank.
    const G = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let v = 0
        for (let row = 0; row < k; row++) v += M.a[row * n + i] * M.a[row * n + j]
        G[i * n + j] = v
      }
    const e = eigh(fromData(G, [n, n]))
    const lambda = toFlat(e.values)
    const V = toFlat(e.vectors)
    const cutoff = 1e-12 * Math.max(lambda[0], 0)
    const keep: number[] = []
    for (let c = 0; c < n; c++) if (lambda[c] <= cutoff) keep.push(c)
    r = keep.length
    if (r === 0) return false
    Z = new Float64Array(n * r)
    for (let i = 0; i < n; i++) keep.forEach((c, q) => (Z[i * r + q] = V[i * n + c]))
  }
  const QZ = new Float64Array(n * r)
  for (let i = 0; i < n; i++)
    for (let c = 0; c < r; c++) {
      let v = 0
      for (let j = 0; j < n; j++) v += qp.Q.a[i * n + j] * Z[j * r + c]
      QZ[i * r + c] = v
    }
  const H = new Float64Array(r * r)
  for (let a = 0; a < r; a++)
    for (let b = 0; b < r; b++) {
      let v = 0
      for (let i = 0; i < n; i++) v += Z[i * r + a] * QZ[i * r + b]
      H[a * r + b] = v
    }
  for (let a = 0; a < r; a++)
    for (let b = 0; b < a; b++) H[a * r + b] = H[b * r + a] = 0.5 * (H[a * r + b] + H[b * r + a])
  const values = toFlat(eigh(fromData(H, [r, r])).values)
  return Math.min(...values) < -tol * (1 + dense.maxAbs(qp.Q.a))
}

/**
 * Greedily choose inequalities active at $\xvec$ whose rows are independent of $\Emat$'s and each other's (by
 * Gram–Schmidt with re-orthogonalisation).
 *
 * @param qp The parsed quadratic program.
 * @param x The starting point, $n$ values.
 * @param tol An inequality is active when its slack is at most `tol` times $1 + \lvert b_i \rvert$.
 * @returns The chosen inequalities' indices, ascending.
 */
function initialWorkingSet(qp: ParsedQP, x: Float64Array, tol: number): number[] {
  const Ax = matVec(qp.A, x)
  const chosen: number[] = []
  const basis: Float64Array[] = []
  const add = (row: Float64Array) => {
    const v = row.slice()
    for (let pass = 0; pass < 2; pass++)
      for (const q of basis) {
        const d = dense.dot(v, q)
        for (let k = 0; k < v.length; k++) v[k] -= d * q[k]
      }
    const r = dense.norm(v)
    if (r <= 1e-9 * Math.max(dense.norm(row), 1e-300)) return false
    basis.push(v.map((a) => a / r))
    return true
  }
  for (let i = 0; i < qp.E.m; i++) add(qp.E.a.slice(i * qp.n, (i + 1) * qp.n))
  for (let i = 0; i < qp.A.m; i++)
    if (Math.abs(qp.b[i] - Ax[i]) <= tol * (1 + Math.abs(qp.b[i])) && add(qp.A.a.slice(i * qp.n, (i + 1) * qp.n)))
      chosen.push(i)
  return chosen
}

/**
 * A feasible point of $\Amat\xvec \le \bvec$, $\Emat\xvec = \evec$ (a vertex, from the simplex method with a zero
 * objective and free variables), or null when there is none.
 *
 * @param qp The parsed quadratic program.
 * @returns The point, $n$ values, or null.
 */
function feasiblePoint(qp: ParsedQP): Float64Array | null {
  const rows = (M: Mat) => Array.from({ length: M.m }, (_, i) => Array.from(M.a.subarray(i * M.n, (i + 1) * M.n)))
  const r = simplexSolve({
    c: new Array<number>(qp.n).fill(0),
    A_ub: rows(qp.A),
    b_ub: Array.from(qp.b),
    A_eq: rows(qp.E),
    b_eq: Array.from(qp.e),
    bounds: [null, null],
  })
  return r.status === 'optimal' ? Float64Array.from(r.x.data) : null
}

/**
 * The primal active-set method for the convex QP `problem` (Nocedal and Wright, 2006, Algorithm 16.3) as a traceable
 * algorithm; `init` takes `{ x0? }`, a feasible start (default: a vertex found by the simplex method; the initial
 * state is `infeasible` when there is none). The initial working set holds the inequalities active at the start whose
 * rows are independent. Each step solves the equality-constrained QP on the working set; it then either moves to its
 * minimiser (`step`), stops at a blocking constraint and adds it (`add`), drops the inequality with the most negative
 * multiplier (`drop`), or certifies optimality (`optimal`, all multipliers $\ge 0$). It needs $\Qmat$ positive definite
 * on the null space of each working set (e.g. $\Qmat$ positive definite). Each step checks that reduced Hessian and
 * stops `nonconvex` when it has a negative eigenvalue (the subproblem's stationary point is then a saddle, not a
 * minimiser), or `singular` when the KKT system cannot be solved (zero curvature).
 *
 * @param problem The quadratic program.
 * @param options The tolerance on zero steps, multipliers and active constraints.
 * @returns The algorithm. Once the run has ended, a step returns the state unchanged.
 *
 * @example Nocedal and Wright's Example 16.4, from the vertex (2, 0)
 * // Minimise (x - 1)^2 + (y - 2.5)^2 subject to five linear inequalities. The minimiser is (1.4, 1.7).
 * const A = [[-1, 2], [1, 2], [1, -2], [-1, 0], [0, -1]]
 * const problem = { Q: [[2, 0], [0, 2]], c: [-2, -5], A, b: [2, 6, 2, 0, 0] }
 * const tr = trace(activeSet(problem), { x0: [2, 0] }, 20)
 * print('events =', tr.steps.map((s) => s.event))
 * print('x =', tr.steps.map((s) => Array.from(s.x.data)))
 * print('working sets =', tr.steps.map((s) => Array.from(s.working.data)))
 * print('multipliers =', tr.steps.at(-1).lambda)
 */
export function activeSet(
  problem: QuadraticProgram,
  options: ActiveSetOptions = {},
): Algorithm<ActiveSetStart, ActiveSetState> {
  const qp = parseQP(problem)
  const tol = options.tolerance ?? 1e-10
  return {
    name: 'active-set-qp',
    init: (start) => {
      const x0 = start?.x0 !== undefined ? readVector(start.x0, 'activeSet: x0', qp.n) : feasiblePoint(qp)
      const base = {
        t: 0,
        problem: qp,
        p: vector(new Float64Array(qp.n)),
        alpha: 0,
        lambda: vector(new Float64Array(qp.A.m)),
        nu: vector(new Float64Array(qp.E.m)),
        changed: -1,
        event: 'start' as const,
      }
      if (!x0) {
        const x = new Float64Array(qp.n).fill(NaN)
        return {
          ...base,
          x: vector(x),
          working: intTensor([]),
          objective: NaN,
          status: 'infeasible',
          ...flags('infeasible'),
        }
      }
      return {
        ...base,
        x: vector(x0),
        working: intTensor(initialWorkingSet(qp, x0, Math.max(tol, 1e-9))),
        objective: objectiveOf(qp, x0),
        status: 'running',
        ...flags('running'),
      }
    },
    step: (s) => {
      if (s.status !== 'running') return s
      const x = Float64Array.from(s.x.data)
      const working = Array.from(s.working.data)
      const g = matVec(qp.Q, x)
      for (let j = 0; j < qp.n; j++) g[j] += qp.c[j]
      const rows = workingRows(qp, working)
      const next = { ...s, t: s.t + 1 }
      if (negativeCurvature(qp, rows, tol)) return { ...next, status: 'nonconvex' as const, ...flags('nonconvex') }
      const { p, mu, singular } = equalityQP(qp, rows, g)
      if (singular) return { ...next, status: 'singular' as const, ...flags('singular') }
      const lambda = new Float64Array(qp.A.m)
      working.forEach((i, r) => (lambda[i] = mu[qp.E.m + r]))
      const nu = Float64Array.from(mu.subarray(0, qp.E.m))
      const scale = 1 + dense.maxAbs(x)
      if (dense.maxAbs(p) <= tol * scale * 1e3) {
        // x minimises over the working set: optimal if every working multiplier is non-negative, else drop the most
        // negative one (its constraint is pushing the wrong way).
        let worst = -1
        for (const i of working) if (lambda[i] < -tol * 1e3 && (worst < 0 || lambda[i] < lambda[worst])) worst = i
        const common = { p: vector(p), alpha: 0, lambda: vector(lambda), nu: vector(nu) }
        if (worst < 0)
          return {
            ...next,
            ...common,
            changed: -1,
            event: 'optimal' as const,
            status: 'optimal' as const,
            ...flags('optimal'),
          }
        return {
          ...next,
          ...common,
          working: intTensor(working.filter((i) => i !== worst)),
          changed: worst,
          event: 'drop' as const,
        }
      }
      // Step towards the working-set minimiser, stopping at the first inequality outside the working set it would cross.
      let alpha = 1
      let blocking = -1
      const inW = new Set(working)
      const Ap = matVec(qp.A, p)
      const Ax = matVec(qp.A, x)
      for (let i = 0; i < qp.A.m; i++) {
        if (inW.has(i) || Ap[i] <= tol) continue
        const t = Math.max(0, qp.b[i] - Ax[i]) / Ap[i]
        if (t < alpha) {
          alpha = t
          blocking = i
        }
      }
      for (let j = 0; j < qp.n; j++) x[j] += alpha * p[j]
      const common = {
        x: vector(x),
        p: vector(p),
        alpha,
        lambda: vector(lambda),
        nu: vector(nu),
        objective: objectiveOf(qp, x),
      }
      if (blocking < 0) return { ...next, ...common, changed: -1, event: 'step' as const }
      return {
        ...next,
        ...common,
        working: intTensor([...working, blocking].sort((a, b) => a - b)),
        changed: blocking,
        event: 'add' as const,
      }
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Interior-point method.

/** Options for `quadraticInteriorPoint`. */
export interface QuadraticInteriorPointOptions {
  /**
   * Stop when the residuals and the complementarity measure $\mu$ are below this times $1 + d$, with $d$ the largest
   * absolute entry of $\cvec$, $\bvec$, $\evec$ and $\Qmat$ (default 1e-9). It is also the tolerance of the convexity
   * check.
   */
  tolerance?: Scalar
  /** Fraction of the step to the boundary taken (default 0.99). */
  stepFraction?: Scalar
}

/** One iterate of the QP interior-point method. */
export interface QuadraticInteriorPointState extends Status {
  /** The primal iterate $\xvec$, length $n$ (not necessarily feasible before convergence). */
  x: Tensor
  /**
   * Slacks $\svec$ of the inequalities, length $m$ (strictly positive), with $\Amat\xvec + \svec = \bvec$ at
   * convergence.
   */
  s: Tensor
  /** Multipliers $\lambdavec$ of the inequalities, length $m$ (strictly positive). */
  lambda: Tensor
  /** Multipliers $\nuvec$ of the equalities, length $p$. */
  nu: Tensor
  /** Complementarity measure $\mu = \svec^\top\lambdavec / m$ (0 when there are no inequalities). */
  mu: Scalar
  /** The centring parameter $\sigma = \min(1, (\mu_{\text{aff}}/\mu)^3)$ of the last step; NaN at the start. */
  sigma: Scalar
  /** The step length of the last step, the same for all variables (0 at the start). */
  alpha: Scalar
  /** $\lVert \Qmat\xvec + \cvec + \Amat^\top\lambdavec + \Emat^\top\nuvec \rVert_\infty$. */
  stationarity: Scalar
  /**
   * $\max(\lVert \Amat\xvec + \svec - \bvec \rVert_\infty, \lVert \Emat\xvec - \evec \rVert_\infty)$.
   */
  primalResidual: Scalar
  /** The objective at `x`. */
  objective: Scalar
  /** True when `status` is `optimal`. */
  converged: boolean
  /**
   * True on failure: the Newton system was singular (`status` is then `singular`) or the iterates grew past $10^{12}$
   * or became non-finite (`diverged`).
   */
  diverged: boolean
  /** True when the run stopped without converging or diverging: $\Qmat$ is not convex on the null space of $\Emat$. */
  terminated: boolean
  /** Whether the run goes on, or how it ended. */
  status: QuadraticProgramStatus
  /** The problem as parsed. */
  problem: ParsedQP
}

/**
 * The residuals of the QP's KKT conditions (without complementarity) at an iterate.
 *
 * @param qp The parsed quadratic program.
 * @param x The primal iterate $\xvec$, $n$ values.
 * @param s The slacks $\svec$, $m$ values.
 * @param l The inequality multipliers $\lambdavec$, $m$ values.
 * @param v The equality multipliers $\nuvec$, $p$ values.
 * @returns `rd` $= \Qmat\xvec + \cvec + \Amat^\top\lambdavec + \Emat^\top\nuvec$, `rp` $= \Amat\xvec + \svec - \bvec$
 *   and `re` $= \Emat\xvec - \evec$.
 */
function qpResiduals(qp: ParsedQP, x: Float64Array, s: Float64Array, l: Float64Array, v: Float64Array) {
  const rd = matVec(qp.Q, x)
  const Al = matTVec(qp.A, l)
  const Ev = matTVec(qp.E, v)
  for (let j = 0; j < qp.n; j++) rd[j] += qp.c[j] + Al[j] + Ev[j]
  const rp = matVec(qp.A, x)
  for (let i = 0; i < qp.A.m; i++) rp[i] += s[i] - qp.b[i]
  const re = matVec(qp.E, x)
  for (let i = 0; i < qp.E.m; i++) re[i] -= qp.e[i]
  return { rd, rp, re }
}

/**
 * Solve the QP Newton system by eliminating $\Delta\svec$ and $\Delta\lambdavec$:
 * $\Kmat\Delta\xvec + \Emat^\top\Delta\nuvec = -\rvec_d - \Amat^\top\Smat^{-1}(-\rvec_{sl} + \Lambdamat\rvec_p)$ with
 * $\Kmat = \Qmat + \Amat^\top\Smat^{-1}\Lambdamat\Amat$, and
 * $\Emat\Delta\xvec = -\rvec_e$, then $\Delta\svec = -\rvec_p - \Amat\Delta\xvec$ and
 * $\Delta\lambdavec = \Smat^{-1}(-\rvec_{sl} - \Lambdamat\Delta\svec)$.
 *
 * @param qp The parsed quadratic program.
 * @param s The slacks $\svec$, $m$ positive values (the diagonal of $\Smat$).
 * @param l The inequality multipliers $\lambdavec$, $m$ positive values (the diagonal of $\Lambdamat$).
 * @param rd The stationarity residual $\rvec_d$, $n$ values.
 * @param rp The inequality residual $\rvec_p$, $m$ values.
 * @param re The equality residual $\rvec_e$, $p$ values.
 * @param rsl The complementarity residual $\rvec_{sl}$ to remove ($s_i\lambda_i$ for the predictor; less the centring
 *   target and plus the second-order term for the corrector), $m$ values.
 * @returns The direction `dx`, `dv`, `ds`, `dl`, and `singular`, true when the reduced system could not be solved
 *   (`dx` and `dv` are then zero).
 */
function qpNewton(
  qp: ParsedQP,
  s: Float64Array,
  l: Float64Array,
  rd: Float64Array,
  rp: Float64Array,
  re: Float64Array,
  rsl: Float64Array,
) {
  const { n } = qp
  const m = qp.A.m
  const p = qp.E.m
  const size = n + p
  const K = new Float64Array(size * size)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * size + j] = qp.Q.a[i * n + j]
  const t = new Float64Array(m)
  for (let r = 0; r < m; r++) {
    const w = l[r] / s[r]
    for (let i = 0; i < n; i++) {
      const ai = qp.A.a[r * n + i]
      if (ai === 0) continue
      for (let j = 0; j < n; j++) K[i * size + j] += w * ai * qp.A.a[r * n + j]
    }
    t[r] = (-rsl[r] + l[r] * rp[r]) / s[r]
  }
  for (let r = 0; r < p; r++)
    for (let j = 0; j < n; j++) {
      K[(n + r) * size + j] = qp.E.a[r * n + j]
      K[j * size + n + r] = qp.E.a[r * n + j]
    }
  const At = matTVec(qp.A, t)
  const rhs = new Float64Array(size)
  for (let j = 0; j < n; j++) rhs[j] = -rd[j] - At[j]
  for (let r = 0; r < p; r++) rhs[n + r] = -re[r]
  const { x: sol, singular } = solve(K, size, rhs)
  const dx = sol.slice(0, n)
  const dv = sol.slice(n)
  const Adx = matVec(qp.A, dx)
  const ds = new Float64Array(m)
  const dl = new Float64Array(m)
  for (let r = 0; r < m; r++) {
    ds[r] = -rp[r] - Adx[r]
    dl[r] = (-rsl[r] - l[r] * ds[r]) / s[r]
  }
  return { dx, dv, ds, dl, singular }
}

/**
 * The largest $\alpha \le 1$ with $\vvec + \alpha\,\Delta\vvec \ge \zeros$.
 *
 * @param v The current point $\vvec$, non-negative.
 * @param dv The direction $\Delta\vvec$, of the same length.
 * @returns $\alpha$, 1 when no entry of the direction is negative.
 */
function maxStep(v: Float64Array, dv: Float64Array): number {
  let a = 1
  for (let j = 0; j < v.length; j++) if (dv[j] < 0) a = Math.min(a, -v[j] / dv[j])
  return a
}

/**
 * Assemble an interior-point state: residuals, $\mu$, the objective, and the status (`nonconvex` when flagged, else
 * `optimal` when the residuals and $\mu$ meet the tolerance, else `singular`, `diverged` or `running`).
 *
 * @param qp The parsed quadratic program.
 * @param tol The convergence tolerance, relative to the data's scale.
 * @param x The primal iterate, $n$ values; copied into the state.
 * @param s The slacks, $m$ values; copied into the state.
 * @param l The inequality multipliers, $m$ values; copied into the state.
 * @param v The equality multipliers, $p$ values; copied into the state.
 * @param extra The fields the caller sets: `sigma`, `alpha`, the step count `t`, whether the Newton system was
 *   `singular`, and `nonconvex` for the initial state of a nonconvex problem.
 * @returns The state.
 */
function qpState(
  qp: ParsedQP,
  tol: Scalar,
  x: Float64Array,
  s: Float64Array,
  l: Float64Array,
  v: Float64Array,
  extra: { sigma: Scalar; alpha: Scalar; t: Size; singular: boolean; nonconvex?: boolean },
): QuadraticInteriorPointState {
  const { rd, rp, re } = qpResiduals(qp, x, s, l, v)
  const m = qp.A.m
  const mu = m ? dense.dot(s, l) / m : 0
  const stationarity = dense.maxAbs(rd)
  const primalResidual = Math.max(dense.maxAbs(rp), dense.maxAbs(re))
  const scale = 1 + Math.max(dense.maxAbs(qp.c), dense.maxAbs(qp.b), dense.maxAbs(qp.e), dense.maxAbs(qp.Q.a))
  const nonconvex = extra.nonconvex === true
  const converged = !nonconvex && stationarity < tol * scale && primalResidual < tol * scale && mu < tol * scale
  const diverged =
    !converged &&
    !nonconvex &&
    (extra.singular ||
      !(Math.max(dense.maxAbs(x), dense.maxAbs(l), dense.maxAbs(v)) < 1e12) ||
      !Number.isFinite(stationarity))
  return {
    problem: qp,
    sigma: extra.sigma,
    alpha: extra.alpha,
    t: extra.t,
    x: vector(x),
    s: vector(s),
    lambda: vector(l),
    nu: vector(v),
    mu,
    stationarity,
    primalResidual,
    objective: objectiveOf(qp, x),
    converged,
    diverged,
    terminated: nonconvex,
    status: nonconvex
      ? 'nonconvex'
      : converged
        ? 'optimal'
        : extra.singular
          ? 'singular'
          : diverged
            ? 'diverged'
            : 'running',
  }
}

/**
 * A primal–dual interior-point method for the convex QP `problem` with Mehrotra's predictor–corrector (Nocedal and
 * Wright, 2006, §16.6) as a traceable algorithm with no start. It starts from $\xvec = \zeros$ with slacks
 * $\max(b_i, 1)$ and multipliers of 1 (it need not be feasible) and each step is one predictor–corrector iteration
 * with one step length for all variables; the iterates $\xvec$ trace a path through the interior towards the optimum.
 * The run is done when `converged`; it stops `diverged` when the iterates grow without bound (an infeasible problem),
 * or `singular` (with the `diverged` flag) when the Newton system is singular. The method assumes convexity: when
 * $\Qmat$ has negative curvature on the null space of $\Emat$ (an eigenvalue of the reduced Hessian
 * $\Zmat^\top\Qmat\Zmat$ below $-$`tolerance` relative to $\Qmat$'s scale) the initial state is `nonconvex` and
 * `terminated`, and no step is taken, because the iterates could converge to a saddle or a maximiser and report it
 * optimal. A singular positive semi-definite $\Qmat$ passes.
 *
 * @param problem The quadratic program.
 * @param options The tolerance and the fraction of the step to the boundary taken.
 * @returns The algorithm. Once the run has ended, a step returns the state unchanged.
 *
 * @example The complementarity measure falls to zero
 * // Minimise (x - 1)^2 + (y - 2.5)^2 subject to five linear inequalities (Nocedal and Wright, Example 16.4).
 * const A = [[-1, 2], [1, 2], [1, -2], [-1, 0], [0, -1]]
 * const problem = { Q: [[2, 0], [0, 2]], c: [-2, -5], A, b: [2, 6, 2, 0, 0] }
 * const tr = trace(quadraticInteriorPoint(problem), {}, 50)
 * print('mu =', tr.steps.map((s) => s.mu))
 * const s = tr.steps.at(-1)
 * print('status =', s.status, ' x =', s.x)
 * print('multipliers =', s.lambda)
 *
 * @example A nonconvex problem is refused before the first step
 * const s = run(quadraticInteriorPoint({ Q: [[1, 0], [0, -1]], c: [0, 0], A: [[0, 1], [0, -1]], b: [1, 1] }), {}, 50)
 * print('status =', s.status, ' steps =', s.t)
 */
export function quadraticInteriorPoint(
  problem: QuadraticProgram,
  options: QuadraticInteriorPointOptions = {},
): Algorithm<object, QuadraticInteriorPointState> {
  const qp = parseQP(problem)
  const tol = options.tolerance ?? 1e-9
  const stepFraction = options.stepFraction ?? 0.99
  // Convexity on the feasible set's directions: Q on null(E). Checked once; it does not change along the path.
  const nonconvex = negativeCurvature(qp, qp.E, tol)
  return {
    name: 'quadratic-interior-point',
    init: () => {
      const x = new Float64Array(qp.n)
      // Slacks start at max(b − Ax, 1) so that they are positive; multipliers at 1.
      const s = qp.b.map((b) => Math.max(b, 1))
      const l = new Float64Array(qp.A.m).fill(1)
      const v = new Float64Array(qp.E.m)
      return qpState(qp, tol, x, s, l, v, { sigma: NaN, alpha: 0, t: 0, singular: false, nonconvex })
    },
    step: (st) => {
      if (st.status !== 'running') return st
      const x = Float64Array.from(st.x.data)
      const s = Float64Array.from(st.s.data)
      const l = Float64Array.from(st.lambda.data)
      const v = Float64Array.from(st.nu.data)
      const m = qp.A.m
      const { rd, rp, re } = qpResiduals(qp, x, s, l, v)
      const mu = m ? dense.dot(s, l) / m : 0
      const rsl = s.map((si, r) => si * l[r])
      const aff = qpNewton(qp, s, l, rd, rp, re, rsl)
      const aAff = Math.min(maxStep(s, aff.ds), maxStep(l, aff.dl))
      let muAff = 0
      for (let r = 0; r < m; r++) muAff += (s[r] + aAff * aff.ds[r]) * (l[r] + aAff * aff.dl[r])
      muAff = m ? muAff / m : 0
      const sigma = mu > 0 ? Math.min(1, (muAff / mu) ** 3) : 0
      for (let r = 0; r < m; r++) rsl[r] = s[r] * l[r] + aff.ds[r] * aff.dl[r] - sigma * mu
      const dir = qpNewton(qp, s, l, rd, rp, re, rsl)
      const alpha = Math.min(1, stepFraction * Math.min(maxStep(s, dir.ds), maxStep(l, dir.dl)))
      for (let j = 0; j < qp.n; j++) x[j] += alpha * dir.dx[j]
      for (let r = 0; r < m; r++) {
        s[r] += alpha * dir.ds[r]
        l[r] += alpha * dir.dl[r]
      }
      for (let r = 0; r < qp.E.m; r++) v[r] += alpha * dir.dv[r]
      return qpState(qp, tol, x, s, l, v, {
        sigma,
        alpha,
        t: st.t + 1,
        singular: aff.singular || dir.singular,
      })
    },
  }
}

/** The result of `quadprog`. */
export interface QuadraticProgramResult {
  /** How the method ended, or `limit` when it stopped at `maxSteps`. */
  status: Exclude<QuadraticProgramStatus, 'running'> | 'limit'
  /** The last iterate: the minimiser when `status` is `optimal` (NaN when `infeasible` by the active set). */
  x: Tensor
  /** The objective at `x`. */
  objective: Scalar
  /** Steps taken by the method. */
  steps: Size
  /** The method that produced the result. */
  method: 'active-set' | 'interior-point'
  /** The KKT residuals and multipliers at the returned point. */
  report: KKTReport
}

/** Options for `quadprog`; `maxSteps` defaults to 10 000 (active set) or 200 (interior point). */
export interface QuadprogOptions extends Pick<RunOptions, 'maxSteps'> {
  /** `active-set` (default) or `interior-point`. */
  method?: 'active-set' | 'interior-point'
  /** The method's tolerance (default 1e-10 for the active set, 1e-9 for the interior point). */
  tolerance?: Scalar
  /** A feasible start for the active-set method (ignored by the interior point). */
  x0?: VectorLike
}

/**
 * Solve a convex quadratic program: minimise $\frac{1}{2}\xvec^\top\Qmat\xvec + \cvec^\top\xvec$ subject to
 * $\Amat\xvec \le \bvec$, $\Emat\xvec = \evec$, by the primal active-set method (default) or the interior-point method.
 * The result carries the multipliers and KKT residuals. Throws as `parseQP` does for an ill-formed problem; every
 * other failure is reported in `status`.
 *
 * @param problem The quadratic program.
 * @param options The method, its tolerance, `maxSteps`, and the active set's start `x0`.
 * @returns The point, its objective, the status and the KKT report.
 *
 * @example The minimiser of a small QP by both methods
 * // Minimise (x - 1)^2 + (y - 2.5)^2 subject to five linear inequalities (Nocedal and Wright, Example 16.4).
 * const A = [[-1, 2], [1, 2], [1, -2], [-1, 0], [0, -1]]
 * const problem = { Q: [[2, 0], [0, 2]], c: [-2, -5], A, b: [2, 6, 2, 0, 0] }
 * const r = quadprog(problem)
 * print('active set:', r.x, ' objective =', r.objective, ' multipliers =', r.report.lambda)
 * print('interior point:', quadprog(problem, { method: 'interior-point' }).x)
 *
 * @example The nearest point on a line
 * // Minimise (x^2 + y^2) / 2 subject to x + y = 1: the answer is (0.5, 0.5), with multiplier -0.5.
 * const r = quadprog({ Q: [[1, 0], [0, 1]], c: [0, 0], E: [[1, 1]], e: [1] })
 * print('x =', r.x, ' nu =', r.report.nu)
 */
export function quadprog(problem: QuadraticProgram, options: QuadprogOptions = {}): QuadraticProgramResult {
  const qp = parseQP(problem)
  const { tolerance } = options
  if (options.method === 'interior-point') {
    const s = run(quadraticInteriorPoint(problem, { tolerance }), {}, options.maxSteps ?? 200)
    return {
      status: s.status === 'running' ? 'limit' : s.status,
      x: s.x,
      objective: s.objective,
      steps: s.t,
      method: 'interior-point',
      report: kktReport(qp, s.x, s.lambda, s.nu),
    }
  }
  const s = run(activeSet(problem, { tolerance }), { x0: options.x0 }, options.maxSteps ?? 10_000)
  return {
    status: s.status === 'running' ? 'limit' : s.status,
    x: s.x,
    objective: s.objective,
    steps: s.t,
    method: 'active-set',
    report: kktReport(qp, s.x, s.lambda, s.nu),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Box-constrained QP.

/**
 * A box-constrained QP: minimise $\frac{1}{2}\xvec^\top\Qmat\xvec + \cvec^\top\xvec$ subject to
 * $\mathbf{l} \le \xvec \le \uvec$ (`-Infinity` or `Infinity` for no bound).
 */
export interface BoxQuadraticProblem {
  /** Symmetric Hessian $\Qmat$, $n \times n$ (positive semi-definite for a convex problem). */
  Q: MatrixLike
  /** Linear term $\cvec$, length $n$. */
  c: VectorLike
  /** Lower bounds $\mathbf{l}$, length $n$. */
  lower: VectorLike
  /** Upper bounds $\uvec$, length $n$, each at least its lower bound. */
  upper: VectorLike
}

/** Options for `boxQuadraticProgram`. */
export interface BoxQuadraticProgramOptions {
  /**
   * Stop when $\lVert \xvec - P(\xvec - \nabla f) \rVert_\infty$ is at most this times
   * $1 + \lVert \nabla f \rVert_\infty$ (default 1e-10), $P$ the projection onto the box.
   */
  tolerance?: Scalar
}

/** The start of `boxQuadraticProgram`: `x0`, a point projected onto the box (default: the projection of 0). */
export type BoxQuadraticProgramStart = { x0?: VectorLike }

/** One iterate of the box-QP solver. */
export interface BoxQuadraticProgramState extends Status {
  /** The iterate $\xvec$, inside the box, length $n$. */
  x: Tensor
  /** $\nabla f = \Qmat\xvec + \cvec$. */
  grad: Tensor
  /** $\lVert \xvec - P(\xvec - \nabla f) \rVert_\infty$: zero exactly at a KKT point. */
  projectedGradient: Scalar
  /** Where each variable sits: $-1$ at its lower bound, 1 at its upper bound, 0 free; int32. */
  bounds: Tensor
  /** The point after the projected-gradient (Cauchy) half of the last step. */
  cauchy: Tensor
  /** Whether the last step's subspace Newton half was taken. */
  newton: boolean
  /** The objective at `x`. */
  objective: Scalar
  /** True when the projected gradient meets the tolerance. */
  converged: boolean
  /** True when `x` has a non-finite entry. */
  diverged: boolean
  /** The lower bounds, as read. */
  lower: Tensor
  /** The upper bounds, as read. */
  upper: Tensor
  /** The Hessian, as read. */
  Q: Tensor
  /** The linear term, as read. */
  c: Tensor
}

/**
 * Project a point onto a box, entry by entry.
 *
 * @param x The point (not modified).
 * @param lo The lower bounds.
 * @param hi The upper bounds.
 * @returns A new array, each entry clipped to its bounds.
 */
const project = (x: Float64Array, lo: ArrayLike<number>, hi: ArrayLike<number>) =>
  x.map((v, j) => Math.min(hi[j], Math.max(lo[j], v)))

/**
 * Assemble a box-QP state: the gradient, the projected-gradient measure, which bounds are active, and the objective.
 *
 * @param base The problem's data, carried from state to state.
 * @param tol The convergence tolerance.
 * @param x The iterate, inside the box; copied into the state.
 * @param extra The fields the caller sets: the Cauchy point of the last step, whether its Newton half was taken, and
 *   the step count `t`.
 * @returns The state.
 */
function boxState(
  base: Pick<BoxQuadraticProgramState, 'Q' | 'c' | 'lower' | 'upper'>,
  tol: Scalar,
  x: Float64Array,
  extra: { cauchy: Float64Array; newton: boolean; t: Size },
): BoxQuadraticProgramState {
  const n = x.length
  const Q: Mat = { m: n, n, a: base.Q.data as dense.F64 }
  const c = base.c.data
  const lo = base.lower.data
  const hi = base.upper.data
  const g = matVec(Q, x)
  for (let j = 0; j < n; j++) g[j] += c[j]
  let pg = 0
  const at = new Int32Array(n)
  for (let j = 0; j < n; j++) {
    pg = Math.max(pg, Math.abs(x[j] - Math.min(hi[j], Math.max(lo[j], x[j] - g[j]))))
    at[j] = x[j] <= lo[j] ? -1 : x[j] >= hi[j] ? 1 : 0
  }
  return {
    Q: base.Q,
    c: base.c,
    lower: base.lower,
    upper: base.upper,
    x: vector(x),
    grad: vector(g),
    projectedGradient: pg,
    bounds: intTensor(at),
    cauchy: vector(extra.cauchy),
    newton: extra.newton,
    objective: 0.5 * dense.dot(x, matVec(Q, x)) + dense.dot(c, x),
    t: extra.t,
    converged: pg <= tol * (1 + dense.maxAbs(g)),
    diverged: !dense.allFinite(x),
  }
}

/**
 * Box-constrained convex QP by projected gradient with subspace Newton steps (after Moré and Toraldo, 1991) as a
 * traceable algorithm; `init` takes `{ x0? }`, projected onto the box. Each step (1) takes a projected-gradient step
 * along the projection arc $P(\xvec - t\nabla f)$ with an Armijo backtracking search (from the exact minimiser along
 * $-\nabla f$, halving $t$ at most 60 times), which identifies the bounds that are active, and (2) solves
 * $\Qmat_{FF}\dvec = -\nabla f_F$ on the variables $F$ strictly inside their bounds and halves $t$ along
 * $P(\xvec + t\dvec)$ until the objective does not increase. With $\Qmat$ positive definite it finds the active set
 * in finitely many steps and then converges in one Newton step. Bound multipliers at the end are $\max(\nabla f, 0)$
 * at lower bounds and $\max(-\nabla f, 0)$ at upper bounds. Throws `DomainError` for a lower bound above its upper
 * bound (or NaN), and `ShapeError` when a length disagrees with `c`.
 *
 * @param problem The box-constrained QP.
 * @param options The tolerance on the projected gradient.
 * @returns The algorithm. Once converged, a step returns the state unchanged.
 *
 * @example A Cauchy step finds the active bound, a Newton step the free variable
 * // Minimise x^2 + xy + y^2 - 4x + y over the box [0, 3]^2: y rests on its lower bound and x = 2.
 * const box = { Q: [[2, 1], [1, 2]], c: [-4, 1], lower: [0, 0], upper: [3, 3] }
 * const tr = trace(boxQuadraticProgram(box), { x0: [3, 3] }, 10)
 * print('Cauchy points =', tr.steps.map((s) => Array.from(s.cauchy.data)))
 * print('x =', tr.steps.map((s) => Array.from(s.x.data)))
 * print('at bound =', tr.steps.at(-1).bounds, ' gradient =', tr.steps.at(-1).grad)
 */
export function boxQuadraticProgram(
  problem: BoxQuadraticProblem,
  options: BoxQuadraticProgramOptions = {},
): Algorithm<BoxQuadraticProgramStart, BoxQuadraticProgramState> {
  const where = 'boxQuadraticProgram'
  const c = readVector(problem.c, `${where}: c`)
  const n = c.length
  const Qm = readMatrix(problem.Q, `${where}: Q`, n)
  const lower = readVector(problem.lower, `${where}: lower`, n)
  const upper = readVector(problem.upper, `${where}: upper`, n)
  for (let j = 0; j < n; j++)
    if (!(lower[j] <= upper[j])) throw new DomainError(where, `${where}: empty box for x${j + 1}`)
  const tol = options.tolerance ?? 1e-10
  const base = { Q: matrix(Qm.a, n, n), c: vector(c), lower: vector(lower), upper: vector(upper) }
  return {
    name: 'box-quadratic-program',
    init: (start) => {
      const x0 = start?.x0 !== undefined ? readVector(start.x0, `${where}: x0`, n) : new Float64Array(n)
      const x = project(x0, lower, upper)
      return boxState(base, tol, x, { cauchy: x, newton: false, t: 0 })
    },
    step: (s) => {
      if (s.converged) return s
      const Q: Mat = { m: n, n, a: s.Q.data as dense.F64 }
      const c = s.c.data
      const lo = s.lower.data
      const hi = s.upper.data
      const f = (x: Float64Array) => 0.5 * dense.dot(x, matVec(Q, x)) + dense.dot(c, x)
      const grad = (x: Float64Array) => {
        const g = matVec(Q, x)
        for (let j = 0; j < n; j++) g[j] += c[j]
        return g
      }
      const x = Float64Array.from(s.x.data)
      const g = grad(x)
      const fx = f(x)
      // (1) Cauchy step: backtrack along the projection arc from the exact minimiser of the unconstrained line.
      const gQg = dense.dot(g, matVec(Q, g))
      let t = gQg > 0 ? dense.dot(g, g) / gQg : 1
      let cauchy = x
      for (let k = 0; k < 60; k++) {
        const trial = project(
          x.map((v, j) => v - t * g[j]),
          lo,
          hi,
        )
        const decrease = dense.dot(
          g,
          trial.map((v, j) => v - x[j]),
        )
        if (f(trial) <= fx + 1e-4 * decrease) {
          cauchy = trial
          break
        }
        t /= 2
      }
      // (2) Newton step on the variables strictly inside their bounds after the Cauchy step.
      const free: number[] = []
      for (let j = 0; j < n; j++) if (cauchy[j] > lo[j] && cauchy[j] < hi[j]) free.push(j)
      let next = cauchy
      let newton = false
      if (free.length > 0) {
        const gc = grad(cauchy)
        const k = free.length
        const H = new Float64Array(k * k)
        free.forEach((i, a) => free.forEach((j, b) => (H[a * k + b] = Q.a[i * n + j])))
        const { x: d, singular } = solve(
          H,
          k,
          Float64Array.from(free, (i) => -gc[i]),
        )
        if (!singular) {
          const fc = f(cauchy)
          let step = 1
          for (let it = 0; it < 60; it++) {
            const trial = cauchy.slice()
            free.forEach((i, a) => (trial[i] += step * d[a]))
            const projected = project(trial, lo, hi)
            if (f(projected) <= fc) {
              next = projected
              newton = true
              break
            }
            step /= 2
          }
        }
      }
      return boxState(s, tol, next, { cauchy, newton, t: s.t + 1 })
    },
  }
}

/**
 * Solve a box-constrained QP (a `run` of `boxQuadraticProgram`, at most `maxSteps` steps, default 1000); returns the
 * final state, whose `converged` says whether it met `tolerance`.
 *
 * @param problem The box-constrained QP.
 * @param options The tolerance, `maxSteps`, and the start `x0` (projected onto the box; default the projection of 0).
 * @returns The final state.
 *
 * @example The minimiser of a quadratic on a box
 * // Unconstrained, x^2 + xy + y^2 - 4x + y is least at (3, -2); on [0, 3]^2 the minimiser is (2, 0).
 * const s = boxQuadprog({ Q: [[2, 1], [1, 2]], c: [-4, 1], lower: [0, 0], upper: [3, 3] })
 * print('x =', s.x, ' objective =', s.objective)
 * print('converged =', s.converged, ' steps =', s.t)
 */
export function boxQuadprog(
  problem: BoxQuadraticProblem,
  options: BoxQuadraticProgramOptions & Pick<RunOptions, 'maxSteps'> & BoxQuadraticProgramStart = {},
): BoxQuadraticProgramState {
  return run(boxQuadraticProgram(problem, options), { x0: options.x0 }, options.maxSteps ?? 1000)
}
