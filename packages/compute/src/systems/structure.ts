/**
 * Structural properties of linear systems: controllability and observability by Kalman's rank tests, and the
 * controllability and observability Gramians (Kalman, 1960, "On the general theory of control systems"; Antsaklis &
 * Michel, 2006, "A Linear Systems Primer", §3.4 and §4.5).
 *
 * Every function realises its system in state space first (`toStateSpace`), so a transfer function is tested in its
 * controllable canonical form. Ranks are numerical, from the singular values; the Gramians solve a Lyapunov equation
 * (`lyapunov`, continuous or discrete by the system's domain) and are reported as `null` for an unstable system.
 */

import { lyapunov, svd } from 'aifn-compute/numerics/linalg'
import { dense, fromData, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { LtiSystem, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { stability, toStateSpace } from './system'

const { matMul } = dense

/**
 * $\Amat^\top$ of a row-major $r \times c$ matrix.
 *
 * @param a The matrix $\Amat$, row-major, $rc$ values; not modified.
 * @param r Its rows.
 * @param c Its columns.
 * @returns $\Amat^\top$, row-major, $c \times r$.
 */
function transposed(a: ArrayLike<number>, r: Size, c: Size): dense.F64 {
  const out = new Float64Array(r * c)
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = a[i * c + j]
  return out
}

/**
 * The Krylov matrix $[\Bmat, \Amat\Bmat, \Amat^2\Bmat, \dots, \Amat^{n-1}\Bmat]$ ($n \times nm$), row-major.
 *
 * @param A The square matrix $\Amat$, row-major, $n^2$ values.
 * @param B The block $\Bmat$, row-major, $n \times m$.
 * @param n The order of $\Amat$, and the number of blocks.
 * @param m The columns of $\Bmat$.
 * @returns The $n$ blocks side by side, row-major, $n \times nm$.
 */
function krylov(A: dense.F64, B: dense.F64, n: Size, m: Size): dense.F64 {
  const out = new Float64Array(n * n * m)
  let block: dense.F64 = B
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i * n * m + k * m + j] = block[i * m + j]
    block = matMul(A, block, n, n, m)
  }
  return out
}

/** A rank test's result. */
export type RankTest = {
  /** The test matrix. */
  matrix: Matrix
  /** Its numerical rank: the singular values above the tolerance. */
  rank: Size
  /** Its singular values, descending; the smallest says how near the test is to failing. */
  singularValues: Vector
  /** True when the rank equals the state dimension $n$. */
  full: boolean
}

/**
 * The numerical rank of a test matrix by SVD, with numpy's `matrix_rank` tolerance
 * $\sigma_1 \max(\text{rows}, \text{cols})\,\varepsilon$ unless one is given.
 *
 * @param m The test matrix, row-major; kept (as a tensor) in the result.
 * @param rows Its rows.
 * @param cols Its columns.
 * @param n The state dimension, the rank that makes the test pass.
 * @param tolerance Singular values at or below this do not count; default numpy's.
 * @returns The matrix, its rank and singular values, and whether the rank is $n$.
 */
function rankTest(m: dense.F64, rows: Size, cols: Size, n: Size, tolerance?: Scalar): RankTest {
  const matrix = fromData(m, [rows, cols])
  const S = rows * cols === 0 ? fromData(new Float64Array(0), [0]) : svd(matrix).S
  const s = toFlat(S)
  const tol = tolerance ?? (s.length ? s[0] * Math.max(rows, cols) * Number.EPSILON : 0)
  const rank = s.filter((v) => v > tol).length
  return { matrix, rank, singularValues: S, full: rank === n }
}

/**
 * Kalman's controllability test: the pair $(\Amat, \Bmat)$ is controllable (every state reachable from the origin)
 * iff $\Ccal = [\Bmat, \Amat\Bmat, \dots, \Amat^{n-1}\Bmat]$ has rank $n$. Rank by SVD with numpy's default
 * tolerance unless `tolerance` is given; the smallest singular value says how nearly uncontrollable it is.
 *
 * @param sys The system (realised in state space first).
 * @param options The rank tolerance.
 * @param options.tolerance Singular values of $\Ccal$ at or below this count as zero; default
 *   $\sigma_1 \cdot nm \cdot \varepsilon$.
 * @returns $\Ccal$ ($n \times nm$), its rank and singular values, and whether it has full rank.
 *
 * @example The double integrator is controllable from its force
 * const integrator = stateSpace({ A: [[0, 1], [0, 0]], B: [0, 1] })
 * const test = controllability(integrator)
 * print('C =', test.matrix)
 * print('rank =', test.rank, ' full =', test.full)
 *
 * @example Two decoupled modes, the input reaching only one of them
 * const sys = stateSpace({ A: [[-1, 0], [0, -2]], B: [1, 0] })
 * const test = controllability(sys)
 * print('rank =', test.rank, ' full =', test.full)
 * print('singular values =', test.singularValues)
 */
export function controllability(sys: LtiSystem, { tolerance }: { tolerance?: Scalar } = {}): RankTest {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  return rankTest(krylov(dense.data(r.A), dense.data(r.B), n, m), n, n * m, n, tolerance)
}

/**
 * Kalman's observability test: $(\Amat, \Cmat)$ is observable (the initial state is determined by the output) iff
 * $\Ocal = [\Cmat; \Cmat\Amat; \dots; \Cmat\Amat^{n-1}]$ has rank $n$. The dual of controllability:
 * $\Ocal = \Ccal(\Amat^\top, \Cmat^\top)^\top$.
 *
 * @param sys The system (realised in state space first).
 * @param options The rank tolerance.
 * @param options.tolerance Singular values of $\Ocal$ at or below this count as zero; default
 *   $\sigma_1 \cdot np \cdot \varepsilon$.
 * @returns $\Ocal$ ($np \times n$), its rank and singular values, and whether it has full rank.
 *
 * @example The double integrator: observable from its position, not from its velocity
 * const A = [[0, 1], [0, 0]]
 * print('position measured:', observability(stateSpace({ A, B: [0, 1], C: [1, 0] })).full)
 * print('velocity measured:', observability(stateSpace({ A, B: [0, 1], C: [0, 1] })).full)
 * print('O =', observability(stateSpace({ A, B: [0, 1], C: [1, 0] })).matrix)
 */
export function observability(sys: LtiSystem, { tolerance }: { tolerance?: Scalar } = {}): RankTest {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const p = r.C.shape[0]
  const K = krylov(transposed(dense.data(r.A), n, n), transposed(dense.data(r.C), p, n), n, p)
  return rankTest(transposed(K, n, n * p), n * p, n, n, tolerance)
}

/**
 * A Gramian: `W` is the $n \times n$ matrix, or `null` when the system is not stable (the defining integral or sum
 * diverges), and `stable` says which.
 */
export type Gramian = { W: Matrix | null; stable: boolean }

/**
 * The solution of the Lyapunov equation of a Gramian, $\Amat\Wmat + \Wmat\Amat^\top + \Qmat = \zeros$ (or
 * $\Wmat = \Amat\Wmat\Amat^\top + \Qmat$ for a discrete system), after checking that the system is stable.
 *
 * @param sys The system, whose stability and domain decide the equation.
 * @param A The matrix of the equation, row-major $n \times n$: $\Amat$, or $\Amat^\top$ for the observability Gramian.
 * @param Q The constant term, row-major $n \times n$: $\Bmat\Bmat^\top$ or $\Cmat^\top\Cmat$.
 * @param n The state dimension.
 * @returns The Gramian, or `null` with `stable: false`.
 */
function gramian(sys: LtiSystem, A: dense.F64, Q: dense.F64, n: Size): Gramian {
  const { stable } = stability(sys)
  if (!stable) return { W: null, stable }
  return { W: lyapunov(fromData(A, [n, n]), fromData(Q, [n, n]), { discrete: sys.domain === 'discrete' }).X, stable }
}

/**
 * The controllability Gramian $\Wmat_c = \int_0^\infty e^{\Amat t}\Bmat\Bmat^\top e^{\Amat^\top t} \, dt$ (or
 * $\sum_k \Amat^k \Bmat\Bmat^\top (\Amat^\top)^k$ for a discrete system), which solves
 * $\Amat\Wmat + \Wmat\Amat^\top + \Bmat\Bmat^\top = \zeros$ (discrete:
 * $\Wmat = \Amat\Wmat\Amat^\top + \Bmat\Bmat^\top$). It exists for a stable system;
 * $\xvec^\top \Wmat_c^{-1} \xvec$ is the least input energy that reaches $\xvec$.
 *
 * @param sys The system (realised in state space first).
 * @returns $\Wmat_c$ ($n \times n$), or `null` when the system is not stable.
 *
 * @example Modes at $-1$ and $-2$, both driven: $W_{ij} = 1/(\lvert \lambda_i \rvert + \lvert \lambda_j \rvert)$
 * const sys = stateSpace({ A: [[-1, 0], [0, -2]], B: [1, 1], C: [1, 1] })
 * print('Wc =', controllabilityGramian(sys).W)
 * print('unstable =', controllabilityGramian(stateSpace({ A: [[1]], B: [1] })))
 */
export function controllabilityGramian(sys: LtiSystem): Gramian {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  const B = dense.data(r.B)
  return gramian(sys, dense.data(r.A), matMul(B, transposed(B, n, m), n, m, n), n)
}

/**
 * The observability Gramian $\Wmat_o = \int_0^\infty e^{\Amat^\top t}\Cmat^\top\Cmat e^{\Amat t} \, dt$, solving
 * $\Amat^\top\Wmat + \Wmat\Amat + \Cmat^\top\Cmat = \zeros$ (discrete:
 * $\Wmat = \Amat^\top\Wmat\Amat + \Cmat^\top\Cmat$). It exists for a stable system; $\xvec_0^\top \Wmat_o \xvec_0$ is
 * the output energy of the free response from $\xvec_0$.
 *
 * @param sys The system (realised in state space first).
 * @returns $\Wmat_o$ ($n \times n$), or `null` when the system is not stable.
 *
 * @example The dual of the controllability example, and a discrete pole at 0.5: $1/(1 - 0.5^2) = 4/3$
 * const sys = stateSpace({ A: [[-1, 0], [0, -2]], B: [1, 1], C: [1, 1] })
 * print('Wo =', observabilityGramian(sys).W)
 * const d = stateSpace({ A: [[0.5]], B: [1], C: [1], dt: 1 })
 * print('discrete Wo =', observabilityGramian(d).W)
 */
export function observabilityGramian(sys: LtiSystem): Gramian {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const p = r.C.shape[0]
  const C = dense.data(r.C)
  return gramian(sys, transposed(dense.data(r.A), n, n), matMul(transposed(C, p, n), C, n, p, n), n)
}
