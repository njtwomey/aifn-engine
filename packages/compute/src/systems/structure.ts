/**
 * Structural properties of linear systems: controllability and observability by Kalman's rank tests, and the
 * controllability and observability Gramians (Kalman, 1960, "On the general theory of control systems"; Antsaklis &
 * Michel, 2006, "A Linear Systems Primer", §3.4 and §4.5).
 */

import { lyapunov, svd } from 'aifn-compute/numerics/linalg'
import { dense, fromData, toFlat, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { LtiSystem, Scalar, Size } from 'aifn-compute/foundation/contracts'
import { stability, toStateSpace } from './system'

const { matMul } = dense

/** Aᵀ of a row-major r×c matrix. */
function transposed(a: ArrayLike<number>, r: Size, c: Size): dense.F64 {
  const out = new Float64Array(r * c)
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = a[i * c + j]
  return out
}

/** The Krylov matrix [B, AB, A²B, …, A^{n−1}B] (n × nm), row-major. */
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
  /** Its numerical rank and singular values (descending). */
  rank: Size
  singularValues: Vector
  /** True when the rank equals the state dimension n. */
  full: boolean
}

function rankTest(m: dense.F64, rows: Size, cols: Size, n: Size, tolerance?: Scalar): RankTest {
  const matrix = fromData(m, [rows, cols])
  const S = rows * cols === 0 ? fromData(new Float64Array(0), [0]) : svd(matrix).S
  const s = toFlat(S)
  const tol = tolerance ?? (s.length ? s[0] * Math.max(rows, cols) * Number.EPSILON : 0)
  const rank = s.filter((v) => v > tol).length
  return { matrix, rank, singularValues: S, full: rank === n }
}

/**
 * Kalman's controllability test: the pair (A, B) is controllable (every state reachable from the origin) iff
 * 𝒞 = [B, AB, …, A^{n−1}B] has rank n. Rank by SVD with numpy's default tolerance unless `tolerance` is given; the
 * smallest singular value says how nearly uncontrollable it is.
 */
export function controllability(sys: LtiSystem, { tolerance }: { tolerance?: Scalar } = {}): RankTest {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  return rankTest(krylov(dense.data(r.A), dense.data(r.B), n, m), n, n * m, n, tolerance)
}

/**
 * Kalman's observability test: (A, C) is observable (the initial state is determined by the output) iff
 * 𝒪 = [C; CA; …; CA^{n−1}] has rank n. The dual of controllability: 𝒪 = 𝒞(Aᵀ, Cᵀ)ᵀ.
 */
export function observability(sys: LtiSystem, { tolerance }: { tolerance?: Scalar } = {}): RankTest {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const p = r.C.shape[0]
  const K = krylov(transposed(dense.data(r.A), n, n), transposed(dense.data(r.C), p, n), n, p)
  return rankTest(transposed(K, n, n * p), n * p, n, n, tolerance)
}

/** A Gramian, or null when the system is not stable (the defining integral or sum diverges). */
export type Gramian = { W: Matrix | null; stable: boolean }

function gramian(sys: LtiSystem, A: dense.F64, Q: dense.F64, n: Size): Gramian {
  const { stable } = stability(sys)
  if (!stable) return { W: null, stable }
  return { W: lyapunov(fromData(A, [n, n]), fromData(Q, [n, n]), { discrete: sys.domain === 'discrete' }).X, stable }
}

/**
 * The controllability Gramian W_c = ∫₀^∞ e^{At}BBᵀe^{Aᵀt} dt (or Σ A^k BBᵀ (Aᵀ)^k for a discrete system), which
 * solves A W + W Aᵀ + BBᵀ = 0. It exists for a stable system; xᵀ W_c⁻¹ x is the least input energy that reaches x.
 */
export function controllabilityGramian(sys: LtiSystem): Gramian {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  const B = dense.data(r.B)
  return gramian(sys, dense.data(r.A), matMul(B, transposed(B, n, m), n, m, n), n)
}

/** The observability Gramian W_o, solving Aᵀ W + W A + CᵀC = 0 (discrete: W = AᵀWA + CᵀC). */
export function observabilityGramian(sys: LtiSystem): Gramian {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const p = r.C.shape[0]
  const C = dense.data(r.C)
  return gramian(sys, transposed(dense.data(r.A), n, n), matMul(transposed(C, p, n), C, n, p, n), n)
}
