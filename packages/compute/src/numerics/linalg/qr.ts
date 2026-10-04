/**
 * QR factorisation by Householder reflections (Golub and Van Loan, 2013, Algorithm 5.2.1), with LAPACK's sign
 * convention (`dgeqrf`/`dlarfg`): each reflector maps its column to −sign(α)·‖x‖·e₁, so R's diagonal may be negative,
 * and a column that is already zero below the diagonal is left alone. This matches NumPy's `np.linalg.qr`.
 */

import {
  add,
  concat,
  definePrimitive,
  isTraced,
  matmul,
  mul,
  type Op,
  shapeOfValue,
  slice,
  sub,
  type Tensor,
  type TensorResult,
  transpose,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { NotDifferentiableError, NumericalError } from 'aifn-compute/foundation/errors'
import type { MatrixLike, Status } from 'aifn-compute/foundation/contracts'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { asMatrix, dense, EPS, matrix, vector } from './dense'
import {
  concreteExamples,
  float64Aval,
  kernelBatch,
  lowerMask,
  pack,
  packRaw,
  scaleOf,
  symmetricFromLower,
  unpack,
} from './rules'
import { solveTriangular } from './triangular'

/** The result of `qr` (Q and R traced for traced input). */
export type QR<T = Tensor> = {
  /** Orthonormal columns: m×k for `reduced` (k = min(m, n)), m×m for `complete`. */
  Q: T
  /** Upper triangular (trapezoidal): k×n for `reduced`, m×n for `complete`. */
  R: T
}

/**
 * One Householder step on the row-major m×n working matrix r, in place: the reflector H = I − τvvᵀ (v[0] = 1, stored
 * from row j) that maps column j below the diagonal to −sign(α)·‖x‖·e₁, applied to columns j … n − 1. A column already
 * zero below the diagonal gives H = I (τ = 0), as `dlarfg`. `qr` and `householderSteps` both run this step.
 */
function reflectColumn(r: Float64Array, m: number, n: number, j: number): { v: Float64Array; tau: number } {
  const alpha = r[j * n + j]
  let sigma = 0
  for (let i = j + 1; i < m; i++) sigma += r[i * n + j] * r[i * n + j]
  const v = new Float64Array(m - j)
  v[0] = 1
  if (sigma === 0) return { v, tau: 0 }
  const norm = Math.hypot(alpha, Math.sqrt(sigma))
  const beta = alpha >= 0 ? -norm : norm
  const tau = (beta - alpha) / beta
  const scale = 1 / (alpha - beta)
  for (let i = j + 1; i < m; i++) v[i - j] = r[i * n + j] * scale
  // Apply H to the trailing columns of R.
  for (let c = j; c < n; c++) {
    let s = 0
    for (let i = j; i < m; i++) s += v[i - j] * r[i * n + c]
    s *= tau
    for (let i = j; i < m; i++) r[i * n + c] -= s * v[i - j]
  }
  r[j * n + j] = beta
  for (let i = j + 1; i < m; i++) r[i * n + j] = 0
  return { v, tau }
}

/** Q ← Q·H in place for H = I − τvvᵀ acting on coordinates j … m − 1 (v stored from j); Q is m×m row-major. */
function applyReflectorRight(q: Float64Array, m: number, v: Float64Array, tau: number, j: number): void {
  if (tau === 0) return
  for (let row = 0; row < m; row++) {
    let s = 0
    for (let i = j; i < m; i++) s += q[row * m + i] * v[i - j]
    s *= tau
    for (let i = j; i < m; i++) q[row * m + i] -= s * v[i - j]
  }
}

/** Householder QR of a dense copy of A: `reflectColumn` for each of the k = min(m, n) columns, then Q. */
function householder(a: Value, mode: 'reduced' | 'complete'): QR {
  const { m, n, a: r } = dense(a, 'qr')
  const k = Math.min(m, n)
  // Householder vectors v (v[0] = 1 implied by storing it explicitly) and their scalars τ, with H = I − τvvᵀ.
  const vs: Float64Array[] = []
  const taus: number[] = []
  for (let j = 0; j < k; j++) {
    const { v, tau } = reflectColumn(r, m, n, j)
    vs.push(v)
    taus.push(tau)
  }
  const cols = mode === 'complete' ? m : k
  // Q = H₀H₁⋯H_{k−1} applied to the first `cols` columns of I, accumulated backwards.
  const q = new Float64Array(m * cols)
  for (let i = 0; i < Math.min(m, cols); i++) q[i * cols + i] = 1
  for (let j = k - 1; j >= 0; j--) {
    const v = vs[j]
    const tau = taus[j]
    if (tau === 0) continue
    for (let c = 0; c < cols; c++) {
      let s = 0
      for (let i = j; i < m; i++) s += v[i - j] * q[i * cols + c]
      s *= tau
      for (let i = j; i < m; i++) q[i * cols + c] -= s * v[i - j]
    }
  }
  const rows = mode === 'complete' ? m : k
  return { Q: matrix(q, m, cols), R: matrix(r.slice(0, rows * n), rows, n) }
}

/** One state of `householderSteps`. */
export interface HouseholderState extends Status {
  /** The working matrix: columns 0 … column − 1 reduced to upper-triangular form (m×n). */
  R: Tensor
  /** The product H₀H₁⋯H_{column−1} of the reflectors so far (m×m, orthogonal); Q·R = A at every step. */
  Q: Tensor
  /** The reflector vector v of the last step, zero above its column (length m; zeros at t = 0). */
  reflector: Tensor
  /** τ of the last step (H = I − τvvᵀ; 0 at t = 0 or when the column was already reduced). */
  tau: number
  /** The next column to reduce; k = min(m, n) when done. */
  column: number
  /** ‖A − QR‖_F, which stays at rounding level. */
  residual: number
  done: boolean
}

/**
 * Householder QR as a traceable algorithm (Golub and Van Loan, 2013, Algorithm 5.2.1): each step reflects one column
 * onto a multiple of e₁ below the diagonal and applies the reflector to the trailing columns, so after step j the first
 * j columns of R are upper triangular and Q = H₀⋯H_{j−1} is orthogonal with QR = A throughout. Done after
 * k = min(m, n) steps. The step is `qr`'s own (`reflectColumn`); `qr` runs it without keeping states.
 */
export function householderSteps(A: MatrixLike): Algorithm<void, HouseholderState> {
  const { m, n, a: a0 } = dense(asMatrix(A, 'householderSteps'), 'householderSteps')
  const k = Math.min(m, n)
  const residual = (q: Float64Array, r: Float64Array) => {
    let s = 0
    for (let i = 0; i < m; i++)
      for (let c = 0; c < n; c++) {
        let qr = 0
        for (let l = 0; l < m; l++) qr += q[i * m + l] * r[l * n + c]
        s += (a0[i * n + c] - qr) ** 2
      }
    return Math.sqrt(s)
  }
  return {
    name: 'householder-qr',
    init: () => {
      const q = new Float64Array(m * m)
      for (let i = 0; i < m; i++) q[i * m + i] = 1
      return {
        t: 0,
        R: matrix(Float64Array.from(a0), m, n),
        Q: matrix(q, m, m),
        reflector: vector(new Float64Array(m)),
        tau: 0,
        column: 0,
        residual: 0,
        done: k === 0,
      }
    },
    step: (s) => {
      if (s.done) return { ...s, t: s.t + 1 }
      const j = s.column
      const r = dense(s.R, 'householderSteps').a
      const q = dense(s.Q, 'householderSteps').a
      const { v, tau } = reflectColumn(r, m, n, j)
      applyReflectorRight(q, m, v, tau, j)
      const full = new Float64Array(m)
      full.set(v, j)
      return {
        t: s.t + 1,
        R: matrix(r, m, n),
        Q: matrix(q, m, m),
        reflector: vector(full),
        tau,
        column: j + 1,
        residual: residual(q, r),
        done: j + 1 === k,
      }
    },
    done: (s) => s.done,
  }
}

/** X R⁻¹ for upper-triangular R. */
const rightSolve = (x: Value, R: Value) =>
  transpose(solveTriangular(R, transpose(x), { lower: false, transpose: true }))
/** X R⁻ᵀ for upper-triangular R. */
const rightSolveTransposed = (x: Value, R: Value) => transpose(solveTriangular(R, transpose(x), { lower: false }))

/**
 * A rank-deficient A (a zero diagonal entry of R) has no derivative of Q: report it rather than divide by zero, example
 * by example inside `vmap`.
 */
function refuseRankDeficient(R: Value, k: number): void {
  const examples = concreteExamples(R)
  if (examples === null) return
  const n = shapeOfValue(R)[1]
  examples.forEach((r, b) => {
    const scale = scaleOf(r)
    for (let i = 0; i < k; i++) {
      if (Math.abs(r[i * n + i]) <= k * EPS * scale) {
        const which = examples.length > 1 ? ` of batch example ${b}` : ''
        throw new NumericalError(
          'qr',
          `qr: the matrix${which} is rank deficient (R[${i}, ${i}] is zero), so Q has no derivative`,
          'singular',
        )
      }
    }
  })
}

/** The tangents (Q̇, Ṙ) of a square or tall QR, A = QR with R k×k invertible, for a tangent Ȧ. */
function jvpTall(Q: Value, R: Value, t: Value, k: number): [Value, Value] {
  const C = matmul(transpose(Q), rightSolve(t, R))
  const low = mul(C, lowerMask(k, false))
  const omega = sub(low, transpose(low))
  const dR = matmul(sub(C, omega), R)
  const dQ = add(sub(rightSolve(t, R), matmul(Q, C)), matmul(Q, omega))
  return [dQ, dR]
}

/** The adjoint Ā of a square or tall QR, A = QR with R k×k invertible, for cotangents (Q̄, R̄). */
function vjpTall(Q: Value, R: Value, gQ: Value, gR: Value, k: number): Value {
  const M = sub(matmul(R, transpose(gR)), matmul(transpose(gQ), Q))
  return rightSolveTransposed(add(gQ, matmul(Q, symmetricFromLower(M, k))), R)
}

// Rules (Seeger et al., 2017, "Auto-differentiating linear algebra", arXiv:1710.08717; Walter and Lehmann, 2018; the
// wide case from Liao et al., 2019, "Differentiable programming tensor networks", §B). For A = QR with k×k R
// invertible and C = QᵀȦR⁻¹, Ω = tril(C, −1) − tril(C, −1)ᵀ:  Ṙ = (C − Ω)R and Q̇ = ȦR⁻¹ − QC + QΩ; the adjoint is
// Ā = (Q̄ + Q·copyltu(M))R⁻ᵀ with M = RR̄ᵀ − Q̄ᵀQ and copyltu the symmetric matrix of M's lower triangle. A wide A = [X Y]
// (X m×m) factors X = QU and gives R = [U QᵀY]: the square rule applies to X with Q̄ + YR̄_Yᵀ, and Ȳ = QR̄_Y.
// The output is Q (m×k) and R (k×n) packed into one vector, for the reduced factorisation.
const qrOp: Op<undefined> = definePrimitive<undefined>({
  id: 'numerics/linalg/qr',
  arity: 1,
  impl: ([a]) => {
    const f = householder(a, 'reduced')
    return packRaw([f.Q.data as Float64Array, f.R.data as Float64Array])
  },
  vjp: (g, [a], out) => {
    const [m, n] = shapeOfValue(a)
    const k = Math.min(m, n)
    const [Q, R] = unpack(out, [
      [m, k],
      [k, n],
    ])
    const [gQ, gR] = unpack(g, [
      [m, k],
      [k, n],
    ])
    refuseRankDeficient(R, k)
    if (m >= n) return [vjpTall(Q, R, gQ, gR, k)]
    const U = slice(R, null, [0, m])
    const gU = slice(gR, null, [0, m])
    const gRy = slice(gR, null, [m, n])
    const Y = slice(a, null, [m, n])
    const gX = vjpTall(Q, U, add(gQ, matmul(Y, transpose(gRy))), gU, k)
    return [concat([gX, matmul(Q, gRy)], 1)]
  },
  jvp: ([t], [a], out) => {
    if (t === null) return null
    const [m, n] = shapeOfValue(a)
    const k = Math.min(m, n)
    const [Q, R] = unpack(out, [
      [m, k],
      [k, n],
    ])
    refuseRankDeficient(R, k)
    if (m >= n) return pack(jvpTall(Q, R, t, k))
    const [dQ, dU] = jvpTall(Q, slice(R, null, [0, m]), slice(t, null, [0, m]), k)
    const Y = slice(a, null, [m, n])
    const dRy = add(matmul(transpose(dQ), Y), matmul(transpose(Q), slice(t, null, [m, n])))
    return pack([dQ, concat([dU, dRy], 1)])
  },
  batch: kernelBatch('numerics/linalg/qr'),
  shape: ([a]) => {
    const [m, n] = a.shape
    const k = Math.min(m, n)
    return float64Aval([m * k + k * n])
  },
  doc: { note: 'qr-decomposition', summary: 'The reduced QR factorisation by Householder reflections.' },
  test: {
    rtol: 1e-4,
    cases: (draw) => [
      { inputs: [draw([4, 3])], params: undefined },
      { inputs: [draw([2, 3])], params: undefined },
    ],
  },
})

/**
 * QR factorisation A = QR of an m×n matrix by Householder reflections. `mode` `reduced` (default) gives the thin
 * factors; `complete` gives a square Q. Differentiable in both modes for the reduced factorisation of a matrix of full
 * rank (Seeger et al., 2017); a rank-deficient A throws `NumericalError` ('singular') when differentiated. The complete
 * factorisation of a tall matrix is not differentiable (its extra columns of Q are not unique) and refuses traced input.
 */
export function qr<X extends Value>(
  a: X,
  { mode = 'reduced' }: { mode?: 'reduced' | 'complete' } = {},
): QR<TensorResult<X>> {
  if (!isTraced(a)) return householder(a, mode) as QR<TensorResult<X>>
  const [m, n] = shapeOfValue(a)
  if (mode === 'complete' && m > n) {
    throw new NotDifferentiableError(
      'qr',
      "qr: the complete factorisation of a tall matrix has no derivative (Q's extra columns are not unique); use mode 'reduced'",
    )
  }
  const k = Math.min(m, n)
  const [Q, R] = unpack(qrOp([a], undefined), [
    [m, k],
    [k, n],
  ])
  return { Q, R } as QR<TensorResult<X>>
}
