/**
 * QR factorisation by Householder reflections (Golub and Van Loan, 2013, Algorithm 5.2.1), with LAPACK's sign
 * convention (`dgeqrf`/`dlarfg`): each reflector maps its column to
 * $-\operatorname{sign}(\alpha) \cdot \lVert \xvec \rVert \cdot \evec_1$, so $\Rmat$'s diagonal may be negative, and a
 * column that is already zero below the diagonal is left alone. This matches NumPy's `np.linalg.qr`.
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

/** The result of `qr` ($\Qmat$ and $\Rmat$ traced for traced input). */
export type QR<T = Tensor> = {
  /** Orthonormal columns: $m \times k$ for `reduced` ($k = \min(m, n)$), $m \times m$ for `complete`. */
  Q: T
  /** Upper triangular (trapezoidal): $k \times n$ for `reduced`, $m \times n$ for `complete`. */
  R: T
}

/**
 * One Householder step on the row-major $m \times n$ working matrix `r`, in place: the reflector
 * $\Hmat = \Imat - \tau\vvec\vvec^\top$ ($v_0 = 1$, stored from row $j$) that maps column $j$ below the diagonal to
 * $-\operatorname{sign}(\alpha) \cdot \lVert \xvec \rVert \cdot \evec_1$, applied to columns $j, \dots, n - 1$. A
 * column already zero below the diagonal gives $\Hmat = \Imat$ ($\tau = 0$), as `dlarfg`. `qr` and
 * `householderSteps` both run this step.
 *
 * @param r The working matrix as a row-major array of $mn$ values, whose columns before $j$ are already reduced.
 *   Modified in place: column $j$ gets its new diagonal entry and zeros below it, and columns after $j$ are
 *   reflected from row $j$ down. Left untouched when the column is already zero below the diagonal.
 * @param m The number of rows of the working matrix.
 * @param n The number of columns of the working matrix.
 * @param j The index (from 0) of the column to reduce, which is also the row its reflector starts at.
 * @returns The reflector: `v`, its vector for rows $j, \dots, m - 1$ ($m - j$ values, the first of them 1), and
 *   `tau`, its scalar $\tau$ (0 when nothing was reflected).
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

/**
 * $\Qmat \leftarrow \Qmat\Hmat$ in place for $\Hmat = \Imat - \tau\vvec\vvec^\top$ acting on coordinates
 * $j, \dots, m - 1$ ($\vvec$ stored from $j$); $\Qmat$ is $m \times m$ row-major.
 *
 * @param q The matrix $\Qmat$ as a row-major array of $m^2$ values. Modified in place: its columns from $j$ on are
 *   replaced by their reflection, and the earlier ones are untouched.
 * @param m The number of rows (and columns) of $\Qmat$.
 * @param v The reflector's vector for coordinates $j, \dots, m - 1$ ($m - j$ values), as `reflectColumn` returns it.
 *   Read only.
 * @param tau The reflector's scalar $\tau$; 0 means the identity, and `q` is left as it is.
 * @param j The first coordinate (from 0) the reflector acts on.
 */
function applyReflectorRight(q: Float64Array, m: number, v: Float64Array, tau: number, j: number): void {
  if (tau === 0) return
  for (let row = 0; row < m; row++) {
    let s = 0
    for (let i = j; i < m; i++) s += q[row * m + i] * v[i - j]
    s *= tau
    for (let i = j; i < m; i++) q[row * m + i] -= s * v[i - j]
  }
}

/**
 * Householder QR of a dense copy of $\Amat$: `reflectColumn` for each of the $k = \min(m, n)$ columns, then $\Qmat$.
 *
 * @param a The matrix $\Amat$ to factor, $m \times n$. Its values are copied, so it is not modified.
 * @param mode Which factors to build: 'reduced' gives $\Qmat$ as $m \times k$ and $\Rmat$ as $k \times n$, 'complete'
 *   gives a square $m \times m$ $\Qmat$ and an $m \times n$ $\Rmat$.
 * @returns The factors `Q` and `R` as tensors, with $\Qmat\Rmat = \Amat$.
 */
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
  /** The working matrix: columns $0, \dots, \text{column} - 1$ reduced to upper-triangular form ($m \times n$). */
  R: Tensor
  /**
   * The product $\Hmat_0 \Hmat_1 \cdots \Hmat_{\text{column} - 1}$ of the reflectors so far ($m \times m$,
   * orthogonal); $\Qmat\Rmat = \Amat$ at every step.
   */
  Q: Tensor
  /** The reflector vector $\vvec$ of the last step, zero above its column (length $m$; zeros at $t = 0$). */
  reflector: Tensor
  /**
   * $\tau$ of the last step ($\Hmat = \Imat - \tau\vvec\vvec^\top$; 0 at $t = 0$ or when the column was already
   * reduced).
   */
  tau: number
  /** The next column to reduce; $k = \min(m, n)$ when done. */
  column: number
  /** $\lVert \Amat - \Qmat\Rmat \rVert_F$, which stays at rounding level. */
  residual: number
  /** True once every column has been reflected. */
  done: boolean
}

/**
 * Householder QR as a traceable algorithm (Golub and Van Loan, 2013, Algorithm 5.2.1): each step reflects one column
 * onto a multiple of $\evec_1$ below the diagonal and applies the reflector to the trailing columns, so after step $j$
 * the first $j$ columns of $\Rmat$ are upper triangular and $\Qmat = \Hmat_0 \cdots \Hmat_{j-1}$ is orthogonal with
 * $\Qmat\Rmat = \Amat$ throughout. Done after $k = \min(m, n)$ steps. The step is `qr`'s own (`reflectColumn`); `qr`
 * runs it without keeping states.
 *
 * @param A The matrix $\Amat$ to factor, $m \times n$, as a tensor or nested rows of numbers. Its values are copied
 *   once, so it is not modified.
 * @returns The algorithm, which takes no input: run it with `run(alg, undefined, steps)`. Each state holds the
 *   working `R` and the accumulated `Q` after one more column.
 *
 * @example Householder QR, one reflection at a time
 * const A = tensor([[1, 1], [1, 0], [0, 1]])
 * const alg = householderSteps(A)
 * const first = run(alg, undefined, 1)
 * print('after one reflection, R =', first.R)
 * const last = run(alg, undefined, 10)
 * print('finished =', last.done)
 * print('R =', last.R)
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

/**
 * $\Xmat\Rmat^{-1}$ for upper-triangular $\Rmat$.
 *
 * @param x The matrix $\Xmat$, $p \times k$, whose rows are solved against $\Rmat$.
 * @param R The upper-triangular matrix $\Rmat$, $k \times k$ and invertible; only its upper triangle is read.
 * @returns $\Xmat\Rmat^{-1}$, with the shape of `x`.
 */
const rightSolve = (x: Value, R: Value) =>
  transpose(solveTriangular(R, transpose(x), { lower: false, transpose: true }))
/**
 * $\Xmat\Rmat^{-\top}$ for upper-triangular $\Rmat$.
 *
 * @param x The matrix $\Xmat$, $p \times k$, whose rows are solved against $\Rmat^\top$.
 * @param R The upper-triangular matrix $\Rmat$, $k \times k$ and invertible; only its upper triangle is read.
 * @returns $\Xmat\Rmat^{-\top}$, with the shape of `x`.
 */
const rightSolveTransposed = (x: Value, R: Value) => transpose(solveTriangular(R, transpose(x), { lower: false }))

/**
 * A rank-deficient $\Amat$ (a zero diagonal entry of $\Rmat$) has no derivative of $\Qmat$: report it rather than
 * divide by zero, example by example inside `vmap`.
 *
 * @param R The triangular factor $\Rmat$, $k \times n$ (one per example inside `vmap`). Its first $k$ diagonal
 *   entries are compared with $k\varepsilon$ times its largest entry; nothing is checked when its values are not
 *   known as numbers.
 * @param k The number of rows of $\Rmat$, $\min(m, n)$, which is the number of diagonal entries to check.
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

/**
 * The tangents $(\dot{\Qmat}, \dot{\Rmat})$ of a square or tall QR, $\Amat = \Qmat\Rmat$ with $\Rmat$
 * $k \times k$ invertible, for a tangent $\dot{\Amat}$.
 *
 * @param Q The factor $\Qmat$, $m \times k$ with orthonormal columns.
 * @param R The factor $\Rmat$, $k \times k$, upper triangular and invertible.
 * @param t The tangent $\dot{\Amat}$ of the factored matrix, $m \times k$.
 * @param k The number of columns of $\Amat$ (the size of $\Rmat$).
 * @returns The pair $\dot{\Qmat}$ ($m \times k$) and $\dot{\Rmat}$ ($k \times k$, upper triangular), in that order.
 */
function jvpTall(Q: Value, R: Value, t: Value, k: number): [Value, Value] {
  const C = matmul(transpose(Q), rightSolve(t, R))
  const low = mul(C, lowerMask(k, false))
  const omega = sub(low, transpose(low))
  const dR = matmul(sub(C, omega), R)
  const dQ = add(sub(rightSolve(t, R), matmul(Q, C)), matmul(Q, omega))
  return [dQ, dR]
}

/**
 * The adjoint $\bar{\Amat}$ of a square or tall QR, $\Amat = \Qmat\Rmat$ with $\Rmat$ $k \times k$ invertible, for
 * cotangents $(\bar{\Qmat}, \bar{\Rmat})$.
 *
 * @param Q The factor $\Qmat$, $m \times k$ with orthonormal columns.
 * @param R The factor $\Rmat$, $k \times k$, upper triangular and invertible.
 * @param gQ The cotangent $\bar{\Qmat}$ of $\Qmat$, $m \times k$.
 * @param gR The cotangent $\bar{\Rmat}$ of $\Rmat$, $k \times k$.
 * @param k The number of columns of $\Amat$ (the size of $\Rmat$).
 * @returns The cotangent $\bar{\Amat}$ of the factored matrix, $m \times k$.
 */
function vjpTall(Q: Value, R: Value, gQ: Value, gR: Value, k: number): Value {
  const M = sub(matmul(R, transpose(gR)), matmul(transpose(gQ), Q))
  return rightSolveTransposed(add(gQ, matmul(Q, symmetricFromLower(M, k))), R)
}

/**
 * The QR primitive, with its derivative rules (Seeger et al., 2017, "Auto-differentiating linear algebra",
 * arXiv:1710.08717; Walter and Lehmann, 2018; the wide case from Liao et al., 2019, "Differentiable programming tensor
 * networks", §B). For $\Amat = \Qmat\Rmat$ with the $k \times k$ $\Rmat$ invertible, $\Cmat = \Qmat^\top \dot{\Amat}
 * \Rmat^{-1}$ and $\Omegamat = \operatorname{tril}(\Cmat, -1) - \operatorname{tril}(\Cmat, -1)^\top$: the tangents are
 * $\dot{\Rmat} = (\Cmat - \Omegamat) \Rmat$ and $\dot{\Qmat} = \dot{\Amat} \Rmat^{-1} - \Qmat\Cmat + \Qmat\Omegamat$;
 * the adjoint is $\bar{\Amat} = (\bar{\Qmat} + \Qmat \operatorname{copyltu}(\Mmat)) \Rmat^{-\top}$ with $\Mmat = \Rmat
 * \bar{\Rmat}^\top - \bar{\Qmat}^\top \Qmat$ and copyltu the symmetric matrix of $\Mmat$'s lower triangle. A wide
 * $\Amat = [\Xmat \; \Ymat]$ ($\Xmat$ is $m \times m$) factors $\Xmat = \Qmat\Umat$ and gives $\Rmat = [\Umat \;
 * \Qmat^\top \Ymat]$: the square rule applies to $\Xmat$ with $\bar{\Qmat} + \Ymat \bar{\Rmat}_Y^\top$, and
 * $\bar{\Ymat} = \Qmat \bar{\Rmat}_Y$. The output is $\Qmat$ ($m \times k$) and $\Rmat$ ($k \times n$) packed into one
 * vector, for the reduced factorisation.
 */
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
 * QR factorisation $\Amat = \Qmat\Rmat$ of an $m \times n$ matrix by Householder reflections. `mode` `reduced`
 * (default) gives the thin factors; `complete` gives a square $\Qmat$. Differentiable in both modes for the reduced
 * factorisation of a matrix of full rank (Seeger et al., 2017); a rank-deficient $\Amat$ throws `NumericalError`
 * ('singular') when differentiated. The complete factorisation of a tall matrix is not differentiable (its extra
 * columns of $\Qmat$ are not unique) and refuses traced input.
 *
 * @param a The matrix $\Amat$ to factor, $m \times n$ with any of tall, square or wide; it is not modified. A traced
 *   value makes the factors differentiable.
 * @param options Which factorisation to return.
 * @param options.mode 'reduced' (the default) gives $\Qmat$ as $m \times k$ and $\Rmat$ as $k \times n$ with
 *   $k = \min(m, n)$; 'complete' gives a square $m \times m$ $\Qmat$ and an $m \times n$ $\Rmat$.
 * @returns The factors `Q` (orthonormal columns) and `R` (upper triangular; its diagonal may be negative).
 *
 * @example The reduced QR factorisation
 * const A = tensor([[1, 1], [1, 0], [0, 1]])
 * const { Q, R } = qr(A)
 * print('Q =', Q)
 * print('R =', R)
 * print('Q R =', matmul(Q, R))
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
