/**
 * The matrix exponential $e^{\Amat}$ by Padé approximation with scaling and squaring (Higham, 2005), as used for linear
 * systems $\xvec' = \Amat\xvec$ ($\xvec(t) = e^{\Amat t} \xvec_0$) and the discretisation of continuous-time models.
 */

import {
  concat,
  definePrimitive,
  dense,
  fromData,
  isTraced,
  type Matrix,
  type MatrixLike,
  type Op,
  shapeOfValue,
  slice,
  type Tensor,
  toFlat,
  type Traced,
  transpose,
  type Value,
  zeros,
} from 'aifn-compute/foundation/tensor'
import { NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { solve } from './lu'
import { concrete, float64Aval, kernelBatch } from './rules'

type F64 = dense.F64
const { identity, matMul, norm1, toMatrixF64 } = dense

// Padé coefficients b_k of the [m/m] approximant to eˣ and the largest 1-norm θ_m for which degree m is accurate to
// double precision (Higham, 2005, "The scaling and squaring method for the matrix exponential revisited", SIAM J.
// Matrix Anal. Appl. 26(4), Table 2.3 and Algorithm 2.3).
const PADE: Record<number, number[]> = {
  3: [120, 60, 12, 1],
  5: [30240, 15120, 3360, 420, 30, 1],
  7: [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1],
  9: [17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1],
  13: [
    64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600,
    670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
  ],
}
const THETA: [number, number][] = [
  [3, 1.495585217958292e-2],
  [5, 2.53939833006323e-1],
  [7, 9.504178996162932e-1],
  [9, 2.097847961257068],
]
const THETA13 = 5.371920351148152

/**
 * `out += alpha * x`, in place, on flat arrays.
 *
 * @param out The array that is added to; modified in place.
 * @param alpha The scalar that multiplies each entry of `x` before it is added.
 * @param x The array to add, at least as long as `out`; read, not modified.
 */
const addScaled = (out: F64, alpha: number, x: F64) => {
  for (let i = 0; i < out.length; i++) out[i] += alpha * x[i]
}

/**
 * $\Umat$ and $\Vmat$ of the $[m/m]$ Padé approximant $r_m(\Amat) = (\Vmat - \Umat)^{-1}(\Vmat + \Umat)$, from the even
 * powers $\Amat^2, \Amat^4, \dots$.
 *
 * @param A The (already scaled) matrix $\Amat$ as a row-major array of $n^2$ values; read, not modified.
 * @param n The number of rows (and columns) of $\Amat$.
 * @param m The Padé degree: 3, 5, 7, 9 or 13 (the degrees with coefficients in `PADE`).
 * @param powers The even powers of $\Amat$, each a row-major array of $n^2$ values: entry $k - 1$ holds
 *   $\Amat^{2k}$. Degrees below 13 read the powers up to $\Amat^{m-1}$; degree 13 reads $\Amat^2$, $\Amat^4$ and
 *   $\Amat^6$ only. Not modified.
 * @returns `U`, the odd part of the approximant's numerator (the terms with odd powers of $\Amat$), and `V`, the even
 *   part, each as a new row-major array of $n^2$ values.
 */
function padeUV(A: F64, n: number, m: number, powers: F64[]): { U: F64; V: F64 } {
  const b = PADE[m]
  const I = identity(n)
  if (m < 13) {
    // U = A Σ b_{2k+1} A^{2k}, V = Σ b_{2k} A^{2k}.
    const u = new Float64Array(n * n)
    const V = new Float64Array(n * n)
    addScaled(u, b[1], I)
    addScaled(V, b[0], I)
    for (let k = 1; 2 * k <= m; k++) {
      addScaled(u, b[2 * k + 1], powers[k - 1])
      addScaled(V, b[2 * k], powers[k - 1])
    }
    return { U: matMul(A, u, n, n, n), V }
  }
  const [A2, A4, A6] = powers
  const inner = new Float64Array(n * n)
  addScaled(inner, b[13], A6)
  addScaled(inner, b[11], A4)
  addScaled(inner, b[9], A2)
  const u = matMul(A6, inner, n, n, n)
  addScaled(u, b[7], A6)
  addScaled(u, b[5], A4)
  addScaled(u, b[3], A2)
  addScaled(u, b[1], I)
  const innerV = new Float64Array(n * n)
  addScaled(innerV, b[12], A6)
  addScaled(innerV, b[10], A4)
  addScaled(innerV, b[8], A2)
  const V = matMul(A6, innerV, n, n, n)
  addScaled(V, b[6], A6)
  addScaled(V, b[4], A4)
  addScaled(V, b[2], A2)
  addScaled(V, b[0], I)
  return { U: matMul(A, u, n, n, n), V }
}

/** The result of `expm`: $e^{\Amat}$ and how it was computed. */
export type MatrixExponential<T = Matrix> = {
  /** $e^{\Amat}$ ($n \times n$). */
  value: T
  /** The Padé degree used (3, 5, 7, 9 or 13; NaN inside `vmap`, where it is per example). */
  degree: number
  /** The number of squarings $s$: $e^{\Amat} = (r_m(\Amat/2^s))^{2^s}$ (NaN inside `vmap`). */
  squarings: number
}

/**
 * $e^{\Amat}$ of a dense row-major $n \times n$ matrix by scaling and squaring.
 *
 * @param A0 The matrix $\Amat$ as a row-major array of $n^2$ values; read, not modified. A non-finite entry throws
 *   `NumericalError` ('not-finite').
 * @param n The number of rows (and columns) of $\Amat$; 0 gives an empty $0 \times 0$ result.
 * @returns `value`, $e^{\Amat}$ as an $n \times n$ tensor, with the Padé `degree` used and the number of `squarings`.
 */
function compute(A0: F64, n: number): MatrixExponential {
  for (let i = 0; i < A0.length; i++)
    if (!Number.isFinite(A0[i])) throw new NumericalError('expm', 'expm: the matrix must be finite', 'not-finite')
  if (n === 0) return { value: fromData(new Float64Array(0), [0, 0]), degree: 3, squarings: 0 }
  const norm = norm1(A0, n)
  const A2 = matMul(A0, A0, n, n, n)
  for (const [m, theta] of THETA) {
    if (norm <= theta) {
      const powers = [A2]
      for (let k = 2; 2 * k <= m; k++) powers.push(matMul(powers[k - 2], A2, n, n, n))
      return { value: finish(A0, n, m, powers, 0), degree: m, squarings: 0 }
    }
  }
  const s = Math.max(0, Math.ceil(Math.log2(norm / THETA13)))
  const f = 2 ** -s
  const A = A0.map((v) => v * f)
  const B2 = A2.map((v) => v * f * f)
  const B4 = matMul(B2, B2, n, n, n)
  const B6 = matMul(B4, B2, n, n, n)
  return { value: finish(A, n, 13, [B2, B4, B6], s), degree: 13, squarings: s }
}

/**
 * A raw square matrix as dense data.
 *
 * @param a The matrix, as a tensor or nested arrays; a matrix that is not square throws `ShapeError`.
 * @param where The caller's name, used in error messages.
 * @returns `A`, the entries as a row-major array of $n^2$ values, and `n`, the number of rows (and columns).
 */
function squareData(a: MatrixLike, where: string): { A: F64; n: number } {
  const { data, m: rows, n } = toMatrixF64(a, where)
  if (rows !== n) throw new ShapeError(where, `${where}: expected a square matrix, got ${rows}×${n}`)
  return { A: data, n }
}

/**
 * The Fréchet derivative $L(\Amat, \Emat) = \frac{d}{dt} e^{\Amat + t\Emat}$ at $t = 0$, as the top-right block of
 * $\exp \begin{bmatrix} \Amat & \Emat \\ 0 & \Amat \end{bmatrix}$ (Van Loan, 1978, "Computing integrals involving the
 * matrix exponential", Theorem 1; Al-Mohy and Higham, 2009, "Computing the Fréchet derivative of the matrix
 * exponential", §1). Written with the `expm` primitive, so it differentiates again.
 *
 * @param a The matrix $\Amat$ ($n \times n$) at which the derivative is taken; may be traced.
 * @param e The direction $\Emat$ ($n \times n$) in which $\Amat$ is perturbed: a tangent in forward mode, the output's
 *   adjoint in reverse mode. May be traced.
 * @param n The number of rows (and columns) of $\Amat$ and $\Emat$.
 * @returns $L(\Amat, \Emat)$ as an $n \times n$ matrix, traced when an input is.
 */
function frechet(a: Value, e: Value, n: number): Value {
  const zero = zeros([n, n])
  const block = concat([concat([a, e], 1), concat([zero, a], 1)], 0)
  return slice(expmOp([block], {}), [0, n], [n, 2 * n])
}

/** Parameters of the `expm` primitive: $e^{\Amat}$ already computed for this input by the wrapper. */
type Params = { readonly found?: Tensor }

/**
 * The matrix exponential primitive, with its derivative rules. The tangent is the Fréchet derivative $L(\Amat,
 * \dot{\Amat})$ and, since $L(\Amat, \cdot)^\top = L(\Amat^\top, \cdot)$ for real $\Amat$, the adjoint is $\bar{\Amat}
 * = L(\Amat^\top, \bar{\Ymat})$ (Al-Mohy and Higham, 2009, §1; Najfeld and Havel, 1995).
 */
const expmOp: Op<Params> = definePrimitive<Params>({
  id: 'numerics/linalg/expm',
  arity: 1,
  impl: ([a], p) => {
    if (p.found) return p.found
    const { A, n } = squareData(a as Tensor, 'expm')
    return compute(A, n).value
  },
  vjp: (g, [a]) => [frechet(transpose(a), g, shapeOfValue(a)[0])],
  jvp: ([t], [a]) => (t === null ? null : frechet(a, t, shapeOfValue(a)[0])),
  batch: kernelBatch('numerics/linalg/expm'),
  shape: ([a]) => float64Aval(a.shape),
  doc: { note: 'linear-systems-and-the-matrix-exponential', summary: 'The matrix exponential e^A.' },
  test: { rtol: 1e-4, secondOrder: true, cases: (draw) => [{ inputs: [draw([3, 3])], params: {} }] },
})

/**
 * The matrix exponential $e^{\Amat} = \sum_k \Amat^k/k!$ of a real square matrix by the scaling and squaring method
 * with Padé approximants (Higham, 2005, Algorithm 2.3, as scipy's `expm`): pick the smallest Padé degree $m$ whose
 * accuracy bound $\theta_m$ covers $\lVert \Amat \rVert_1$, or scale $\Amat$ by $2^{-s}$ until
 * $\lVert \Amat/2^s \rVert_1 \le \theta_{13}$ and use degree 13, solve
 * $(\Vmat - \Umat)\Rmat = \Vmat + \Umat$, then square $\Rmat$ $s$ times. Accurate to near machine precision relative
 * to $\lVert e^{\Amat} \rVert$ for normal matrices. Differentiable in both modes through the Fréchet derivative (Van
 * Loan's block matrix, which costs one exponential of a $2n \times 2n$ matrix).
 *
 * @param a The real square matrix $\Amat$ ($n \times n$), as a tensor or nested arrays; every entry must be finite. A
 *   traced value makes the result differentiable.
 * @returns `value`, $e^{\Amat}$ ($n \times n$, traced when `a` is), with the Padé `degree` used and the number of
 *   `squarings` (both NaN inside `vmap`, where they differ per example).
 *
 * @example The exponential of a rotation generator is a rotation
 * // exp of [[0, −θ], [θ, 0]] is the rotation by θ; here θ = π/2.
 * const { value } = expm(tensor([[0, -Math.PI / 2], [Math.PI / 2, 0]]))
 * print(value)
 *
 * @example How the result was computed
 * const { degree, squarings } = expm(tensor([[1, 2], [3, 4]]))
 * print('Padé degree =', degree)
 * print('squarings =', squarings)
 */
export function expm(a: MatrixLike): MatrixExponential
export function expm(a: Traced): MatrixExponential<Traced>
export function expm(a: MatrixLike | Traced): MatrixExponential<Matrix | Traced>
export function expm(a: MatrixLike | Traced): MatrixExponential<Matrix | Traced> {
  if (!isTraced(a)) {
    const { A, n } = squareData(a, 'expm')
    return compute(A, n)
  }
  const raw = concrete(a)
  if (raw === null) return { value: expmOp([a], {}) as Traced, degree: NaN, squarings: NaN }
  const { A, n } = squareData(raw as Tensor, 'expm')
  const r = compute(A, n)
  return { ...r, value: expmOp([a], { found: r.value }) as Traced }
}

/**
 * The exponential from the degree-`m` Padé approximant of the scaled matrix: solve
 * $(\Vmat - \Umat)\Rmat = \Vmat + \Umat$, then square $\Rmat$ `squarings` times to undo the scaling.
 *
 * @param A The scaled matrix $\Amat/2^s$ as a row-major array of $n^2$ values; read, not modified.
 * @param n The number of rows (and columns) of the matrix.
 * @param m The Padé degree: 3, 5, 7, 9 or 13.
 * @param powers The even powers of the scaled matrix that degree `m` needs, each a row-major array of $n^2$ values:
 *   entry $k - 1$ holds the power $2k$. Not modified.
 * @param squarings The number $s$ of times the approximant is squared, matching the scaling by $2^{-s}$ already
 *   applied to `A` and `powers` (0 for none).
 * @returns $e^{\Amat}$ of the unscaled matrix as an $n \times n$ tensor.
 */
function finish(A: F64, n: number, m: number, powers: F64[], squarings: number): Matrix {
  const { U, V } = padeUV(A, n, m, powers)
  const P = new Float64Array(n * n)
  const Q = new Float64Array(n * n)
  for (let i = 0; i < n * n; i++) {
    P[i] = V[i] + U[i]
    Q[i] = V[i] - U[i]
  }
  let R = Float64Array.from(toFlat(solve(fromData(Q, [n, n]), fromData(P, [n, n])) as Tensor))
  for (let k = 0; k < squarings; k++) R = matMul(R, R, n, n, n)
  return fromData(R, [n, n])
}
