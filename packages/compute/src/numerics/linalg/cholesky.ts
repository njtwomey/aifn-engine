/**
 * Cholesky factorisation $\Amat = \Lmat\Lmat^\top$ with jitter reporting, and the solves and log-determinant built on
 * it.
 *
 * The factorisation is the column-by-column (Cholesky–Crout) algorithm of Golub and Van Loan (2013), Algorithm 4.2.2.
 * When a pivot is not positive, the smallest jitter $j = s \cdot 10^k$ ($s$ the mean diagonal) that lets $\Amat +
 * j\Imat$ factor is added and reported, in the manner of GPy's `jitchol` (Rasmussen and Williams, 2006, §A.4, recommend
 * adding a small multiple of I for numerical stability). The derivative rule is Murray (2016), "Differentiation of the
 * Cholesky decomposition", arXiv:1602.07527: eq. 10 (reverse) and §3.1 (forward).
 */

import {
  defineOp,
  diagonal,
  log,
  matmul,
  mul,
  shapeOfValue,
  sum,
  transpose,
  type NumberResult,
  type Op,
  type Tensor,
  type TensorResult,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { NumericalError } from 'aifn-compute/foundation/errors'
import { denseSquare, EPS, matrix, positiveDefinite } from './dense'
import { concrete, float64Aval, kernelBatch, lowerAdjoint, phiMask, symmetricFromLower } from './rules'
import { solveTriangular } from './triangular'

/** Options of `cholesky`. */
export type CholeskyOptions = {
  /**
   * `'auto'` (default): factor A as given, and if that fails add the smallest jitter from the ladder s·10⁻¹², …,
   * s·10⁻² (s = mean |diagonal|, or 1 if that is 0) that succeeds. A number: add exactly that jitter. `false`: never
   * add jitter.
   */
  jitter?: 'auto' | number | false
  /** Largest jitter tried by `'auto'`, relative to s (default 1e-2). */
  maxRelativeJitter?: number
}

/** The result of `cholesky`. */
export type Cholesky<L = Tensor> = {
  /** Lower-triangular factor with LLᵀ = A + jitter·I. Columns from the failed pivot on are zero when `failed`. */
  L: L
  /** The diagonal jitter that was added (0 when A factored as given; NaN inside `vmap`, where it is per example). */
  jitter: number
  /** True when no jitter allowed by the options made A + jitter·I positive definite. L then contains no NaN. */
  failed: boolean
  /** The column whose pivot was not positive in the last attempt, or −1. */
  failedAt: number
}

/**
 * Factor in place: returns the failing column or −1. A pivot fails when it is not above n·ε times the largest
 * diagonal entry, the threshold below which the factor is dominated by rounding (Higham, 2002, "Accuracy and Stability
 * of Numerical Algorithms", §10.1).
 */
function factor(a: Float64Array, n: number, jitter: number, out: Float64Array): number {
  let scale = 0
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(a[i * n + i] + jitter))
  const floor = n * EPS * scale
  out.fill(0)
  for (let j = 0; j < n; j++) {
    let d = a[j * n + j] + jitter
    for (let k = 0; k < j; k++) d -= out[j * n + k] * out[j * n + k]
    // Columns from j on stay zero, so the partial factor is NaN-free.
    if (!(d > floor)) return j
    const ljj = Math.sqrt(d)
    out[j * n + j] = ljj
    for (let i = j + 1; i < n; i++) {
      // Only the lower triangle of A is read.
      let s = a[i * n + j]
      for (let k = 0; k < j; k++) s -= out[i * n + k] * out[j * n + k]
      out[i * n + j] = s / ljj
    }
  }
  return -1
}

/**
 * Parameters of the `cholesky` primitive. `jitter` is the jitter added, or the search to run (`'auto'`, with
 * `maxRelativeJitter`) when the impl factors per example (inside `vmap`, where no single input is known in advance).
 * `factor` is the factor the wrapper's jitter search already found for this input (so the matrix is factored once),
 * with whether it failed. Without `factor` (the generated tests, `vmap`) the impl factors itself.
 */
type Params = {
  readonly jitter: number | 'auto' | false
  readonly maxRelativeJitter?: number
  readonly factor?: Float64Array
  readonly failed?: boolean
}

/** The jitter search: factor A + jI for the ladder of j until one succeeds. `work` holds the last factor tried. */
function search(
  data: Float64Array,
  n: number,
  jitter: number | 'auto' | false,
  maxRelativeJitter: number,
  work: Float64Array,
): { chosen: number; failedAt: number } {
  let chosen = typeof jitter === 'number' ? jitter : 0
  let failedAt = factor(data, n, chosen, work)
  if (failedAt >= 0 && jitter === 'auto') {
    let s = 0
    for (let i = 0; i < n; i++) s += Math.abs(data[i * n + i]) / n
    if (s === 0) s = 1
    for (let k = -12; failedAt >= 0 && 10 ** k <= maxRelativeJitter * (1 + 1e-12); k++) {
      chosen = s * 10 ** k
      failedAt = factor(data, n, chosen, work)
    }
  }
  return { chosen, failedAt }
}

/** A partial factor has no derivative: a rule would silently return a wrong one. */
function refuseFailed(failed: boolean | undefined): void {
  if (failed) {
    throw new NumericalError(
      'cholesky',
      'cholesky: the factorisation failed (the matrix is not positive definite), so it cannot be differentiated',
      'not-positive-definite',
    )
  }
}

// Rules: Murray (2016), "Differentiation of the Cholesky decomposition", arXiv:1602.07527. Reverse, eq. 10:
// Ā = Φ(S + Sᵀ) with S = L⁻ᵀ Φ(Lᵀ L̄) L⁻¹ (A is read through its lower triangle). Forward, §3.1:
// L̇ = L Φ(L⁻¹ Ȧ L⁻ᵀ), with Ȧ the symmetric matrix of the tangent's lower triangle. The jitter is a constant.
const choleskyOp: Op<Params> = defineOp<Params>(
  'numerics/linalg/cholesky',
  ([a], { jitter, maxRelativeJitter = 1e-2, factor: found }) => {
    if (found) {
      const n = Math.round(Math.sqrt(found.length))
      return matrix(found, n, n)
    }
    const { n, a: data } = denseSquare(a, 'cholesky')
    const out = new Float64Array(n * n)
    const { failedAt } = search(data, n, jitter, maxRelativeJitter, out)
    // Only a search (inside vmap) cannot report a failure per example, so it throws; a fixed jitter keeps the partial
    // factor, as the wrapper does.
    if (failedAt >= 0 && jitter === 'auto') refuseFailed(true)
    return matrix(out, n, n)
  },
  (g, _inputs, L, { failed }) => {
    refuseFailed(failed)
    const n = shapeOfValue(L)[0]
    const phi = phiMask(n)
    const inner = mul(matmul(transpose(L), g), phi)
    const left = solveTriangular(L, inner, { transpose: true })
    const S = transpose(solveTriangular(L, transpose(left), { transpose: true }))
    return [lowerAdjoint(S, n)]
  },
  {
    arity: 1,
    jvp: ([t], _inputs, L, { failed }) => {
      if (t === null) return null
      refuseFailed(failed)
      const n = shapeOfValue(L)[0]
      // L⁻¹ Ȧ L⁻ᵀ = L⁻¹ (L⁻¹ Ȧ)ᵀ, as Ȧ is symmetric.
      const half = solveTriangular(L, symmetricFromLower(t, n))
      const inner = solveTriangular(L, transpose(half))
      return matmul(L, mul(inner, phiMask(n)))
    },
    batch: kernelBatch('numerics/linalg/cholesky'),
    shape: ([a]) => float64Aval(a.shape),
    doc: {
      note: 'cholesky-decomposition',
      summary: 'The Cholesky factor L of a symmetric positive-definite matrix, LLᵀ = A.',
    },
    test: { secondOrder: true, cases: (draw) => [{ inputs: [positiveDefinite(draw, 3)], params: { jitter: 0 } }] },
  },
)

/**
 * Cholesky factor of a symmetric positive-definite matrix $\Amat$ ($n \times n$; only its lower triangle is read):
 * lower-triangular $\Lmat$ with $\Lmat\Lmat^\top = \Amat + \text{jitter} \cdot \Imat$. The matrix is factored once per
 * jitter tried, and the factor the search ends on is the result. Failure is reported, never hidden: `jitter` says what
 * was added, and `failed` that nothing allowed worked (L is then partial but NaN-free, and differentiating it throws
 * `NumericalError` 'not-positive-definite'). A non-finite entry throws `LinAlgError`. Differentiable in both modes
 * (Murray, 2016). Inside `vmap` each example is factored with its own jitter search, which is not reported (`jitter` is
 * NaN), and a failed example throws.
 *
 * @example Factor a positive-definite matrix
 * const A = tensor([[4, 1], [1, 3]])
 * const { L, jitter } = cholesky(A)
 * print('L =', L)
 * print('L Lᵀ =', matmul(L, transpose(L)))
 * print('jitter =', jitter)
 *
 * @example Failure is reported, not hidden
 * // Eigenvalues 3 and −1: not positive definite, and with no jitter allowed the factorisation stops at column 1.
 * const { failed, failedAt } = cholesky(tensor([[1, 2], [2, 1]]), { jitter: false })
 * print('failed =', failed)
 * print('failed at column', failedAt)
 *
 * @example Jitter rescues a matrix that is only just singular
 * // A kernel matrix of two identical points: jitter > 0 says it was not numerically positive definite.
 * const { jitter, failed } = cholesky(tensor([[1, 1], [1, 1]]))
 * print('jitter =', jitter)
 * print('failed =', failed)
 */
export function cholesky<X extends Value>(a: X, options: CholeskyOptions = {}): Cholesky<TensorResult<X>> {
  const { jitter = 'auto', maxRelativeJitter = 1e-2 } = options
  if (concrete(a) === null) {
    const L = choleskyOp([a], { jitter, maxRelativeJitter }) as TensorResult<X>
    return { L, jitter: NaN, failed: false, failedAt: -1 }
  }
  const { n, a: data } = denseSquare(a, 'cholesky')
  const work = new Float64Array(n * n)
  const { chosen, failedAt } = search(data, n, jitter, maxRelativeJitter, work)
  const failed = failedAt >= 0
  const L = choleskyOp([a], { jitter: chosen, factor: work, failed }) as TensorResult<X>
  return { L, jitter: chosen, failed, failedAt }
}

/**
 * Solve $\Amat\Xmat = \Bmat$ given $\Amat$'s Cholesky factor $\Lmat$ ($n \times n$) and $\Bmat$ ($n$ or $n \times r$),
 * by two triangular solves (differentiable).
 *
 * @example Factor once, then solve for several right-hand sides
 * const { L } = cholesky(tensor([[4, 1], [1, 3]]))
 * print('x1 =', choleskySolve(L, tensor([1, 2])))
 * print('x2 =', choleskySolve(L, tensor([0, 1])))
 *
 * @example A matrix of right-hand sides solves every column at once
 * const A = tensor([[4, 1], [1, 3]])
 * const { L } = cholesky(A)
 * const X = choleskySolve(L, tensor([[1, 0], [0, 1]]))
 * print('A⁻¹ =', X)
 * print('A A⁻¹ =', matmul(A, X))
 */
export function choleskySolve<L extends Value, B extends Value>(L: L, b: B): TensorResult<L | B> {
  const y = solveTriangular(L, b)
  return solveTriangular(L, y, { transpose: true }) as TensorResult<L | B>
}

/**
 * $\log\det \Amat = 2 \sum_i \log L_{ii}$ from $\Amat$'s Cholesky factor $\Lmat$.
 *
 * @example The log-determinant from the factor
 * const A = tensor([[4, 1], [1, 3]])
 * const { L } = cholesky(A)
 * print('log det A =', choleskyLogDet(L))
 * print('log(det A) =', Math.log(det(A)))
 *
 * @example It stays finite where the determinant overflows
 * // 400 on the diagonal of a 150×150 matrix: det A = 400¹⁵⁰ is beyond float64, its logarithm is not.
 * const big = mul(eye(150), 400)
 * print('det A =', det(big))
 * print('log det A =', choleskyLogDet(cholesky(big).L))
 */
export function choleskyLogDet<L extends Value>(L: L): NumberResult<L> {
  return mul(2, sum(log(diagonal(L)))) as NumberResult<L>
}
