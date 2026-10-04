/**
 * Symmetric eigendecomposition by the cyclic Jacobi method (Jacobi, 1846; Golub and Van Loan, 2013, Algorithm 8.5.3,
 * with Rutishauser's (1971) stable rotation formulas). Each rotation zeroes one off-diagonal pair; sweeps repeat until
 * the off-diagonal mass is negligible. Jacobi is slower than tridiagonal QR but simple and accurate: eigenvalues are
 * found to within about $\varepsilon \lVert \Amat \rVert$, and small ones of well-scaled matrices to high relative
 * accuracy (Demmel and Veselić, 1992).
 */

import {
  add,
  definePrimitive,
  diag,
  diagonal,
  isTraced,
  matmul,
  mul,
  type Op,
  shapeOfValue,
  type Tensor,
  type TensorResult,
  transpose,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { NumericalError } from 'aifn-compute/foundation/errors'
import { denseSquare, EPS, matrix, positiveDefinite, vector } from './dense'
import {
  concrete,
  concreteExamples,
  exampleOf,
  float64Aval,
  fMatrix,
  kernelBatch,
  lowerAdjoint,
  negligible,
  pack,
  packRaw,
  scaleOf,
  symmetricFromLower,
  unpack,
} from './rules'

/** The result of `eigh` (values and vectors traced for traced input). */
export type Eigh<T = Tensor> = {
  /** Eigenvalues in descending order. */
  values: T
  /**
   * Orthonormal eigenvectors as columns, column $j$ for `values[j]`. Each is signed so that its largest-magnitude
   * component (the first of equals) is positive.
   */
  vectors: T
  /** Number of Jacobi sweeps used (NaN inside `vmap`, where it is per example). */
  sweeps: number
  /** False when `maxSweeps` ran out before the off-diagonal mass fell below tolerance. */
  converged: boolean
}

/**
 * The Jacobi iteration on a dense copy of $\Amat$.
 *
 * @param a The symmetric matrix $\Amat$, $n \times n$. Only its lower triangle is read (the upper one is taken to be
 *   its mirror image), and it is copied, so it is not modified.
 * @param maxSweeps The most sweeps to run, a sweep being one pass of rotations over every off-diagonal pair.
 * @returns The eigenvalues `values` ($n$ values, descending) and eigenvectors `vectors` ($n \times n$, as columns in
 *   the same order), with the number of `sweeps` run and `converged` (false when `maxSweeps` ran out before the
 *   off-diagonal mass fell below tolerance).
 */
function jacobi(a: Value, maxSweeps: number): Eigh {
  const { n, a: A } = denseSquare(a, 'eigh')
  // Symmetrise from the lower triangle, as LAPACK's dsyevd with UPLO='L'.
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) A[i * n + j] = A[j * n + i]
  const V = new Float64Array(n * n)
  for (let i = 0; i < n; i++) V[i * n + i] = 1
  let total = 0
  for (let k = 0; k < n * n; k++) total += A[k] * A[k]
  const tolerance = (EPS * Math.sqrt(total)) ** 2
  let sweeps = 0
  let converged = false
  for (; sweeps <= maxSweeps; sweeps++) {
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q]
    if (off <= tolerance) {
      converged = true
      break
    }
    if (sweeps === maxSweeps) break
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q]
        if (apq === 0) continue
        const app = A[p * n + p]
        const aqq = A[q * n + q]
        // tan of the rotation angle, the smaller root of t² + 2θt − 1 = 0 (Rutishauser).
        const theta = (aqq - app) / (2 * apq)
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p]
          const akq = A[k * n + q]
          A[k * n + p] = c * akp - s * akq
          A[k * n + q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k]
          const aqk = A[q * n + k]
          A[p * n + k] = c * apk - s * aqk
          A[q * n + k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p]
          const vkq = V[k * n + q]
          V[k * n + p] = c * vkp - s * vkq
          V[k * n + q] = s * vkp + c * vkq
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => A[j * n + j] - A[i * n + i])
  const values = new Float64Array(n)
  const vectors = new Float64Array(n * n)
  order.forEach((src, col) => {
    values[col] = A[src * n + src]
    let big = 0
    for (let k = 0; k < n; k++) if (Math.abs(V[k * n + src]) > Math.abs(V[big * n + src])) big = k
    const flip = V[big * n + src] < 0 ? -1 : 1
    for (let k = 0; k < n; k++) vectors[k * n + col] = flip * V[k * n + src]
  })
  return { values: vector(values), vectors: matrix(vectors, n, n), sweeps, converged }
}

/** Parameters of the `eigh` primitive: the sweep limit, and the decomposition already found for this input. */
type Params = { readonly maxSweeps: number; readonly found?: Eigh }

/**
 * Differentiating a decomposition that did not converge would differentiate garbage: report it.
 *
 * @param p The primitive's parameters. Only `found` is read: the decomposition already computed for this input, whose
 *   `converged` flag is checked. Nothing is thrown when `found` is absent.
 */
function refuseUnconverged(p: Params): void {
  if (p.found && !p.found.converged) {
    throw new NumericalError(
      'eigh',
      'eigh: the Jacobi iteration did not converge, so the decomposition cannot be differentiated',
      'not-converged',
    )
  }
}

/**
 * The symmetric eigendecomposition primitive, with its derivative rules (Giles, 2008, "Collected matrix derivative
 * results", §3.1; Seeger et al., 2017, "Auto-differentiating linear algebra", arXiv:1710.08717). With $\dot{\Amat}$ the
 * symmetric matrix of the tangent's lower triangle and $\Pmat = \Vmat^\top \dot{\Amat} \Vmat$, the tangents are
 * $\dot{\lambdavec} = \diag(\Pmat)$ and $\dot{\Vmat} = \Vmat (\Fmat \circ \Pmat)$, and the adjoint is $\bar{\Amat} =
 * \Vmat (\diag(\bar{\lambdavec}) + \Fmat \circ (\Vmat^\top \bar{\Vmat})) \Vmat^\top$ read back through the lower
 * triangle, with $F_{ij} = 1 / (\lambda_j - \lambda_i)$. At repeated eigenvalues $\Fmat$ is undefined: the rules go
 * through only when the function is invariant within the degenerate subspace, and otherwise throw `NumericalError`
 * ('degenerate'). The output is $\lambdavec$ ($n$) and $\Vmat$ ($n \times n$) packed into one vector.
 */
const eighOp: Op<Params> = definePrimitive<Params>({
  id: 'numerics/linalg/eigh',
  arity: 1,
  impl: ([a], p) => {
    const e = p.found ?? jacobi(a, p.maxSweeps)
    if (!e.converged) refuseUnconverged({ ...p, found: e })
    return packRaw([e.values.data as Float64Array, e.vectors.data as Float64Array])
  },
  vjp: (g, [a], out, p) => {
    refuseUnconverged(p)
    const n = shapeOfValue(a)[0]
    const [values, V] = unpack(out, [[n], [n, n]])
    const [gValues, gV] = unpack(g, [[n], [n, n]])
    const M = matmul(transpose(V), gV)
    // Invariant within a degenerate pair (i, j) when the pair's λ̄ agree and M's antisymmetric part vanishes there.
    const ms = concreteExamples(M)
    const gls = concreteExamples(gValues)
    const invariant =
      ms && gls
        ? (i: number, j: number, b: number) => {
            const [m, gl] = [exampleOf(ms, b), exampleOf(gls, b)]
            if (!m || !gl) return false
            const scale = Math.max(scaleOf(m), scaleOf(gl))
            return negligible(m[i * n + j] - m[j * n + i], scale) && negligible(gl[i] - gl[j], scale)
          }
        : null
    const F = fMatrix(values, 'eigh', invariant)
    const G = matmul(matmul(V, add(diag(gValues), mul(F, M))), transpose(V))
    return [lowerAdjoint(G, n)]
  },
  jvp: ([t], [a], out, p) => {
    if (t === null) return null
    refuseUnconverged(p)
    const n = shapeOfValue(a)[0]
    const [values, V] = unpack(out, [[n], [n, n]])
    const P = matmul(matmul(transpose(V), symmetricFromLower(t, n)), V)
    const ps = concreteExamples(P)
    const invariant = ps
      ? (i: number, j: number, b: number) => {
          const pc = exampleOf(ps, b)
          return pc !== undefined && negligible(pc[i * n + j], scaleOf(pc))
        }
      : null
    const F = fMatrix(values, 'eigh', invariant)
    return pack([diagonal(P), matmul(V, mul(F, P))])
  },
  batch: kernelBatch('numerics/linalg/eigh'),
  shape: ([a]) => float64Aval([a.shape[0] + a.shape[0] * a.shape[0]]),
  doc: {
    note: 'eigendecomposition',
    summary: 'Eigenvalues (descending) and orthonormal eigenvectors of a symmetric matrix.',
  },
  test: { rtol: 1e-4, cases: (draw) => [{ inputs: [positiveDefinite(draw, 3)], params: { maxSweeps: 100 } }] },
})

/**
 * Eigendecomposition $\Amat = \Vmat \operatorname{diag}(\lambdavec) \Vmat^\top$ of a symmetric matrix (only its lower
 * triangle is read), with eigenvalues in descending order and eigenvectors as the columns of $\Vmat$. Differentiable
 * in both modes (Giles, 2008): at repeated eigenvalues only functions invariant within the degenerate subspace (a sum
 * of the repeated eigenvalues, a projector onto their subspace) have a derivative; others throw `NumericalError`
 * ('degenerate'), as does differentiating a decomposition that did not converge ('not-converged'). Inside `vmap`,
 * `sweeps` is NaN and an example that does not converge throws.
 *
 * @param a The symmetric matrix $\Amat$, $n \times n$. Only its lower triangle is read, so the upper one may hold
 *   anything; it is not modified. A traced value makes the decomposition differentiable.
 * @param options How long the Jacobi iteration may run.
 * @param options.maxSweeps The most Jacobi sweeps (passes of rotations over every off-diagonal pair) to run, 100 by
 *   default. When they run out, `converged` is false and the result is only approximate.
 * @returns The eigenvalues `values` ($\lambdavec$, descending) and the eigenvectors `vectors` ($\Vmat$, as columns),
 *   with the number of `sweeps` used and whether the iteration `converged`.
 *
 * @example Eigenvalues and eigenvectors of a symmetric matrix
 * const { values, vectors } = eigh(tensor([[2, 1], [1, 2]]))
 * print('values (descending) =', values)
 * print('vectors (as columns) =', vectors)
 *
 * @example Rebuild the matrix from its decomposition
 * const A = tensor([[2, 1], [1, 2]])
 * const { values, vectors } = eigh(A)
 * print(matmul(matmul(vectors, diag(values)), transpose(vectors)))
 */
export function eigh<X extends Value>(a: X, { maxSweeps = 100 }: { maxSweeps?: number } = {}): Eigh<TensorResult<X>> {
  if (!isTraced(a)) return jacobi(a, maxSweeps) as Eigh<TensorResult<X>>
  const found = concrete(a) === null ? undefined : jacobi(a, maxSweeps)
  const n = shapeOfValue(a)[0]
  const [values, vectors] = unpack(eighOp([a], { maxSweeps, found }), [[n], [n, n]])
  return {
    values: values as TensorResult<X>,
    vectors: vectors as TensorResult<X>,
    sweeps: found?.sweeps ?? NaN,
    converged: found?.converged ?? true,
  }
}
