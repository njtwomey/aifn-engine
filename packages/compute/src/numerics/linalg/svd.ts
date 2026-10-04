/**
 * Thin singular value decomposition by one-sided Jacobi (Hestenes, 1958; Demmel and Veselić, 1992, "Jacobi's method
 * is more accurate than QR", SIAM J. Matrix Anal. Appl. 13(4)): plane rotations orthogonalise the columns of $\Amat$
 * in place; their norms are then the singular values. Small singular values come out with high relative accuracy.
 * `pinv`, `lstsq` and `conditionNumber` are built on it.
 */

import {
  add,
  definePrimitive,
  diag,
  diagonal,
  div,
  greater,
  isTraced,
  matmul,
  mul,
  type Op,
  reshape,
  shapeOfValue,
  square,
  sub,
  type Tensor,
  type TensorResult,
  transpose,
  type Value,
  where,
} from 'aifn-compute/foundation/tensor'
import { NotDifferentiableError, NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { dense, EPS, matrix, vector } from './dense'
import {
  concrete,
  concreteExamples,
  exampleOf,
  float64Aval,
  fMatrix,
  kernelBatch,
  negligible,
  pack,
  packRaw,
  scaleOf,
  unpack,
} from './rules'

/** The result of `svd` (U, S and V traced for traced input). */
export type SVD<T = Tensor> = {
  /** Left singular vectors as columns, $m \times k$ with $k = \min(m, n)$. */
  U: T
  /** Singular values $\svec$ in descending order, length $k$. */
  S: T
  /**
   * Right singular vectors as columns, $n \times k$ (so $\Amat = \Umat \operatorname{diag}(\svec) \Vmat^\top$; NumPy
   * returns $\Vmat^\top$ as `vh`). Each pair $(\uvec_j, \vvec_j)$ is signed so that the largest-magnitude component of
   * $\vvec_j$ (the first of equals) is positive.
   */
  V: T
  /** Number of Jacobi sweeps used (NaN inside `vmap`). */
  sweeps: number
  /** False when `maxSweeps` ran out before every pair of columns was orthogonal to working precision. */
  converged: boolean
}

/** The working result of the one-sided Jacobi iteration on a tall matrix, on flat row-major arrays. */
type Jacobi = {
  /** The left singular vectors as columns, $m \times n$. */
  U: Float64Array
  /** The $n$ singular values. */
  S: Float64Array
  /** The right singular vectors as columns, $n \times n$. */
  V: Float64Array
  /** The number of rows $m$. */
  m: number
  /** The number of columns $n$. */
  n: number
  /** The number of sweeps (passes over every pair of columns) that rotated something. */
  sweeps: number
  /** False when the sweep limit ran out before a sweep left every pair of columns alone. */
  converged: boolean
}

/** The smallest normal double: a squared column norm below it has lost its precision to underflow. */
const SUBNORMAL_SQUARE = 2.2250738585072014e-308

/**
 * One-sided Jacobi on an $m \times n$ matrix with $m \ge n$ (row-major in `a`, overwritten).
 *
 * @param a The matrix $\Amat$ as a row-major array of $mn$ values. Rotated in place: on return its columns are mutually
 *   orthogonal but not normalised (column $j$ is $s_j \uvec_j$), and the same array is returned as `U`.
 * @param m The number of rows of $\Amat$.
 * @param n The number of columns of $\Amat$, at most $m$.
 * @param maxSweeps The most sweeps to run, a sweep being one pass of rotations over every pair of columns.
 * @returns `U` (the rotated `a`), `S` (the $n$ column norms, the singular values in no particular order), `V` (the
 *   $n \times n$ product of the rotations, row-major), `m` and `n` as given, `sweeps` (the number of sweeps that
 *   rotated something) and `converged` (false when `maxSweeps` ran out first).
 */
function jacobi(a: Float64Array, m: number, n: number, maxSweeps: number): Jacobi {
  const V = new Float64Array(n * n)
  for (let i = 0; i < n; i++) V[i * n + i] = 1
  let sweeps = 0
  let converged = false
  for (; sweeps < maxSweeps; sweeps++) {
    let rotated = false
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0
        let beta = 0
        let gamma = 0
        for (let i = 0; i < m; i++) {
          const x = a[i * n + p]
          const y = a[i * n + q]
          alpha += x * x
          beta += y * y
          gamma += x * y
        }
        // A column whose squared norm is subnormal (norm below about 1.5e-154, as a rank-deficient input's null column
        // becomes) cannot have its orthogonality measured, and rotating it never converges: treat it as done. √α·√β
        // rather than √(αβ), which underflows sooner.
        if (gamma === 0 || Math.min(alpha, beta) < SUBNORMAL_SQUARE) continue
        if (Math.abs(gamma) <= EPS * Math.sqrt(alpha) * Math.sqrt(beta)) continue
        rotated = true
        const zeta = (beta - alpha) / (2 * gamma)
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = c * t
        for (let i = 0; i < m; i++) {
          const x = a[i * n + p]
          const y = a[i * n + q]
          a[i * n + p] = c * x - s * y
          a[i * n + q] = s * x + c * y
        }
        for (let i = 0; i < n; i++) {
          const x = V[i * n + p]
          const y = V[i * n + q]
          V[i * n + p] = c * x - s * y
          V[i * n + q] = s * x + c * y
        }
      }
    }
    if (!rotated) {
      converged = true
      break
    }
  }
  const S = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    // Scaled by the column's largest entry, so a singular value near 1e-300 does not underflow to 0 when squared.
    let big = 0
    for (let i = 0; i < m; i++) big = Math.max(big, Math.abs(a[i * n + j]))
    let s = 0
    if (big > 0) for (let i = 0; i < m; i++) s += (a[i * n + j] / big) ** 2
    S[j] = big * Math.sqrt(s)
  }
  return { U: a, S, V, m, n, sweeps, converged }
}

/**
 * Normalise $\Umat$'s columns and complete those of zero singular values to an orthonormal set, sort by descending
 * singular value, and fix signs. Returns $\Umat$ ($m \times n$), $\svec$ ($n$), $\Vmat$ ($n \times n$).
 *
 * @param options The result of `jacobi`; none of its arrays is modified.
 * @param options.U The rotated matrix, row-major $m \times n$: orthogonal columns, each a left singular vector times
 *   its singular value.
 * @param options.S The $n$ singular values (the norms of `U`'s columns), in the order of the columns.
 * @param options.V The right singular vectors as the columns of a row-major $n \times n$ array, in the same order.
 * @param options.m The number of rows of `U`.
 * @param options.n The number of columns of `U` (and the size of `V`).
 * @returns New arrays: `U` (row-major $m \times n$, orthonormal columns), `S` ($n$ values, descending) and `V`
 *   (row-major $n \times n$), each pair of vectors signed so that the largest-magnitude component of the column of
 *   `V` (the first of equals) is positive.
 */
function finish({ U, S, V, m, n }: Jacobi): { U: Float64Array; S: Float64Array; V: Float64Array } {
  const order = Array.from({ length: n }, (_, j) => j).sort((i, j) => S[j] - S[i])
  const u = new Float64Array(m * n)
  const v = new Float64Array(n * n)
  const s = new Float64Array(n)
  const smax = S[order[0]] ?? 0
  order.forEach((src, col) => {
    s[col] = S[src]
    for (let i = 0; i < n; i++) v[i * n + col] = V[i * n + src]
    if (S[src] > 0 && S[src] > smax * EPS * EPS) for (let i = 0; i < m; i++) u[i * n + col] = U[i * n + src] / S[src]
    else fillOrthogonal(u, m, n, col)
    let big = 0
    for (let i = 0; i < n; i++) if (Math.abs(v[i * n + col]) > Math.abs(v[big * n + col])) big = i
    if (v[big * n + col] < 0) {
      for (let i = 0; i < n; i++) v[i * n + col] = -v[i * n + col]
      for (let i = 0; i < m; i++) u[i * n + col] = -u[i * n + col]
    }
  })
  return { U: u, S: s, V: v }
}

/**
 * Set column `col` of `u` ($m \times n$) to a unit vector orthogonal to columns $0, \dots, \text{col} - 1$, by
 * Gram–Schmidt on basis vectors.
 *
 * @param u The matrix as a row-major array of $mn$ values. Columns before `col` are read and must already be
 *   orthonormal; column `col` is overwritten; later columns are untouched.
 * @param m The number of rows of `u`.
 * @param n The number of columns of `u`.
 * @param col The index (from 0) of the column to fill.
 */
function fillOrthogonal(u: Float64Array, m: number, n: number, col: number): void {
  let best = new Float64Array(m)
  let bestNorm = -1
  for (let e = 0; e < m; e++) {
    const w = new Float64Array(m)
    w[e] = 1
    for (let pass = 0; pass < 2; pass++) {
      for (let c = 0; c < col; c++) {
        let d = 0
        for (let i = 0; i < m; i++) d += u[i * n + c] * w[i]
        for (let i = 0; i < m; i++) w[i] -= d * u[i * n + c]
      }
    }
    const norm = Math.hypot(...w)
    if (norm > bestNorm) {
      best = w
      bestNorm = norm
    }
    if (norm > 0.5) break
  }
  for (let i = 0; i < m; i++) u[i * n + col] = best[i] / bestNorm
}

/**
 * The SVD of a tall ($m \ge n$, row-major in `data`, overwritten) matrix, with each pair of singular vectors signed so
 * that the largest-magnitude component of $\vvec_j$ (`signOn` 'V') or of $\uvec_j$ ('U') is positive.
 *
 * @param data The matrix as a row-major array of $mn$ values; used as working space and overwritten.
 * @param m The number of rows of the matrix.
 * @param n The number of columns of the matrix, at most $m$.
 * @param maxSweeps The most Jacobi sweeps (passes over every pair of columns) to run before giving up.
 * @param signOn Which vector of each pair fixes the sign: 'V' makes the largest-magnitude component of the right
 *   vector positive, 'U' that of the left vector.
 * @returns The decomposition as tensors: `U` ($m \times n$), `S` ($n$ values, descending) and `V` ($n \times n$), with
 *   the number of `sweeps` used and whether the iteration `converged`.
 */
function tall(data: Float64Array, m: number, n: number, maxSweeps: number, signOn: 'U' | 'V'): SVD {
  const raw = jacobi(data, m, n, maxSweeps)
  const { U, S, V } = finish(raw)
  if (signOn === 'U') {
    for (let col = 0; col < n; col++) {
      let big = 0
      for (let i = 0; i < m; i++) if (Math.abs(U[i * n + col]) > Math.abs(U[big * n + col])) big = i
      if (U[big * n + col] < 0) {
        for (let i = 0; i < m; i++) U[i * n + col] = -U[i * n + col]
        for (let i = 0; i < n; i++) V[i * n + col] = -V[i * n + col]
      }
    }
  }
  return { U: matrix(U, m, n), S: vector(S), V: matrix(V, n, n), sweeps: raw.sweeps, converged: raw.converged }
}

/** Parameters of the `svd` primitive (tall input): the sweep limit, the sign convention, and the SVD already found. */
type Params = { readonly maxSweeps: number; readonly signOn: 'U' | 'V'; readonly found?: SVD }

/**
 * Throws `NumericalError` when the Jacobi iteration did not converge: such a decomposition has no derivative.
 *
 * @param p The primitive's parameters. Only `found` is read: the decomposition already computed for this input, whose
 *   `converged` flag is checked. Nothing is thrown when `found` is absent.
 */
function refuseUnconverged(p: Params): void {
  if (p.found && !p.found.converged) {
    throw new NumericalError(
      'svd',
      'svd: the Jacobi iteration did not converge, so the decomposition cannot be differentiated',
      'not-converged',
    )
  }
}

/**
 * Refuse the projection term of a tall SVD at a zero singular value whose left vector's cotangent is used
 * (`used(j, b)`, every one by default), example by example inside `vmap`.
 *
 * @param examples The singular values, one array of $n$ values per batch example (a single one outside `vmap`), or
 *   null when they are not known as numbers, in which case nothing is checked. Read only.
 * @param m The number of rows of the decomposed matrix. Nothing is checked when $m = n$, where there is no projection
 *   term.
 * @param n The number of columns of the decomposed matrix, which is also the number of singular values.
 * @param used Says whether the derivative needs the left singular vector of singular value `j` (index from 0) of
 *   batch example `b`; a zero singular value throws only where it returns true. By default every one is needed.
 */
function refuseZeroSingular(
  examples: readonly ArrayLike<number>[] | null,
  m: number,
  n: number,
  used: (j: number, b: number) => boolean = () => true,
): void {
  if (examples === null || m === n) return
  examples.forEach((s, b) => {
    const smax = scaleOf(s)
    for (let j = 0; j < n; j++) {
      if (s[j] <= n * EPS * smax && used(j, b)) {
        const which = examples.length > 1 ? ` of batch example ${b}` : ''
        throw new NumericalError(
          'svd',
          `svd: singular value ${j}${which} is zero, so its left singular vector has no derivative`,
          'degenerate',
        )
      }
    }
  })
}

/**
 * The SVD primitive, with its derivative rules for the thin SVD $\Amat = \Umat \diag(\svec) \Vmat^\top$ of a tall $m
 * \times n$ matrix (Townsend, 2016, "Differentiating the singular value decomposition"; Seeger et al., 2017,
 * arXiv:1710.08717). With $F_{ij} = 1 / (s_j^2 - s_i^2)$, $\Smat = \diag(\svec)$ and $\Pmat = \Umat^\top \dot{\Amat}
 * \Vmat$, the tangents are $\dot{\svec} = \diag(\Pmat)$, $\dot{\Umat} = \Umat (\Fmat \circ (\Pmat\Smat +
 * \Smat\Pmat^\top)) + (\Imat - \Umat\Umat^\top) \dot{\Amat} \Vmat \Smat^{-1}$ and $\dot{\Vmat} = \Vmat (\Fmat \circ
 * (\Smat\Pmat + \Pmat^\top\Smat))$ ($\Vmat$ is square); the adjoint is $\bar{\Amat} = \Umat [\diag(\bar{\svec}) +
 * (\Fmat \circ (\Umat^\top \bar{\Umat} - \bar{\Umat}^\top \Umat)) \Smat + \Smat (\Fmat \circ (\Vmat^\top \bar{\Vmat} -
 * \bar{\Vmat}^\top \Vmat))] \Vmat^\top + (\Imat - \Umat\Umat^\top) \bar{\Umat} \Smat^{-1} \Vmat^\top$. Repeated
 * singular values are handled as `eigh`'s repeated eigenvalues. A wide matrix is decomposed through its transpose. The
 * output is $\Umat$ ($m \times n$), $\svec$ ($n$) and $\Vmat$ ($n \times n$) packed into one vector.
 */
const svdOp: Op<Params> = definePrimitive<Params>({
  id: 'numerics/linalg/svd',
  arity: 1,
  impl: ([a], p) => {
    const f = p.found ?? decomposeTall(a, p)
    if (!f.converged) refuseUnconverged({ ...p, found: f })
    return packRaw([f.U.data as Float64Array, f.S.data as Float64Array, f.V.data as Float64Array])
  },
  vjp: (g, [a], out, p) => {
    refuseUnconverged(p)
    const [m, n] = shapeOfValue(a)
    const shapes = [[m, n], [n], [n, n]]
    const [U, s, V] = unpack(out, shapes)
    const [gU, gs, gV] = unpack(g, shapes)
    const S = diag(s)
    const UtgU = matmul(transpose(U), gU)
    const VtgV = matmul(transpose(V), gV)
    const J = sub(UtgU, transpose(UtgU))
    const K = sub(VtgV, transpose(VtgV))
    const js = concreteExamples(J)
    const ks = concreteExamples(K)
    const gss = concreteExamples(gs)
    const ss = concreteExamples(s)
    const invariant =
      js && ks && gss && ss
        ? (i: number, j: number, b: number) => {
            const [jc, kc, gc, sc] = [exampleOf(js, b), exampleOf(ks, b), exampleOf(gss, b), exampleOf(ss, b)]
            if (!jc || !kc || !gc || !sc) return false
            const scale = Math.max(scaleOf(jc), scaleOf(kc), scaleOf(gc))
            return (
              negligible((jc[i * n + j] + kc[i * n + j]) * sc[j], scale * scaleOf(sc)) &&
              negligible(gc[i] - gc[j], scale)
            )
          }
        : null
    const F = fMatrix(square(s), 'svd', invariant)
    const inner = add(diag(gs), add(matmul(mul(F, J), S), matmul(S, mul(F, K))))
    let ga: Value = matmul(matmul(U, inner), transpose(V))
    if (m > n) {
      const gUs = concreteExamples(gU)
      refuseZeroSingular(ss, m, n, (j, b) => {
        const gUc = gUs && exampleOf(gUs, b)
        return !gUc || Array.from({ length: m }, (_, i) => gUc[i * n + j]).some((v) => v !== 0)
      })
      const projected = sub(gU, matmul(U, UtgU))
      // A zero singular value passes the check above only when its column of Ū is zero, so its projected column is
      // exactly zero: divide it by 1 rather than 0 (0 · ∞ would be NaN).
      const nonzero = greater(s, 0)
      const inverse = where(nonzero, div(1, where(nonzero, s, 1)), 0)
      ga = add(ga, matmul(mul(projected, reshape(inverse, [1, n])), transpose(V)))
    }
    return [ga]
  },
  jvp: ([t], [a], out, p) => {
    if (t === null) return null
    refuseUnconverged(p)
    const [m, n] = shapeOfValue(a)
    const [U, s, V] = unpack(out, [[m, n], [n], [n, n]])
    const S = diag(s)
    const P = matmul(matmul(transpose(U), t), V)
    const ps = concreteExamples(P)
    const invariant = ps
      ? (i: number, j: number, b: number) => {
          const pc = exampleOf(ps, b)
          return pc !== undefined && negligible(pc[i * n + j] + pc[j * n + i], scaleOf(pc))
        }
      : null
    const F = fMatrix(square(s), 'svd', invariant)
    let dU: Value = matmul(U, mul(F, add(matmul(P, S), matmul(S, transpose(P)))))
    const dV = matmul(V, mul(F, add(matmul(S, P), matmul(transpose(P), S))))
    if (m > n) {
      refuseZeroSingular(concreteExamples(s), m, n)
      const AV = matmul(t, V)
      const projected = sub(AV, matmul(U, matmul(transpose(U), AV)))
      dU = add(dU, mul(projected, reshape(div(1, s), [1, n])))
    }
    return pack([dU, diagonal(P), dV])
  },
  batch: kernelBatch('numerics/linalg/svd'),
  shape: ([a]) => {
    const [m, n] = a.shape
    return float64Aval([m * n + n + n * n])
  },
  doc: { note: 'singular-value-decomposition', summary: 'The thin singular value decomposition of a tall matrix.' },
  test: {
    rtol: 1e-4,
    cases: (draw) => [{ inputs: [draw([4, 3])], params: { maxSweeps: 60, signOn: 'V' } }],
  },
})

/**
 * The tall SVD of a raw input, per the primitive's parameters.
 *
 * @param a The matrix to decompose, $m \times n$ with $m \ge n$. Its values are copied, so it is not modified.
 * @param p The primitive's parameters, of which the sweep limit `maxSweeps` and the sign convention `signOn` are read.
 * @returns The decomposition as `tall` returns it: `U`, `S` and `V` with `sweeps` and `converged`.
 */
function decomposeTall(a: Value, p: Params): SVD {
  const { m, n, a: data } = dense(a, 'svd')
  return tall(data, m, n, p.maxSweeps, p.signOn)
}

/**
 * Thin singular value decomposition $\Amat = \Umat \operatorname{diag}(\svec) \Vmat^\top$ of an $m \times n$ matrix:
 * $\Umat$ is $m \times k$, $\svec$ (the field `S`) has length $k$ and $\Vmat$ is $n \times k$, with $k = \min(m, n)$
 * and $\svec$ descending. Left singular vectors of zero singular values are completed to an orthonormal set.
 * Differentiable in both modes (Townsend, 2016): repeated singular values are handled as `eigh`'s repeated eigenvalues
 * (`NumericalError` 'degenerate' unless the function is invariant), and the vectors of a zero singular value of a
 * non-square matrix have no derivative. Inside `vmap`, `sweeps` is NaN and an example that does not converge throws.
 *
 * @param a The matrix $\Amat$ to decompose, $m \times n$ with any of tall, square or wide; it is not modified. A
 *   traced value makes the factors differentiable.
 * @param options How long the Jacobi iteration may run.
 * @param options.maxSweeps The most Jacobi sweeps (passes of rotations over every pair of columns) to run, 60 by
 *   default. When they run out, `converged` is false and the factors are only approximate.
 * @returns The factors `U`, `S` and `V`, with the number of `sweeps` used and whether the iteration `converged`.
 *
 * @example The thin singular value decomposition
 * const A = tensor([[3, 0], [0, 4], [0, 0]])
 * const { U, S, V, converged } = svd(A)
 * print('S =', S)
 * print('U =', U)
 * print('V =', V)
 * print('converged =', converged)
 */
export function svd<X extends Value>(a: X, { maxSweeps = 60 }: { maxSweeps?: number } = {}): SVD<TensorResult<X>> {
  const [m, n] = shapeOfValue(a)
  if (!isTraced(a)) {
    const { a: data } = dense(a, 'svd')
    if (m >= n) return tall(data, m, n, maxSweeps, 'V') as SVD<TensorResult<X>>
    // A wide matrix: decompose Aᵀ = V diag(S) Uᵀ, signed on Aᵀ's left vectors (our V), and swap the factors.
    const t = new Float64Array(n * m)
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) t[j * m + i] = data[i * n + j]
    const r = tall(t, n, m, maxSweeps, 'U')
    return { U: r.V, S: r.S, V: r.U, sweeps: r.sweeps, converged: r.converged } as SVD<TensorResult<X>>
  }
  const wide = m < n
  const input: Value = wide ? transpose(a) : a
  const [rows, cols] = wide ? [n, m] : [m, n]
  const params: Params = { maxSweeps, signOn: wide ? 'U' : 'V' }
  const found = concrete(input) === null ? undefined : decomposeTall(input, params)
  const [U, S, V] = unpack(svdOp([input], { ...params, found }), [[rows, cols], [cols], [cols, cols]])
  const r = { S, sweeps: found?.sweeps ?? NaN, converged: found?.converged ?? true }
  return (wide ? { ...r, U: V, V: U } : { ...r, U, V }) as SVD<TensorResult<X>>
}

/**
 * The default relative cutoff for treating a singular value as zero: $\max(m, n) \cdot \varepsilon$, as NumPy's
 * `matrix_rank`.
 *
 * @param m The number of rows of the matrix.
 * @param n The number of columns of the matrix.
 * @returns The cutoff as a fraction of the largest singular value.
 */
function defaultRtol(m: number, n: number): number {
  return Math.max(m, n) * EPS
}

/**
 * The dense routines below read the SVD's values; under a transformation they say so instead of failing inside.
 *
 * @param a The argument to check: a traced value throws `NotDifferentiableError`, anything else passes.
 * @param where The name of the calling function, used in the error and its message.
 */
function concreteOnly(a: unknown, where: string): void {
  if (isTraced(a as Value))
    throw new NotDifferentiableError(
      where,
      `${where}: not differentiable (it thresholds singular values); differentiate through \`svd\` or \`solve\``,
    )
}

/**
 * Moore–Penrose pseudo-inverse ($n \times m$) of an $m \times n$ matrix from its SVD: singular values at most
 * `rtol` $\cdot \sigma_{\max}$ are treated as zero (default $\max(m, n) \cdot \varepsilon$).
 *
 * @param a The matrix $\Amat$ to invert, $m \times n$ and of any rank; it is not modified. A traced value throws
 *   `NotDifferentiableError`.
 * @param options The cutoff below which a singular value counts as zero.
 * @param options.rtol The cutoff as a fraction of the largest singular value: singular values at most `rtol` times it
 *   are dropped rather than inverted. By default $\max(m, n) \cdot \varepsilon$.
 * @returns The pseudo-inverse $\Amat^+$, an $n \times m$ matrix.
 *
 * @example The pseudo-inverse of a rectangular matrix
 * const A = tensor([[1, 0], [0, 1], [1, 1]])
 * const Ap = pinv(A)
 * print('A⁺ =', Ap)
 * print('A⁺ A =', matmul(Ap, A))
 */
export function pinv(a: Tensor, { rtol }: { rtol?: number } = {}): Tensor {
  concreteOnly(a, 'pinv')
  const [m, n] = a.shape
  const { U, S, V } = svd(a)
  const k = S.shape[0]
  const cutoff = (rtol ?? defaultRtol(m, n)) * (S.data[0] ?? 0)
  const out = new Float64Array(n * m)
  for (let c = 0; c < k; c++) {
    const s = S.data[c]
    if (!(s > cutoff)) continue
    for (let i = 0; i < n; i++) {
      const vi = V.data[i * k + c] / s
      for (let j = 0; j < m; j++) out[i * m + j] += vi * U.data[j * k + c]
    }
  }
  return matrix(out, n, m)
}

/** The result of `lstsq`. */
export type LeastSquares = {
  /** The minimum-norm least-squares solution ($n$, or $n \times r$ for a matrix right-hand side). */
  x: Tensor
  /** Squared residual norm $\lVert \Amat\xvec - \bvec \rVert^2$ (per column for a matrix right-hand side). */
  residuals: Tensor
  /** Numerical rank: the number of singular values above the cutoff. */
  rank: number
  /** Singular values of $\Amat$, descending. */
  singularValues: Tensor
}

/**
 * Least squares: the $\xvec$ minimising $\lVert \Amat\xvec - \bvec \rVert$, and among those the one of least norm,
 * from the SVD of $\Amat$ ($m \times n$) with singular values at most `rtol` $\cdot \sigma_{\max}$ treated as zero
 * (default $\max(m, n) \cdot \varepsilon$). Rank deficiency is reported by `rank`.
 *
 * @param a The matrix $\Amat$, $m \times n$ and of any rank; it is not modified. A traced value throws
 *   `NotDifferentiableError`.
 * @param b The right-hand side: a vector of $m$ values, or an $m \times r$ matrix whose columns are $r$ separate
 *   problems solved together. It is not modified, and must not be traced.
 * @param options The cutoff below which a singular value counts as zero.
 * @param options.rtol The cutoff as a fraction of the largest singular value: singular values at most `rtol` times it
 *   are left out of the solution and of `rank`. By default $\max(m, n) \cdot \varepsilon$.
 * @returns The solution `x` (with $n$ values, or $n \times r$ for a matrix `b`), the squared residual norm of each
 *   column in `residuals` ($r$ values, one for a vector `b`), the numerical `rank`, and the `singularValues` of
 *   $\Amat$.
 *
 * @example Fit a line by least squares
 * // y = 1 + 2x, observed at x = 0, 1, 2.
 * const A = tensor([[1, 0], [1, 1], [1, 2]])
 * const y = tensor([1, 3, 5])
 * const { x, rank } = lstsq(A, y)
 * print('intercept and slope =', x)
 * print('rank =', rank)
 */
export function lstsq(a: Tensor, b: Tensor, { rtol }: { rtol?: number } = {}): LeastSquares {
  concreteOnly(a, 'lstsq')
  concreteOnly(b, 'lstsq')
  const [m, n] = a.shape
  const isVector = b.shape.length === 1
  const rhs = dense(isVector ? reshape(b, [-1, 1]) : b, 'lstsq')
  if (rhs.m !== m) throw new ShapeError('lstsq', `lstsq: A has ${m} rows but b has ${rhs.m}`)
  const r = rhs.n
  const { U, S, V } = svd(a)
  const k = S.shape[0]
  const cutoff = (rtol ?? defaultRtol(m, n)) * (S.data[0] ?? 0)
  const x = new Float64Array(n * r)
  let rank = 0
  for (let c = 0; c < k; c++) {
    const s = S.data[c]
    if (!(s > cutoff)) continue
    rank++
    for (let col = 0; col < r; col++) {
      let d = 0
      for (let j = 0; j < m; j++) d += U.data[j * k + c] * rhs.a[j * r + col]
      d /= s
      for (let i = 0; i < n; i++) x[i * r + col] += V.data[i * k + c] * d
    }
  }
  const residuals = new Float64Array(r)
  const A = dense(a, 'lstsq')
  for (let col = 0; col < r; col++) {
    for (let j = 0; j < m; j++) {
      let e = -rhs.a[j * r + col]
      for (let i = 0; i < n; i++) e += A.a[j * n + i] * x[i * r + col]
      residuals[col] += e * e
    }
  }
  return {
    x: isVector ? vector(x) : matrix(x, n, r),
    residuals: vector(residuals),
    rank,
    singularValues: S,
  }
}

/**
 * The 2-norm condition number $\sigma_{\max} / \sigma_{\min}$ ($\infty$ when $\sigma_{\min} = 0$) of an $m \times n$
 * matrix, over its $\min(m, n)$ singular values.
 *
 * @param a The matrix, $m \times n$; it is not modified. A traced value throws `NotDifferentiableError`.
 * @returns The ratio of the largest singular value to the smallest: at least 1, `Infinity` for a matrix that is not of
 *   full rank, and 0 for a matrix with no rows or no columns.
 *
 * @example Well and badly conditioned matrices
 * print(conditionNumber(tensor([[1, 0], [0, 1]])))
 * print(conditionNumber(tensor([[1, 1], [1, 1.0001]])))
 * print(conditionNumber(tensor([[1, 2], [2, 4]])))
 */
export function conditionNumber(a: Tensor): number {
  concreteOnly(a, 'conditionNumber')
  const { S } = svd(a)
  const k = S.shape[0]
  if (k === 0) return 0
  const smin = S.data[k - 1]
  return smin === 0 ? Infinity : S.data[0] / smin
}
