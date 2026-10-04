/**
 * Matrix-free linear operators and a few eigenpairs of a large symmetric one by the thick-restart Lanczos method (Wu &
 * Simon, 2000, "Thick-restart Lanczos method for large symmetric eigenvalue problems", SIAM J. Matrix Anal. Appl.
 * 22(2); Lanczos, 1950; Saad, 2011, "Numerical Methods for Large Eigenvalue Problems", §5.3 and §8.3), the task of
 * `scipy.sparse.linalg.eigsh`.
 */

import { dense, fromData, type Matrix, type Vector } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { child, standardNormals, stream, type Stream } from 'aifn-compute/foundation/random'
import { eigh } from './eigh'

type F64 = dense.F64

/**
 * A linear operator: a dense $n \times n$ matrix, or a function returning $\Amat\vvec$ for a vector $\vvec$
 * (matrix-free: a sparse matrix, a graph Laplacian, a Hessian–vector product). Shared by the iterative solvers
 * (`linearConjugateGradient`, `eigsh`).
 */
export type LinearOperator = MatrixLike | ((v: Vector) => VectorLike)

/**
 * $\Amat\vvec$ on working arrays for an operator of size $n$; `where` names the caller in errors.
 *
 * @param A The operator $\Amat$: an $n \times n$ matrix (copied once, here), or a function that returns
 *   $\Amat\vvec$ for a vector $\vvec$ of $n$ values.
 * @param n The size of the operator: the length of the vectors it takes and returns.
 * @param where The name of the calling function, used as the prefix of error messages (a matrix that is not
 *   $n \times n$, or a function that returns other than $n$ values, throws).
 * @returns The map $\vvec \mapsto \Amat\vvec$ on flat arrays of $n$ values: it reads its argument and returns
 *   the product as another array.
 *
 * @example A matrix as a function v ↦ Av
 * const apply = operatorOf(tensor([[2, 1], [1, 2]]), 2, 'example')
 * print(apply(Float64Array.of(1, 0)))
 */
export function operatorOf(A: LinearOperator, n: Size, where: string): (v: F64) => F64 {
  if (typeof A === 'function')
    return (v) => {
      const out = dense.toF64(A(dense.vec(v)), where)
      if (out.length !== n)
        throw new DomainError(where, `${where}: the operator returned ${out.length} values, not ${n}`)
      return out
    }
  const { data: a } = dense.toMatrixF64(A as MatrixLike, where, n, n)
  return (v) => dense.matVec(a, v, n, n)
}

/** Which eigenvalues `eigsh` finds: the largest or smallest (algebraically), or the largest in magnitude. */
export type EigshWhich = 'largest' | 'smallest' | 'magnitude'

/** Options for `eigsh`. */
export type EigshOptions = {
  /** The number of eigenpairs $k$ ($1 \le k < n$). Default 6, as scipy. */
  k?: Size
  /** Which end of the spectrum. Default `'largest'`. */
  which?: EigshWhich
  /**
   * The basis size $m$ (Lanczos vectors per cycle, $k < m \le n$). Default $\min(n, \max(2k + 1, 20))$, as scipy's
   * ncv.
   */
  basis?: Size
  /**
   * A Ritz pair $(\theta, \xvec)$ is converged when $\lVert \Amat\xvec - \theta\xvec \rVert$ is at most
   * `tolerance` times $\lVert \Amat \rVert$ (estimated by $\max_i \lvert \theta_i \rvert$). Default 1e-10.
   */
  tolerance?: Scalar
  /** Most restart cycles. Default 1000. */
  maxRestarts?: Size
  /** The starting vector (length $n$), or a stream to draw it from. Default: drawn from `stream('eigsh')`. */
  start?: VectorLike | Stream
}

/** The result of `eigsh`. */
export type EigshResult = {
  /**
   * The $k$ eigenvalues: descending for `largest`, ascending for `smallest`, by descending $\lvert \lambda \rvert$
   * for `magnitude`.
   */
  values: Vector
  /** The eigenvectors as columns ($n \times k$), unit length, column $j$ for `values[j]`. */
  vectors: Matrix
  /** $\lVert \Amat\xvec - \theta\xvec \rVert$ of each pair (the Lanczos estimate). */
  residuals: Vector
  /** True when all $k$ pairs met the tolerance. */
  converged: boolean
  /** Restart cycles used. */
  restarts: Size
  /** Products $\Amat\vvec$ used. */
  products: Size
}

/**
 * Is `x` a random stream (rather than a start vector)?
 *
 * @param x The `start` option of `eigsh`: a vector (an array, a typed array or a tensor), a stream, or undefined.
 * @returns True when `x` is a stream: an object with a `key` field that is neither an array nor a typed array.
 */
const isStream = (x: unknown): x is Stream =>
  typeof x === 'object' && x !== null && !Array.isArray(x) && !ArrayBuffer.isView(x) && 'key' in (x as object)

/**
 * The $k$ extreme eigenpairs of a symmetric operator $\Amat$ of size $n$ by thick-restart Lanczos, touching $\Amat$
 * only through products $\Amat\vvec$ (so $\Amat$ may be a function: matrix-free). Each cycle extends an orthonormal
 * basis $\Umat = [\uvec_1 \dots \uvec_m]$ of a Krylov space by the Lanczos recurrence
 * $\Amat\uvec_j = \Umat\Tmat\evec_j + \beta_j \uvec_{j+1}$, with full reorthogonalisation (each new vector is
 * orthogonalised twice against all of $\Umat$, so the basis stays orthonormal to rounding and no spurious copies of
 * converged eigenvalues appear). The eigenpairs $(\theta_i, \yvec_i)$ of the small $m \times m$ projection
 * $\Tmat = \Umat^\top\Amat\Umat$ are Ritz pairs $(\theta_i, \Umat\yvec_i)$ with residual norm
 * $\lVert \Amat\Umat\yvec_i - \theta_i \Umat\yvec_i \rVert = \beta_m \lvert \evec_m^\top \yvec_i \rvert$.
 * When the wanted $k$ have not converged, the method restarts thickly: it keeps the $p \approx k + (m - k)/2$ best
 * Ritz vectors and the residual direction $\uvec_{m+1}$, so $\Tmat$ becomes diagonal plus one arrow row
 * ($\beta_m \evec_m^\top \yvec_i$), and Lanczos continues from there. This is mathematically the implicitly
 * restarted Lanczos method that ARPACK (and scipy's `eigsh`) runs, with exact shifts.
 *
 * Cost per cycle: $m - p$ products $\Amat\vvec$ and $O(n m^2)$ for the reorthogonalisation. For the smallest
 * eigenvalues of a Laplacian, `which: 'smallest'` converges slowly when the bottom of the spectrum is clustered; a
 * shifted operator $(\Amat - \sigma\Imat)^{-1}$ or $\sigma\Imat - \Amat$ turns them into the largest. Throws
 * `DomainError` for $k$ outside $[1, n)$.
 *
 * @param A The symmetric operator $\Amat$: an $n \times n$ matrix, or a function returning $\Amat\vvec$ for a
 *   vector $\vvec$ of $n$ values. Symmetry is assumed, not checked.
 * @param n The size of the operator (the length of the vectors it acts on); an integer of at least 2.
 * @param options How many eigenpairs and which end of the spectrum, the basis size, the tolerance, the restart limit
 *   and the starting vector (default: the 6 largest, or $n - 1$ of them when $n \le 6$).
 * @returns The `values` and `vectors` (eigenvectors as columns) of the $k$ pairs, with their `residuals`, whether all
 *   `converged`, and the `restarts` and `products` used. Pairs that did not converge are still returned.
 *
 * @example The largest eigenvalues of a symmetric matrix, matrix-free
 * // The 1-D Laplacian of size 50, given only as a function v ↦ Av.
 * const n = 50
 * const laplacian = (v) => {
 *   const x = toArray(v)
 *   return x.map((xi, i) => 2 * xi - (x[i - 1] ?? 0) - (x[i + 1] ?? 0))
 * }
 * const { values, converged, products } = eigsh(laplacian, n, { k: 3 })
 * print('top 3 eigenvalues =', values)
 * print('converged =', converged)
 * print('matrix–vector products =', products)
 */
export function eigsh(A: LinearOperator, n: Size, options: EigshOptions = {}): EigshResult {
  const where = 'eigsh'
  const { k = Math.min(6, n - 1), which = 'largest', tolerance = 1e-10, maxRestarts = 1000 } = options
  if (!(Number.isInteger(n) && n >= 2)) throw new DomainError(where, 'eigsh: n must be an integer ≥ 2')
  if (!(Number.isInteger(k) && k >= 1 && k < n)) throw new DomainError(where, `eigsh: k must be in [1, ${n - 1}]`)
  const m = Math.min(n, options.basis ?? Math.max(2 * k + 1, 20))
  if (!(m > k)) throw new DomainError(where, 'eigsh: the basis must be larger than k')
  const op = operatorOf(A, n, where)
  const key = isStream(options.start) ? options.start : stream('eigsh')
  let restartDraws = 0
  const random = () => standardNormals(child(key, 'restart', restartDraws++), n) as F64

  const U: F64[] = []
  const T = new Float64Array(m * m)
  let products = 0

  /** Orthogonalise w against U[0..count) twice (classical Gram–Schmidt, repeated); returns the coefficients. */
  const orthogonalise = (w: F64, count: number): F64 => {
    const h = new Float64Array(count)
    for (let pass = 0; pass < 2; pass++)
      for (let i = 0; i < count; i++) {
        const c = dense.dot(U[i], w)
        h[i] += c
        const u = U[i]
        for (let r = 0; r < n; r++) w[r] -= c * u[r]
      }
    return h
  }
  /** A unit vector orthogonal to U[0..count), from the given vector or a fresh random one. */
  const unitOrthogonal = (v: F64 | null, count: number): F64 => {
    for (let attempt = 0; attempt < 5; attempt++) {
      const w = attempt === 0 && v ? Float64Array.from(v) : random()
      orthogonalise(w, count)
      const norm = dense.norm(w)
      if (norm > 1e-10) return dense.scale(1 / norm, w)
    }
    throw new DomainError(where, 'eigsh: could not extend the Krylov basis (n too small for the basis size?)')
  }

  const start =
    options.start !== undefined && !isStream(options.start) ? dense.toF64(options.start as VectorLike, where) : null
  if (start && start.length !== n) throw new DomainError(where, `eigsh: start has ${start.length} values, not ${n}`)
  U.push(unitOrthogonal(start, 0))
  let p = 0 // vectors kept from the last restart (columns 0 … p − 1 of T hold diag(θ) and the arrow in row/column p)
  let beta = 0
  let residualVector: F64 = new Float64Array(n)
  let restarts = 0
  let theta: F64 = new Float64Array(0)
  let Y: F64 = new Float64Array(0)
  let order: number[] = []
  let res: F64 = new Float64Array(k)
  let converged = false

  for (;;) {
    // Extend the basis from column p (or from 0 on the first cycle) to m by the Lanczos recurrence.
    for (let j = p; j < m; j++) {
      const w = op(U[j])
      products++
      const h = orthogonalise(w, j + 1)
      for (let i = 0; i <= j; i++) {
        // Columns before p hold the restart's diagonal and arrow already; only column j is new.
        T[i * m + j] = h[i]
        T[j * m + i] = h[i]
      }
      beta = dense.norm(w)
      if (j + 1 < m) {
        // An invariant subspace (β ≈ 0) is complete: continue with any unit vector orthogonal to it (coupling 0).
        const scaleA = Math.max(1e-300, Math.abs(T[j * m + j]))
        if (beta <= 1e-12 * scaleA) {
          U[j + 1] = unitOrthogonal(null, j + 1)
          T[(j + 1) * m + j] = 0
          T[j * m + j + 1] = 0
        } else {
          U[j + 1] = dense.scale(1 / beta, w)
          T[(j + 1) * m + j] = beta
          T[j * m + j + 1] = beta
        }
      } else residualVector = w
    }
    // Ritz pairs of the projection, symmetrised against rounding.
    const Ts = dense.symmetrise(T, m)
    const e = eigh(fromData(Ts, [m, m]))
    theta = dense.data(e.values) // descending
    Y = dense.data(e.vectors)
    const idx = Array.from({ length: m }, (_, i) => i)
    order =
      which === 'largest'
        ? idx
        : which === 'smallest'
          ? idx.reverse()
          : idx.sort((a, b) => Math.abs(theta[b]) - Math.abs(theta[a]))
    const normA = Math.max(...Array.from(theta, Math.abs), Number.MIN_VALUE)
    res = Float64Array.from({ length: k }, (_, i) => beta * Math.abs(Y[(m - 1) * m + order[i]]))
    converged = res.every((r) => r <= tolerance * normA)
    if (converged || restarts >= maxRestarts) break
    // Thick restart: keep the p best Ritz vectors and the residual direction.
    restarts++
    p = Math.min(m - 1, k + Math.floor((m - k) / 2))
    const kept = order.slice(0, p)
    const newU: F64[] = kept.map((c) => {
      const v = new Float64Array(n)
      for (let i = 0; i < m; i++) {
        const y = Y[i * m + c]
        if (y !== 0) for (let r = 0; r < n; r++) v[r] += y * U[i][r]
      }
      return v
    })
    T.fill(0)
    kept.forEach((c, i) => {
      T[i * m + i] = theta[c]
      const s = beta * Y[(m - 1) * m + c]
      T[i * m + p] = s
      T[p * m + i] = s
    })
    U.length = 0
    U.push(...newU)
    const unit = beta > 0 ? dense.scale(1 / beta, residualVector) : null
    U.push(unitOrthogonal(unit, p))
    // Columns 0 … p − 1 are set; the loop recomputes column p onwards (its h reproduces the arrow entries).
  }

  // The wanted Ritz vectors x = U y.
  const values = new Float64Array(k)
  const vectors = new Float64Array(n * k)
  for (let j = 0; j < k; j++) {
    const c = order[j]
    values[j] = theta[c]
    for (let i = 0; i < m; i++) {
      const y = Y[i * m + c]
      if (y !== 0) for (let r = 0; r < n; r++) vectors[r * k + j] += y * U[i][r]
    }
  }
  return {
    values: fromData(values, [k]),
    vectors: fromData(vectors, [n, k]),
    residuals: fromData(res, [k]),
    converged,
    restarts,
    products,
  }
}
