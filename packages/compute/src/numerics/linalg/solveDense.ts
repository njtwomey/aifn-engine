/**
 * A dense solve for small systems inside inner loops, on row-major `Float64Array`s rather than tensors: the LU
 * elimination with partial pivoting of `luFactor` (Golub & Van Loan, 2013, "Matrix Computations", 4th ed., Algorithm
 * 3.4.1), without the tensor wrapping. It reports a singular matrix instead of throwing, so iterative methods can react
 * (damp, regularise, stop).
 */

import type { Size } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { factor, substitute } from './lu'

/** The result of `solveDense`. */
export type DenseSolution = {
  /**
   * $\Xmat$ ($n \times k$, row-major; length $n$ for one right-hand side), or null when $\Amat$ is singular.
   */
  x: Float64Array | null
  /** $\log\lvert\det\Amat\rvert$ ($-\infty$ when singular). */
  logAbsDet: number
  /**
   * True when a pivot is at most $n \varepsilon \max_{ij} \lvert A_{ij} \rvert$ (or $\Amat$ has a non-finite entry).
   */
  singular: boolean
}

/** An LU factor of a dense $n \times n$ matrix from `factorDense`, for repeated solves with `solveFactored`. */
export type DenseFactor = {
  /**
   * The packed $\Lmat$ (unit diagonal, below) and $\Umat$ (on and above the diagonal), row-major $n \times n$.
   */
  lu: Float64Array
  /** The row permutation: row $i$ of $\Pmat\Amat$ is row `perm[i]` of $\Amat$. */
  perm: Int32Array
  /** The number of rows (and columns) $n$ of $\Amat$. */
  n: Size
  /** $\log\lvert\det\Amat\rvert$ ($-\infty$ when singular). */
  logAbsDet: number
  /**
   * True when a pivot is at most $n \varepsilon \max_{ij} \lvert A_{ij} \rvert$ (or $\Amat$ has a non-finite entry).
   */
  singular: boolean
}

/**
 * The LU factor $\Pmat\Amat = \Lmat\Umat$ of a row-major $n \times n$ matrix `a` (not modified), for solving several
 * right-hand sides with one factorisation (modified Newton iterations, implicit ODE solvers). A singular or non-finite
 * matrix is reported in `singular`, never thrown.
 *
 * @param a The matrix $\Amat$ as a row-major array of $n^2$ values (element $(i, j)$ at index `i * n + j`). It is
 *   copied, not modified; any other length throws `ShapeError`.
 * @param n The number of rows (and columns) of $\Amat$.
 * @returns The factor: the packed `lu`, the row permutation `perm`, `n`, `logAbsDet` ($-\infty$ when singular, NaN
 *   when $\Amat$ has a non-finite entry) and the `singular` flag.
 *
 * @example Factor a row-major array once for repeated solves
 * const f = factorDense([4, 1, 1, 3], 2)
 * print('singular =', f.singular)
 * print('log |det| =', f.logAbsDet)
 * print('x1 =', solveFactored(f, [1, 2]))
 * print('x2 =', solveFactored(f, [0, 1]))
 */
export function factorDense(a: ArrayLike<number>, n: Size): DenseFactor {
  if (a.length !== n * n)
    throw new ShapeError('factorDense', `factorDense: a has ${a.length} entries, expected ${n}×${n}`)
  const lu = Float64Array.from(a)
  const perm = Int32Array.from({ length: n }, (_, i) => i)
  for (let i = 0; i < lu.length; i++)
    if (!Number.isFinite(lu[i])) return { lu, perm, n, logAbsDet: NaN, singular: true }
  const f = factor({ m: n, n, a: lu })
  if (f.singular) return { lu, perm: f.perm, n, logAbsDet: -Infinity, singular: true }
  let logAbsDet = 0
  for (let i = 0; i < n; i++) logAbsDet += Math.log(Math.abs(lu[i * n + i]))
  return { lu, perm: f.perm, n, logAbsDet, singular: false }
}

/**
 * Solve $\Amat\Xmat = \Bmat$ with a factor from `factorDense`; `b` is row-major $n \times k$ ($k$ is `b.length / n`)
 * and is not modified. Null when the factor is singular.
 *
 * @param options The factor of $\Amat$, as `factorDense` returns it. Read only.
 * @param options.lu The packed factor as a row-major array of $n^2$ values: $\Lmat$ strictly below the diagonal (unit
 *   diagonal implied), $\Umat$ on and above.
 * @param options.perm The row permutation of the factor: row $i$ of $\Pmat\Amat$ is row `perm[i]` of $\Amat$.
 * @param options.n The number of rows (and columns) of $\Amat$.
 * @param options.singular Whether the factor is singular; when true nothing is solved and the result is null.
 * @param b The right-hand side $\Bmat$ as a row-major array of $n \cdot k$ values (a vector of $n$ values for
 *   $k = 1$). Not modified; a length that is not a multiple of $n$ throws `ShapeError`.
 * @returns The solution $\Xmat$ as a new row-major array of $n \cdot k$ values, or null when the factor is singular.
 *
 * @example A singular factor gives null instead of throwing
 * print(solveFactored(factorDense([4, 1, 1, 3], 2), [1, 2]))
 * print(solveFactored(factorDense([1, 2, 2, 4], 2), [1, 1]))
 */
export function solveFactored(
  { lu, perm, n, singular }: DenseFactor,
  b: ArrayLike<number>,
): Float64Array<ArrayBuffer> | null {
  if (singular) return null
  if (n === 0) return new Float64Array(0)
  if (b.length % n !== 0)
    throw new ShapeError('solveFactored', `solveFactored: b has ${b.length} entries, not a multiple of ${n}`)
  const k = b.length / n
  const x = new Float64Array(n * k)
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) x[i * k + j] = b[perm[i] * k + j]
  substitute(lu, n, x, k)
  return x
}

/**
 * Solve $\Amat\Xmat = \Bmat$ for a row-major $n \times n$ matrix `a` and a row-major $n \times k$ right-hand side `b`
 * ($k$ is `b.length / n`; a vector for $k = 1$). Neither input is modified. `singular` is set, and `x` is null, when a
 * pivot is at most $n \varepsilon \max_{ij} \lvert A_{ij} \rvert$.
 *
 * @param a The matrix $\Amat$ as a row-major array of $n^2$ values (element $(i, j)$ at index `i * n + j`). Not
 *   modified; any other length throws `ShapeError`.
 * @param b The right-hand side $\Bmat$ as a row-major array of $n \cdot k$ values (a vector of $n$ values for
 *   $k = 1$). Not modified; a length that is not a multiple of $n$ throws `ShapeError`.
 * @param n The number of rows (and columns) of $\Amat$.
 * @returns `x`, the solution as a new row-major array of $n \cdot k$ values (null when singular); `logAbsDet`,
 *   $\log\lvert\det\Amat\rvert$; and the `singular` flag.
 *
 * @example Solve on plain arrays, for an inner loop
 * const { x, singular, logAbsDet } = solveDense([4, 1, 1, 3], [1, 2], 2)
 * print('x =', x)
 * print('singular =', singular)
 * print('log |det| =', logAbsDet)
 */
export function solveDense(a: ArrayLike<number>, b: ArrayLike<number>, n: Size): DenseSolution {
  if (b.length % (n || 1) !== 0)
    throw new ShapeError('solveDense', `solveDense: b has ${b.length} entries, not a multiple of ${n}`)
  const f = factorDense(a, n)
  return { x: solveFactored(f, b), logAbsDet: n === 0 ? 0 : f.logAbsDet, singular: f.singular }
}
