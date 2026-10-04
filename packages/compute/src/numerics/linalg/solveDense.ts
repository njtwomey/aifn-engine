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
  /** X (n × k, row-major; length n for one right-hand side), or null when A is singular. */
  x: Float64Array | null
  /** log |det A| (−∞ when singular). */
  logAbsDet: number
  /** True when a pivot is at most n·ε·max|A| (or A has a non-finite entry). */
  singular: boolean
}

/** An LU factor of a dense n × n matrix from `factorDense`, for repeated solves with `solveFactored`. */
export type DenseFactor = {
  /** The packed L (unit diagonal, below) and U (on and above the diagonal), row-major n × n. */
  lu: Float64Array
  /** The row permutation: row i of PA is row perm[i] of A. */
  perm: Int32Array
  n: Size
  /** log |det A| (−∞ when singular). */
  logAbsDet: number
  /** True when a pivot is at most n·ε·max|A| (or A has a non-finite entry). */
  singular: boolean
}

/**
 * The LU factor PA = LU of a row-major n × n matrix `a` (not modified), for solving several right-hand sides with one
 * factorisation (modified Newton iterations, implicit ODE solvers). A singular or non-finite matrix is reported in
 * `singular`, never thrown.
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
 * Solve A X = B with a factor from `factorDense`; `b` is row-major n × k (k = b.length / n) and is not modified. Null
 * when the factor is singular.
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
 * Solve A X = B for a row-major n × n matrix `a` and a row-major n × k right-hand side `b` (k = b.length / n; a vector
 * for k = 1). Neither input is modified. `singular` is set, and `x` is null, when a pivot is at most n·ε·max|A|.
 */
export function solveDense(a: ArrayLike<number>, b: ArrayLike<number>, n: Size): DenseSolution {
  if (b.length % (n || 1) !== 0)
    throw new ShapeError('solveDense', `solveDense: b has ${b.length} entries, not a multiple of ${n}`)
  const f = factorDense(a, n)
  return { x: solveFactored(f, b), logAbsDet: n === 0 ? 0 : f.logAbsDet, singular: f.singular }
}
