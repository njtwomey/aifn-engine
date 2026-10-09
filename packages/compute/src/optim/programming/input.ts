/**
 * Internal helpers: problem data for `aifn-compute/optim/programming`, read from tensors or plain arrays into dense
 * row-major working arrays (absent constraints read as empty), results wrapped as tensors, a linear solve that reports
 * singularity, and the independent-row test used to drop redundant equality rows.
 *
 * The arithmetic is `aifn-compute/foundation/tensor`'s `dense` kernels and `aifn-compute/numerics/linalg`'s
 * `solveDense`; this file only adapts shapes. The row test is modified Gram–Schmidt with re-orthogonalisation (Björck,
 * 1996, "Numerical Methods for Least Squares Problems", §2.4).
 */

import { solveDense } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Shape, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/**
 * A dense row-major working matrix of $m \times n$ values: `m` rows, `n` columns, and `a`, in which `a[i * n + j]` is
 * the element in row $i$ and column $j$.
 */
export type Mat = { m: Size; n: Size; a: Float64Array }

/**
 * Read a vector into a fresh `Float64Array`; `undefined` reads as `length` (default 0) zeros. Throws `ShapeError` when
 * `length` is given and the vector has another length.
 *
 * @param x The vector, as a tensor or plain array, or `undefined` for an absent one.
 * @param where The caller's name for error messages.
 * @param length The length the vector must have. Left out, any length is accepted, and an absent vector is empty.
 * @returns A copy of the values, not shared with `x`.
 */
export function readVector(x: VectorLike | undefined, where: string, length?: Size): dense.F64 {
  if (x === undefined) return new Float64Array(length ?? 0)
  const out = dense.toF64(x, where)
  if (length !== undefined && out.length !== length)
    throw new ShapeError(where, `${where}: expected length ${length}, got ${out.length}`)
  return out
}

/**
 * Read a matrix with `columns` columns; `undefined` or an empty list of rows reads as a $0 \times$ `columns` matrix.
 * Throws `ShapeError` when a non-empty matrix has another number of columns.
 *
 * @param x The matrix, as a tensor or a list of rows, or `undefined` for an absent constraint block.
 * @param where The caller's name for error messages.
 * @param columns The number of columns the matrix must have (the number of variables of the problem).
 * @returns The working matrix, row-major.
 */
export function readMatrix(x: MatrixLike | undefined, where: string, columns: Size): Mat {
  if (x === undefined) return { m: 0, n: columns, a: new Float64Array(0) }
  const { data, m, n } = dense.toMatrixF64(x, where)
  if (m === 0) return { m: 0, n: columns, a: new Float64Array(0) }
  if (n !== columns) throw new ShapeError(where, `${where}: expected ${columns} columns, got ${n}`)
  return { m, n, a: data }
}

/**
 * Throw `DomainError` unless every entry is finite.
 *
 * @param a The values to check (not modified).
 * @param where The caller's name for error messages.
 */
export function checkFinite(a: ArrayLike<number>, where: string): void {
  if (!dense.allFinite(a)) throw new DomainError(where, `${where}: non-finite entry`)
}

/**
 * A float64 vector tensor holding a copy of `a`.
 *
 * @param a The values, copied in order.
 * @returns A vector tensor of `a.length` values.
 */
export const vector = (a: ArrayLike<number>): Tensor => fromData(Float64Array.from(a))

/**
 * A float64 $m \times n$ matrix tensor holding a copy of `a`.
 *
 * @param a The $mn$ values, row-major.
 * @param m The number of rows.
 * @param n The number of columns.
 * @returns The matrix tensor, of shape `[m, n]`.
 */
export const matrix = (a: ArrayLike<number>, m: Size, n: Size): Tensor => fromData(Float64Array.from(a), [m, n])

/**
 * An int32 tensor holding a copy of `a`, of the given shape (default a vector).
 *
 * @param a The values, row-major; truncated to 32-bit integers.
 * @param shape The tensor's shape, whose sizes multiply to `a.length`. Left out, a vector of `a.length` values.
 * @returns The int32 tensor.
 */
export const intTensor = (a: ArrayLike<number>, shape: Shape = [a.length]): Tensor =>
  fromData(Int32Array.from(a), shape)

/**
 * The product $\Amat\xvec$ for a working matrix.
 *
 * @param A The working matrix $\Amat$ ($m \times n$).
 * @param x The vector $\xvec$ of $n$ values.
 * @returns $\Amat\xvec$, $m$ values.
 */
export const matVec = (A: Mat, x: ArrayLike<number>): dense.F64 => dense.matVec(A.a, x, A.m, A.n)

/**
 * The product $\Amat^\top\xvec$ for a working matrix.
 *
 * @param A The working matrix $\Amat$ ($m \times n$).
 * @param x The vector $\xvec$ of $m$ values.
 * @returns $\Amat^\top\xvec$, $n$ values.
 */
export const matTVec = (A: Mat, x: ArrayLike<number>): dense.F64 => dense.matTVec(A.a, x, A.m, A.n)

/**
 * Solve the $n \times n$ system $\Amat\xvec = \bvec$ (`solveDense`); a singular matrix gives `singular: true` and
 * $\xvec = \zeros$ rather than throwing.
 *
 * @param a The matrix $\Amat$ as a row-major array of $n^2$ values (not modified).
 * @param n The number of rows (and columns) of $\Amat$.
 * @param b The right-hand side $\bvec$, $n$ values.
 * @returns `x`, the solution ($n$ zeros when singular), and `singular`.
 */
export function solve(a: ArrayLike<number>, n: Size, b: ArrayLike<number>): { x: dense.F64; singular: boolean } {
  const out = solveDense(a, b, n)
  return out.x === null || out.singular
    ? { x: new Float64Array(n), singular: true }
    : { x: Float64Array.from(out.x), singular: false }
}

/**
 * Indices of a maximal set of linearly independent rows of $\Amat$ (in order, greedily). A row is dependent when the
 * norm of what is left after projecting out the rows kept so far is at most `tolerance` times its own norm (a zero row
 * is always dependent). With `rhs`, also reports `inconsistent`: a dependent row $\avec_i$ whose augmented row
 * $[\avec_i, b_i]$ is independent of the kept augmented rows, so that $\Amat\xvec = \bvec$ has no solution.
 *
 * @param A The working matrix $\Amat$ ($m \times n$), not modified.
 * @param rhs The right-hand side $\bvec$, $m$ values, for the consistency test. Left out, `inconsistent` is false.
 * @param tolerance The relative residual norm at or below which a row counts as dependent. The consistency test uses
 *   it relative to the larger of 1 and the augmented row's norm.
 * @returns `rows`, the indices of the kept rows in increasing order, and `inconsistent`.
 */
export function independentRows(
  A: Mat,
  rhs?: ArrayLike<number>,
  tolerance = 1e-9,
): { rows: number[]; inconsistent: boolean } {
  const basis: Float64Array[] = []
  const augmented: Float64Array[] = []
  const rows: number[] = []
  let inconsistent = false
  const reduce = (v: Float64Array, against: Float64Array[]) => {
    for (let pass = 0; pass < 2; pass++)
      for (const q of against) {
        const d = dense.dot(v, q)
        for (let k = 0; k < v.length; k++) v[k] -= d * q[k]
      }
    return dense.norm(v)
  }
  for (let i = 0; i < A.m; i++) {
    const row = A.a.slice(i * A.n, (i + 1) * A.n)
    const size = dense.norm(row)
    const aug = new Float64Array(A.n + 1)
    aug.set(row)
    aug[A.n] = rhs ? rhs[i] : 0
    const augSize = dense.norm(aug)
    const r = reduce(row, basis)
    const ra = reduce(aug, augmented)
    if (r <= tolerance * Math.max(size, 1e-300) || size === 0) {
      if (rhs && ra > tolerance * Math.max(augSize, 1)) inconsistent = true
      continue
    }
    rows.push(i)
    basis.push(row.map((v) => v / r))
    augmented.push(aug.map((v) => v / ra))
  }
  return { rows, inconsistent }
}
