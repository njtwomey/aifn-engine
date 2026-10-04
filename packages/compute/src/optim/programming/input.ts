/**
 * Problem data for `aifn-compute/optim/programming` (internal): reading vectors and matrices given as tensors or plain arrays
 * into dense row-major working arrays (absent constraints read as empty), wrapping results as tensors, a linear solve
 * that reports singularity, and the independent-row test used to drop redundant equality rows. The arithmetic is
 * `aifn-compute/foundation/tensor`'s `dense` kernels and `aifn-compute/numerics/linalg`'s `solveDense`; this file only adapts shapes.
 * The row test is modified Gram–Schmidt with re-orthogonalisation (Björck, 1996, "Numerical Methods for Least
 * Squares Problems", §2.4).
 */

import { solveDense } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { MatrixLike, Shape, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A dense row-major working matrix: `a[i * n + j]` is element (i, j). */
export type Mat = { m: Size; n: Size; a: Float64Array }

/** Read a vector into a fresh Float64Array; `undefined` reads as `length` (default 0) zeros. Checks the length. */
export function readVector(x: VectorLike | undefined, where: string, length?: Size): dense.F64 {
  if (x === undefined) return new Float64Array(length ?? 0)
  const out = dense.toF64(x, where)
  if (length !== undefined && out.length !== length)
    throw new ShapeError(where, `${where}: expected length ${length}, got ${out.length}`)
  return out
}

/** Read a matrix with `columns` columns; `undefined` or an empty list of rows reads as a 0×columns matrix. */
export function readMatrix(x: MatrixLike | undefined, where: string, columns: Size): Mat {
  if (x === undefined) return { m: 0, n: columns, a: new Float64Array(0) }
  const { data, m, n } = dense.toMatrixF64(x, where)
  if (m === 0) return { m: 0, n: columns, a: new Float64Array(0) }
  if (n !== columns) throw new ShapeError(where, `${where}: expected ${columns} columns, got ${n}`)
  return { m, n, a: data }
}

/** Throw unless every entry is finite. */
export function checkFinite(a: ArrayLike<number>, where: string): void {
  if (!dense.allFinite(a)) throw new DomainError(where, `${where}: non-finite entry`)
}

/** A float64 vector tensor holding a copy of `a`. */
export const vector = (a: ArrayLike<number>): Tensor => fromData(Float64Array.from(a))

/** A float64 m×n matrix tensor holding a copy of `a` (row-major). */
export const matrix = (a: ArrayLike<number>, m: Size, n: Size): Tensor => fromData(Float64Array.from(a), [m, n])

/** An int32 tensor holding a copy of `a`, of the given shape (default a vector). */
export const intTensor = (a: ArrayLike<number>, shape: Shape = [a.length]): Tensor =>
  fromData(Int32Array.from(a), shape)

/** A x for a working matrix. */
export const matVec = (A: Mat, x: ArrayLike<number>): dense.F64 => dense.matVec(A.a, x, A.m, A.n)

/** Aᵀ x for a working matrix. */
export const matTVec = (A: Mat, x: ArrayLike<number>): dense.F64 => dense.matTVec(A.a, x, A.m, A.n)

/** Solve the n×n system a x = b (`solveDense`); a singular matrix gives `singular: true` and x = 0. */
export function solve(a: ArrayLike<number>, n: Size, b: ArrayLike<number>): { x: dense.F64; singular: boolean } {
  const out = solveDense(a, b, n)
  return out.x === null || out.singular
    ? { x: new Float64Array(n), singular: true }
    : { x: Float64Array.from(out.x), singular: false }
}

/**
 * Indices of a maximal set of linearly independent rows of `A` (in order, greedily). A row is dependent when its
 * residual norm is at most `tolerance` times its own norm. With `rhs`, also reports `inconsistent`: a dependent row whose
 * augmented row [aᵢ, bᵢ] is independent, i.e. the system A x = b has no solution.
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
