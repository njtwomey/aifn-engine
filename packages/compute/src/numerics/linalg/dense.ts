/** Internal helpers: dense row-major working copies of matrices, and the error type of this module. */

import {
  add,
  type Draw,
  EPS,
  eye,
  fromData,
  matmul,
  mul,
  transpose,
  isTensor,
  type Tensor,
  toFlat,
  unwrap,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { NumericalError, type NumericalKind, ShapeError } from 'aifn-compute/foundation/errors'

/** Machine epsilon for float64 (defined once in `aifn-compute/foundation/tensor`). */
export { EPS }

/**
 * Raised when a linear-algebra operation cannot produce a meaningful result: a singular system passed to a solver or a
 * non-finite input. Factorisations report such conditions in their results instead (`singular`, `failed`), so a caller
 * that must not throw can factor first and check. A `NumericalError` (from `aifn-compute/foundation/tensor`), so a
 * catch of either works; shape mismatches raise `ShapeError` and refused derivatives `NotDifferentiableError`.
 *
 * @example Catch it and read why
 * try {
 *   solve(tensor([[1, 2], [2, 4]]), tensor([1, 1]))
 * } catch (e) {
 *   print(e.name, 'of kind', e.kind)
 *   print(e.message)
 *   print('a LinAlgError:', e instanceof LinAlgError)
 *   print('a NumericalError:', e instanceof NumericalError)
 * }
 *
 * @example Factor first to test without throwing
 * const f = luFactor(tensor([[1, 2], [2, 4]]))
 * print('singular =', f.singular)
 * const g = luFactor(tensor([[4, 1], [1, 3]]))
 * print('singular =', g.singular)
 * print('x =', luSolve(g, tensor([1, 2])))
 */
export class LinAlgError extends NumericalError {
  constructor(message: string, kind: NumericalKind) {
    super(message.slice(0, Math.max(0, message.indexOf(':'))) || 'linalg', message, kind)
    this.name = 'LinAlgError'
  }
}

/**
 * A dense row-major working copy of an $m \times n$ matrix (`m` rows, `n` columns): `a[i * n + j]` is element
 * $(i, j)$.
 */
export type Dense = {
  /** The number of rows $m$. */
  m: number
  /** The number of columns $n$. */
  n: number
  /** The $m n$ entries, row by row. */
  a: Float64Array
}

/**
 * Copy a rank-2 value (untraced) into a dense float64 working array, checking that every element is finite.
 *
 * @param x The matrix to copy ($m \times n$); a traced value is read through its concrete value. It is not modified.
 *   Anything that is not of rank 2 throws `ShapeError`, and a non-finite entry `LinAlgError`.
 * @param where The caller's name, used in error messages.
 * @returns A fresh row-major copy of the $m \cdot n$ values with the dimensions `m` and `n`, which the caller may
 *   overwrite.
 */
export function dense(x: Value, where: string): Dense {
  const t = unwrap(x)
  if (!isTensor(t) || t.shape.length !== 2) {
    throw new ShapeError(
      where,
      `${where}: expected a matrix, got ${isTensor(t) ? `shape [${t.shape.join(', ')}]` : 'a number'}`,
    )
  }
  const a = Float64Array.from(toFlat(t))
  for (let k = 0; k < a.length; k++) {
    if (!Number.isFinite(a[k])) throw new LinAlgError(`${where}: the matrix has a non-finite entry`, 'not-finite')
  }
  return { m: t.shape[0], n: t.shape[1], a }
}

/**
 * As `dense`, and check that the matrix is square.
 *
 * @param x The square matrix to copy ($n \times n$); a traced value is read through its concrete value. It is not
 *   modified. A matrix that is not square throws `ShapeError`.
 * @param where The caller's name, used in error messages.
 * @returns A fresh row-major copy of the $n^2$ values with its dimensions (`m` and `n` are equal), which the caller
 *   may overwrite.
 */
export function denseSquare(x: Value, where: string): Dense {
  const d = dense(x, where)
  if (d.m !== d.n) throw new ShapeError(where, `${where}: expected a square matrix, got ${d.m}×${d.n}`)
  return d
}

/**
 * A matrix argument of the matrix-equation solvers as a float64 tensor: a matrix as given, a vector (a tensor of rank
 * 1 or an array of numbers) as a column, a number or a rank-0 tensor as $1 \times 1$.
 *
 * @param a The argument: a matrix (a rank-2 tensor or an array of equally long rows), a vector (a rank-1 tensor or an
 *   array of numbers, taken as one column) or a single number. It is copied, not modified; a tensor of rank above 2
 *   or rows of unequal length throw `ShapeError`.
 * @param where The caller's name, used in error messages.
 * @returns A new float64 tensor of rank 2: $m \times n$ for a matrix, $n \times 1$ for a vector of $n$ values,
 *   $1 \times 1$ for a number and $0 \times 0$ for an empty array.
 */
export function asMatrix(a: MatrixLike | VectorLike | number, where: string): Tensor {
  if (typeof a === 'number') return fromData(Float64Array.of(a), [1, 1])
  if (isTensor(a)) {
    if (a.shape.length > 2)
      throw new ShapeError(where, `${where}: expected a matrix, got shape [${a.shape.join(', ')}]`)
    const values = Float64Array.from(toFlat(a))
    if (a.shape.length === 2) return fromData(values, [a.shape[0], a.shape[1]])
    return fromData(values, [values.length, 1])
  }
  const list = a as ArrayLike<ArrayLike<number> | number>
  if (list.length === 0) return fromData(new Float64Array(0), [0, 0])
  if (typeof list[0] === 'number') return fromData(Float64Array.from(list as ArrayLike<number>), [list.length, 1])
  const rows = list as ArrayLike<ArrayLike<number>>
  const c = rows[0].length
  const out = new Float64Array(rows.length * c)
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].length !== c) throw new ShapeError(where, `${where}: ragged matrix rows`)
    for (let j = 0; j < c; j++) out[i * c + j] = rows[i][j]
  }
  return fromData(out, [rows.length, c])
}

/**
 * Wrap a dense row-major array as an $m \times n$ matrix tensor.
 *
 * @param a The entries as a row-major array of $m \cdot n$ values (element $(i, j)$ at index `i * n + j`). It is
 *   handed to the tensor as given, not copied here.
 * @param m The number of rows.
 * @param n The number of columns.
 * @returns The $m \times n$ tensor holding those entries.
 */
export function matrix(a: Float64Array, m: number, n: number): Tensor {
  return fromData(a, [m, n])
}

/**
 * Wrap a Float64Array as a vector tensor.
 *
 * @param a The entries of the vector, in order. It is handed to the tensor as given, not copied here.
 * @returns The rank-1 tensor of `a.length` values.
 */
export function vector(a: Float64Array): Tensor {
  return fromData(a, [a.length])
}

/**
 * The largest absolute entry of a dense array (0 for an empty one).
 *
 * @param a The entries to scan, in any layout (a matrix's row-major data, for instance). Read only.
 * @returns $\max_k \lvert a_k \rvert$, or 0 when `a` is empty.
 */
export function maxAbs(a: Float64Array): number {
  let m = 0
  for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k]))
  return m
}

/**
 * A well-conditioned $n \times n$ test matrix (a random matrix plus $4\Imat$), for the generated primitive tests.
 *
 * @param draw The test's source of random tensors: called once, for an $n \times n$ matrix with entries in its
 *   default domain, $[-2, 2]$.
 * @param n The number of rows (and columns) of the matrix.
 * @returns The $n \times n$ matrix: the drawn one with 4 added to each diagonal entry.
 */
export function wellConditioned(draw: Draw, n: number): Tensor {
  return add(draw([n, n]), mul(4, eye(n)))
}

/**
 * A symmetric positive-definite $n \times n$ test matrix ($\Bmat\Bmat^\top + 3\Imat$), for the generated primitive
 * tests.
 *
 * @param draw The test's source of random tensors: called once, for the $n \times n$ matrix $\Bmat$ with entries in
 *   its default domain, $[-2, 2]$.
 * @param n The number of rows (and columns) of the matrix.
 * @returns The symmetric $n \times n$ matrix $\Bmat\Bmat^\top + 3\Imat$, whose eigenvalues are at least 3.
 */
export function positiveDefinite(draw: Draw, n: number): Tensor {
  const b = draw([n, n])
  return add(matmul(b, transpose(b)), mul(3, eye(n)))
}
