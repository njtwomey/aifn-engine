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
 * that must not throw can factor first and check. A `NumericalError` (from `aifn-compute/foundation/tensor`), so a catch of either works;
 * shape mismatches raise `ShapeError` and refused derivatives `NotDifferentiableError`.
 */
export class LinAlgError extends NumericalError {
  constructor(message: string, kind: NumericalKind) {
    super(message.slice(0, Math.max(0, message.indexOf(':'))) || 'linalg', message, kind)
    this.name = 'LinAlgError'
  }
}

/** A dense row-major working copy of a matrix: `a[i * n + j]` is element (i, j). */
export type Dense = { m: number; n: number; a: Float64Array }

/** Copy a rank-2 value (untraced) into a dense float64 working array, checking that every element is finite. */
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

/** As `dense`, and check that the matrix is square. */
export function denseSquare(x: Value, where: string): Dense {
  const d = dense(x, where)
  if (d.m !== d.n) throw new ShapeError(where, `${where}: expected a square matrix, got ${d.m}×${d.n}`)
  return d
}

/**
 * A matrix argument of the matrix-equation solvers as a float64 tensor: a matrix as given, a vector (a tensor of rank
 * 1 or an array of numbers) as a column, a number or a rank-0 tensor as 1×1.
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

/** Wrap a dense array as a matrix tensor. */
export function matrix(a: Float64Array, m: number, n: number): Tensor {
  return fromData(a, [m, n])
}

/** Wrap a Float64Array as a vector tensor. */
export function vector(a: Float64Array): Tensor {
  return fromData(a, [a.length])
}

/** Largest absolute entry. */
export function maxAbs(a: Float64Array): number {
  let m = 0
  for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k]))
  return m
}

/** A well-conditioned n×n test matrix (a random matrix plus 4I), for the generated primitive tests. */
export function wellConditioned(draw: Draw, n: number): Tensor {
  return add(draw([n, n]), mul(4, eye(n)))
}

/** A symmetric positive-definite n×n test matrix (B Bᵀ + 3I), for the generated primitive tests. */
export function positiveDefinite(draw: Draw, n: number): Tensor {
  const b = draw([n, n])
  return add(matmul(b, transpose(b)), mul(3, eye(n)))
}
