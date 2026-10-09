/**
 * Internal helpers: the Gaussian algebra the state-space filters share, on tensors.
 *
 * They read model arguments as matrices and vectors, form the products $\Amat\Bmat\Amat^\top$ and
 * $(\Amat + \Amat^\top)/2$, solve a system reporting a singular one instead of throwing (with
 * $\log\lvert\det\Amat\rvert$), take the positive semi-definite square root, and stack per-step results.
 * Arithmetic is `aifn-compute/foundation/tensor`'s and factorisations are `aifn-compute/numerics/linalg`'s, so nothing
 * here is a second definition.
 */

import type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'
import { ShapeError } from 'aifn-compute/foundation/errors'
import {
  add,
  diagonal,
  dot,
  fromData,
  isTensor,
  matmul,
  mul,
  reshape,
  stack,
  tensor,
  toFlat,
  transpose,
  type Matrix,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import { eigh, luFactor, luSolve } from 'aifn-compute/numerics/linalg'

export type { MatrixLike, VectorLike } from 'aifn-compute/foundation/contracts'

/**
 * A matrix argument as a float64 `[r, c]` tensor; a number or a scalar tensor is $1 \times 1$. A matrix tensor is
 * returned as it is, not copied. Throws `ShapeError` for a tensor of rank above 2 or a vector, and for ragged rows.
 *
 * @param a The matrix: a number, a tensor, or an array of rows.
 * @param where The caller's name, for error messages.
 * @returns The matrix as a tensor.
 */
export function asMatrix(a: MatrixLike | number, where: string): Matrix {
  if (typeof a === 'number') return tensor([[a]]) as Matrix
  if (isTensor(a)) {
    if (a.shape.length === 0) return reshape(a, [1, 1]) as Matrix
    if (a.shape.length !== 2)
      throw new ShapeError(where, `${where}: expected a matrix, got shape [${a.shape.join(', ')}]`)
    return a as Matrix
  }
  const rows = (a as readonly ArrayLike<number>[]).map((r) => Array.from(r))
  if (rows.some((r) => r.length !== rows[0].length)) throw new ShapeError(where, `${where}: ragged matrix rows`)
  return fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0]) as Matrix
}

/**
 * A vector argument as a float64 `[n]` tensor; a number or a scalar tensor has length 1. Throws `ShapeError` for a
 * tensor of rank above 1.
 *
 * @param v The vector: a number, a tensor or an array of numbers.
 * @param where The caller's name, for error messages.
 * @returns The vector as a tensor (a rank-1 tensor is returned as it is).
 */
export function asVector(v: VectorLike | number, where: string): Vector {
  if (typeof v === 'number') return tensor([v]) as Vector
  if (isTensor(v)) {
    if (v.shape.length > 1)
      throw new ShapeError(where, `${where}: expected a vector, got shape [${v.shape.join(', ')}]`)
    return (v.shape.length === 0 ? reshape(v, [1]) : v) as Vector
  }
  return fromData(Float64Array.from(v as ArrayLike<number>)) as Vector
}

/**
 * Observations as $T$ rows of $m$ numbers: a vector is $T$ scalar observations, a matrix is $T \times m$. NaN marks
 * a missing value, and is kept. Ragged rows throw `ShapeError`.
 *
 * @param y The series: a vector (one scalar per step) or a matrix, as a tensor or nested arrays, with one row per
 *   time step.
 * @param where The caller's name, for error messages.
 * @returns A new array of $T$ rows, each an array of $m$ numbers.
 */
export function asSeries(y: VectorLike | MatrixLike, where: string): number[][] {
  if (isTensor(y)) {
    if (y.shape.length === 1) return toFlat(y).map((v) => [v])
    const [T, m] = asMatrix(y, where).shape
    const flat = toFlat(y)
    return Array.from({ length: T }, (_, t) => flat.slice(t * m, (t + 1) * m))
  }
  const list = y as ArrayLike<number> | readonly ArrayLike<number>[]
  if (list.length > 0 && typeof list[0] === 'number') return Array.from(list as ArrayLike<number>, (v) => [v])
  const rows = (list as readonly ArrayLike<number>[]).map((r) => Array.from(r))
  if (rows.some((r) => r.length !== rows[0].length)) throw new ShapeError(where, `${where}: ragged observation rows`)
  return rows
}

/**
 * $(\Amat + \Amat^\top)/2$, to remove the asymmetry rounding leaves in a covariance.
 *
 * @param a A square matrix $\Amat$.
 * @returns Its symmetric part, a new tensor.
 */
export const symmetrise = (a: Tensor): Tensor => mul(0.5, add(a, transpose(a))) as Tensor

/**
 * $\Amat\Bmat\Amat^\top$: a covariance $\Bmat$ carried through the linear map $\Amat$.
 *
 * @param a The map $\Amat$ ($r \times n$).
 * @param b The matrix $\Bmat$ ($n \times n$).
 * @returns The $r \times r$ product.
 */
export const sandwich = (a: Tensor, b: Tensor): Tensor => matmul(matmul(a, b), transpose(a)) as Tensor

/**
 * $\Xmat$ with $\Amat\Xmat = \Bmat$ and $\log\lvert\det\Amat\rvert$ by one LU factorisation
 * (`aifn-compute/numerics/linalg`), or null when $\Amat$ is singular to working precision (a pivot at most
 * $n \varepsilon \max_{ij}\lvert A_{ij}\rvert$): the caller reports it rather than dividing by zero.
 *
 * @param a The square matrix $\Amat$ ($n \times n$).
 * @param b The right-hand side $\Bmat$: a vector of $n$ values or an $n \times r$ matrix.
 * @returns `x`, the solution with the shape of `b`, and `logAbsDet`; or null for a singular $\Amat$.
 */
export function solveOrNull(a: Tensor, b: Tensor): { x: Tensor; logAbsDet: number } | null {
  const f = luFactor(a)
  if (f.singular) return null
  const logAbsDet = toFlat(diagonal(f.packed)).reduce((s, u) => s + Math.log(Math.abs(u)), 0)
  return { x: luSolve(f, b) as Tensor, logAbsDet }
}

/**
 * A square root $\Smat$ with $\Smat\Smat^\top = \Amat$ for a symmetric positive semi-definite $\Amat$, from the
 * symmetric eigendecomposition: $\Smat = \Vmat \diag(\sqrt{\max(\lambda_i, 0)})$. Unlike a Cholesky factor it exists
 * for singular $\Amat$ (a noise covariance with a deterministic component), so sampling such noise gives exact zeros
 * rather than NaN. `negative` is the most negative eigenvalue found (0 for a valid covariance).
 *
 * @param a The matrix $\Amat$ ($n \times n$); its symmetric part is used.
 * @returns `S`, the square root ($n \times n$, not triangular), and `negative`, the most negative eigenvalue or 0.
 */
export function sqrtPsd(a: Tensor): { S: Matrix; negative: number } {
  const n = a.shape[0]
  if (n === 0) return { S: a as Matrix, negative: 0 }
  const { values, vectors } = eigh(symmetrise(a))
  const lambda = toFlat(values)
  const negative = Math.min(0, ...lambda)
  const roots = fromData(Float64Array.from(lambda, (l) => Math.sqrt(Math.max(l, 0))))
  return { S: mul(vectors, roots) as Matrix, negative }
}

/**
 * $\avec^\top\bvec$ of two vectors as a number.
 *
 * @param a The vector $\avec$.
 * @param b The vector $\bvec$, of the same length.
 * @returns The inner product.
 */
export function quadratic(a: Tensor, b: Tensor): number {
  const v = dot(a, b)
  return typeof v === 'number' ? v : toFlat(v as Tensor)[0]
}

/**
 * $T$ per-step tensors of shape `shape` stacked into `[T, ...shape]` (an empty `[0, ...shape]` when $T = 0$).
 *
 * @param list The per-step tensors, in time order, each of shape `shape`.
 * @param shape The shape of one step, used only to shape the empty result.
 * @returns The stacked tensor.
 */
export function stackSteps(list: readonly Tensor[], shape: readonly number[]): Tensor {
  if (list.length === 0) return fromData(new Float64Array(0), [0, ...shape])
  return stack(list, 0) as Tensor
}
