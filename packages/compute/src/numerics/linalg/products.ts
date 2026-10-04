/** Matrix functions composed from tensor primitives, so they are differentiable through them. */

import {
  diagonal,
  einsum,
  type NumberResult,
  reshape,
  shapeOfValue,
  sqrt,
  square,
  sum,
  type TensorResult,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

/**
 * The shape [rows, columns] of a matrix; throws `ShapeError` for any other rank.
 *
 * @param x The value whose shape is wanted (traced or not); only its shape is read, not its entries.
 * @param where The caller's name, used in error messages.
 * @returns The number of rows and the number of columns, in that order.
 */
function shape2(x: Value, where: string): [number, number] {
  const s = shapeOfValue(x)
  if (s.length !== 2) throw new ShapeError(where, `${where}: expected a matrix, got shape [${s.join(', ')}]`)
  return [s[0], s[1]]
}

/**
 * The Kronecker product $\Amat \otimes \Bmat$ of an $m \times n$ and a $p \times q$ matrix: the $mp \times nq$ block
 * matrix $[a_{ij}\Bmat]$.
 *
 * @param a The left factor $\Amat$ ($m \times n$): each of its entries scales one block of the result.
 * @param b The right factor $\Bmat$ ($p \times q$): the block that is repeated.
 * @returns The $mp \times nq$ matrix whose entry at row $ip + k$, column $jq + l$ is $a_{ij} b_{kl}$.
 *
 * @example The Kronecker product
 * const A = tensor([[1, 2], [3, 4]])
 * const I = tensor([[1, 0], [0, 1]])
 * print(kron(I, A))
 */
export function kron<A extends Value, B extends Value>(a: A, b: B): TensorResult<A | B> {
  const [m, n] = shape2(a, 'kron')
  const [p, q] = shape2(b, 'kron')
  // (A ⊗ B)[i·p + k, j·q + l] = a_ij b_kl: the product indexed ikjl, flattened in pairs.
  return reshape(einsum('ij,kl->ikjl', a, b), [m * p, n * q]) as TensorResult<A | B>
}

/**
 * The trace $\sum_i a_{ii}$ of a square matrix (named so as not to collide with `aifn-compute/foundation/trace`'s
 * runner).
 *
 * @param a The square matrix $\Amat$ ($n \times n$); a matrix that is not square throws `ShapeError`.
 * @returns The sum of the diagonal entries, as a number (or a traced scalar when `a` is traced).
 *
 * @example The trace
 * print(matrixTrace(tensor([[4, 1], [1, 3]])))
 */
export function matrixTrace<A extends Value>(a: A): NumberResult<A> {
  const [m, n] = shape2(a, 'matrixTrace')
  if (m !== n) throw new ShapeError('matrixTrace', `matrixTrace: expected a square matrix, got ${m}×${n}`)
  return sum(diagonal(a)) as NumberResult<A>
}

/**
 * The Frobenius norm $\sqrt{\sum_{ij} a_{ij}^2}$ of a matrix.
 *
 * @param a The matrix $\Amat$ ($m \times n$, not necessarily square); a value of any other rank throws `ShapeError`.
 * @returns The square root of the sum of the squared entries, as a number (or a traced scalar when `a` is traced).
 *
 * @example The Frobenius norm
 * print(normFrobenius(tensor([[3, 0], [0, 4]])))
 */
export function normFrobenius<A extends Value>(a: A): NumberResult<A> {
  shape2(a, 'normFrobenius')
  return sqrt(sum(square(a))) as NumberResult<A>
}
