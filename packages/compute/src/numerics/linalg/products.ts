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

function shape2(x: Value, where: string): [number, number] {
  const s = shapeOfValue(x)
  if (s.length !== 2) throw new ShapeError(where, `${where}: expected a matrix, got shape [${s.join(', ')}]`)
  return [s[0], s[1]]
}

/** The Kronecker product A ⊗ B of an m×n and a p×q matrix: the (mp)×(nq) block matrix [a_ij B]. */
export function kron<A extends Value, B extends Value>(a: A, b: B): TensorResult<A | B> {
  const [m, n] = shape2(a, 'kron')
  const [p, q] = shape2(b, 'kron')
  // (A ⊗ B)[i·p + k, j·q + l] = a_ij b_kl: the product indexed ikjl, flattened in pairs.
  return reshape(einsum('ij,kl->ikjl', a, b), [m * p, n * q]) as TensorResult<A | B>
}

/** The trace Σᵢ aᵢᵢ of a square matrix (named so as not to collide with `aifn-compute/foundation/trace`'s runner). */
export function matrixTrace<A extends Value>(a: A): NumberResult<A> {
  const [m, n] = shape2(a, 'matrixTrace')
  if (m !== n) throw new ShapeError('matrixTrace', `matrixTrace: expected a square matrix, got ${m}×${n}`)
  return sum(diagonal(a)) as NumberResult<A>
}

/** The Frobenius norm √(Σᵢⱼ aᵢⱼ²) of a matrix. */
export function normFrobenius<A extends Value>(a: A): NumberResult<A> {
  shape2(a, 'normFrobenius')
  return sqrt(sum(square(a))) as NumberResult<A>
}
