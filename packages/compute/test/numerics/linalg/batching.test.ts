/**
 * Batching of the linear-algebra primitives: their own batch rules (a batched kernel over contiguous matrices) and
 * shape rules, nested `vmap`, derivatives through the batch in both orders, and degeneracy reported per example.
 */
import { describe, expect, it } from 'vitest'
import { grad, vmap } from 'aifn-compute/foundation/autodiff'
import { NumericalError } from 'aifn-compute/foundation/errors'
import {
  add,
  allclose,
  eye,
  matmul,
  mul,
  registry,
  reshape,
  slice,
  stack,
  sum,
  tensor,
  transpose,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { cholesky, det, eigh, logDet, lu, qr, solve, svd } from 'aifn-compute/numerics/linalg'

/** Deterministic pseudo-random data in [−1, 1]. */
function data(shape: number[], seed: number): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  return tensor(
    Array.from({ length: n }, (_, k) => 2 * ((((Math.sin(12.9898 * (k + seed)) * 43758.5453) % 1) + 1) % 1) - 1),
    shape,
  )
}
/** A batch of `b` well-conditioned n×n matrices (random plus 4I). */
const wellConditioned = (b: number, n: number, seed: number): Tensor =>
  add(data([b, n, n], seed), mul(4, eye(n))) as Tensor
/** A batch of symmetric positive-definite matrices (M Mᵀ + 3I). */
function positiveDefinite(b: number, n: number, seed: number): Tensor {
  const m = data([b, n, n], seed)
  return stack(
    Array.from({ length: b }, (_, k) => {
      const mk = slice(m, k) as Tensor
      return add(matmul(mk, transpose(mk)), mul(3, eye(n)))
    }),
    0,
  ) as Tensor
}
const close = (a: Value, b: Value, tol = 1e-10) =>
  expect(allclose(unwrap(a) as Tensor, unwrap(b) as Tensor, { rtol: tol, atol: tol })).toBe(true)
/** f applied to each example along axis 0, stacked. */
const loop = (f: (x: Value) => Value, x: Tensor): Tensor =>
  stack(
    Array.from({ length: x.shape[0] }, (_, k) => f(slice(x, k))),
    0,
  ) as Tensor

const LINALG = ['cholesky', 'lu', 'det', 'logDet', 'qr', 'eigh', 'svd', 'expm', 'solveTriangular', 'luSolve']

describe('linalg batching', () => {
  it('every factorisation and solve has its own batch rule and shape rule', () => {
    for (const name of LINALG) {
      const p = registry.get(`numerics/linalg/${name}`)
      expect(p, name).toBeDefined()
      expect(p?.rules.batch, name).toBe('own')
      expect(p?.rules.shape, name).toBe('own')
    }
  })

  it('vmap of det, logDet, cholesky and solve equals a loop', () => {
    const A = wellConditioned(4, 3, 1)
    const S = positiveDefinite(4, 3, 2)
    const b = data([4, 3], 3)
    close(
      vmap((a: Value) => det(a))(A) as Value,
      loop((a) => det(a), A),
    )
    close(
      vmap((a: Value) => logDet(a))(A) as Value,
      loop((a) => logDet(a), A),
    )
    close(
      vmap((s: Value) => cholesky(s).L)(S) as Value,
      loop((s) => cholesky(s).L, S),
    )
    close(
      vmap((a: Value, r: Value) => solve(a, r))(A, b) as Value,
      stack(
        Array.from({ length: 4 }, (_, k) => solve(slice(A, k), slice(b, k))),
        0,
      ),
    )
  })

  it('nested vmap merges the batch axes', () => {
    const A = add(data([2, 3, 3, 3], 4), mul(4, eye(3))) as Tensor
    const nested = vmap(vmap((a: Value) => det(a))) as (a: Tensor) => Value
    const expected = stack(
      Array.from({ length: 2 }, (_, i) => loop((a) => det(a), slice(A, i) as Tensor)),
      0,
    )
    close(nested(A), expected)
    const S = stack([positiveDefinite(3, 3, 5), positiveDefinite(3, 3, 6)], 0) as Tensor
    const L = vmap(vmap((s: Value) => cholesky(s).L))(S) as Value
    close(
      L,
      stack(
        Array.from({ length: 2 }, (_, i) => loop((s) => cholesky(s).L, slice(S, i) as Tensor)),
        0,
      ),
    )
  })

  it('per-example gradients (vmap of grad) and the gradient of a batched loss (grad of vmap) agree with a loop', () => {
    const A = wellConditioned(3, 3, 7)
    const loss = (a: Value) => add(logDet(a), sum(qr(a).R))
    const perExample = vmap(grad(loss))(A) as Value
    const looped = loop((a) => grad(loss)(a) as Value, A)
    close(perExample, looped, 1e-9)
    // The sum over examples of the loss: its gradient is the stack of the per-example gradients.
    const total = grad((a: Value) => sum(vmap(loss)(a) as Value))(A) as Value
    close(total, looped, 1e-9)
  })

  it('degeneracy is reported per example in both orders, and an invariant loss still differentiates', () => {
    // Example 1 has a repeated eigenvalue; examples 0 and 2 do not.
    const S = positiveDefinite(3, 3, 8)
    const degenerate = tensor(
      [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 5],
      ],
      [3, 3],
    )
    const batch = stack([slice(S, 0), degenerate, slice(S, 2)], 0) as Tensor
    const notInvariant = (a: Value) => sum(slice(eigh(a).vectors, 0))
    expect(() => vmap(grad(notInvariant))(batch)).toThrow(/batch example 1/)
    expect(() => grad((a: Value) => sum(vmap(notInvariant)(a) as Value))(batch)).toThrow(NumericalError)
    const invariant = (a: Value) => sum(eigh(a).values)
    close(
      vmap(grad(invariant))(batch) as Value,
      loop((a) => grad(invariant)(a) as Value, batch),
      1e-9,
    )
    // svd: the same through the kernel's derivative rules.
    const tall = data([3, 4, 3], 9)
    const s = (a: Value) => sum(svd(a).S)
    close(
      grad((a: Value) => sum(vmap(s)(a) as Value))(tall) as Value,
      loop((a) => grad(s)(a) as Value, tall),
      1e-8,
    )
  })

  it("lu's rules read each example's own permutation, and a singular example is named", () => {
    // Examples whose pivoting differs: the identity and a row swap.
    const A = stack(
      [
        tensor(
          [
            [4, 1],
            [1, 3],
          ],
          [2, 2],
        ),
        tensor(
          [
            [1, 3],
            [4, 1],
          ],
          [2, 2],
        ),
      ],
      0,
    ) as Tensor
    // The primitive (the `lu` wrapper reports per-example fields, so it is not used inside vmap): L, U, P packed.
    const luOp = registry.get('numerics/linalg/lu')
    if (!luOp) throw new Error('lu is not registered')
    const f = (a: Value) => {
      const packed = luOp.apply([a], {})
      return add(sum(mul(slice(packed, [0, 4]), slice(packed, [0, 4]))), sum(slice(packed, [4, 8])))
    }
    const looped = loop((a) => grad(f)(a) as Value, A)
    close(vmap(grad(f))(A) as Value, looped, 1e-9)
    close(grad((a: Value) => sum(vmap(f)(a) as Value))(A) as Value, looped, 1e-9)
    // The permutations differ between the examples, and each matches the wrapper's.
    const P = vmap((a: Value) => slice(luOp.apply([a], {}), [8, 12]))(A) as Value
    const wrapperP = loop((a) => lu(a as Tensor).P, A)
    close(P, reshape(wrapperP, [2, 4]))
    expect(allclose(slice(wrapperP, 0) as Tensor, slice(wrapperP, 1) as Tensor)).toBe(false)
    const singular = stack(
      [
        slice(A, 0),
        tensor(
          [
            [1, 2],
            [2, 4],
          ],
          [2, 2],
        ),
      ],
      0,
    ) as Tensor
    expect(() => vmap(grad(f))(singular)).toThrow(/batch example 1/)
  })
})
