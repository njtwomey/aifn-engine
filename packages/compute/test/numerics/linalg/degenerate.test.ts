/**
 * Gradients of the spectral factorisations at degenerate inputs (review maths 18): repeated eigenvalues, rank-deficient
 * and zero matrices. Each loss is an invariant that is smooth at the input (a trace, a Frobenius norm), so its gradient
 * is known in closed form even where the factors themselves are not unique.
 */
import { describe, expect, it } from 'vitest'
import { grad } from 'aifn-compute/foundation/autodiff'
import { NumericalError } from 'aifn-compute/foundation/errors'
import { mul, sum, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { eigh, qr, svd } from 'aifn-compute/numerics/linalg'

const flat = (v: Value) => Array.from(toFlat(v as Tensor))
const closeTo = (a: number[], b: number[], tol = 1e-10) => {
  expect(a.length).toBe(b.length)
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThanOrEqual(tol))
}

const identity3 = tensor([
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
])
// Eigenvalues 2, 2, 5: a repeated pair.
const repeated = tensor([
  [3, 1, 0],
  [1, 3, 0],
  [0, 0, 2],
])
const rankOne = tensor([
  [1, 2, 3],
  [2, 4, 6],
  [-1, -2, -3],
])
const zero = tensor([
  [0, 0],
  [0, 0],
])

/**
 * `eigh` reads only the lower triangle, so the gradient of a symmetric G sits there: off-diagonal entries collect
 * G[i, j] + G[j, i], the upper triangle is 0.
 */
const lower = (g: number[], n: number) =>
  g.map((_, k) => {
    const [i, j] = [Math.floor(k / n), k % n]
    return i === j ? g[k] : i > j ? g[k] + g[j * n + i] : 0
  })

describe('eigh at repeated eigenvalues', () => {
  it.each([
    ['identity', identity3],
    ['a repeated pair', repeated],
  ])('Σλ = tr A has gradient I and Σλ² = ‖A‖² has gradient 2A (%s)', (_, a) => {
    closeTo(flat(grad((x: Value) => sum(eigh(x).values))(a)), flat(identity3))
    closeTo(flat(grad((x: Value) => sum(mul(eigh(x).values, eigh(x).values)))(a)), lower(flat(mul(2, a)), 3))
  })
})

describe('svd at rank-deficient and zero matrices', () => {
  it.each([
    ['rank one', rankOne],
    ['zero', zero],
  ])('Σσ² = ‖A‖² has gradient 2A (%s)', (_, a) => {
    closeTo(flat(grad((x: Value) => sum(mul(svd(x).S, svd(x).S)))(a)), flat(mul(2, a)))
  })
  it('the nuclear norm Σσ of a rank-one matrix has a finite subgradient', () => {
    const g = flat(grad((x: Value) => sum(svd(x).S))(rankOne))
    expect(g.every(Number.isFinite)).toBe(true)
    // A subgradient of Σσ at A = σ u vᵀ is u vᵀ + W with W orthogonal to A and ‖W‖₂ ≤ 1, so ⟨G, A⟩ = σ = ‖A‖_F.
    const a = flat(rankOne)
    expect(g.reduce((s, v, i) => s + v * a[i], 0)).toBeCloseTo(Math.hypot(...a), 10)
  })
})

describe('qr at a rank-deficient matrix', () => {
  it('differentiating Q throws a NumericalError rather than returning NaN', () => {
    const tall = tensor([
      [1, 2],
      [2, 4],
      [3, 6],
    ])
    expect(() => grad((x: Value) => sum(mul(qr(x).R, qr(x).R)))(tall)).toThrow(NumericalError)
  })
})
