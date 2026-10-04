import { describe, expect, it } from 'vitest'
import {
  cholesky,
  choleskyLogDet,
  choleskySolve,
  conditionNumber,
  det,
  eig,
  inverse,
  kron,
  logDet,
  lstsq,
  normFrobenius,
  pinv,
  solve,
  solveTriangular,
  matrixTrace,
  svd,
} from 'aifn-compute/numerics/linalg'
import {
  add,
  matmul,
  mul,
  realPart,
  sum,
  tensor,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { checkGradient } from '../../foundation/tensor/check-gradient'
import { grad } from 'aifn-compute/foundation/autodiff'
import { NotDifferentiableError } from 'aifn-compute/foundation/errors'

function data(shape: number[], seed: number, lo = -1, hi = 1): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  return tensor(
    Array.from(
      { length: n },
      (_, k) => lo + (hi - lo) * ((((Math.sin(12.9898 * (k + seed)) * 43758.5453) % 1) + 1) % 1),
    ),
    shape,
  )
}

/** A well-conditioned square matrix. */
function wellConditioned(n: number, seed: number): Tensor {
  return add(
    data([n, n], seed),
    mul(
      3,
      tensor(
        Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? 1 : 0)),
        [n, n],
      ),
    ),
  )
}

/** An SPD matrix built from x so that perturbations stay SPD: A = XXᵀ + nI. */
function spdOf(x: Value, n: number): Value {
  return add(
    matmul(x, transpose(x)),
    tensor(
      Array.from({ length: n * n }, (_, k) => (k % (n + 1) === 0 ? n : 0)),
      [n, n],
    ),
  )
}

describe('linear-algebra vjps match finite differences', () => {
  const A = wellConditioned(4, 1)
  const B = data([4, 2], 2)
  const b = data([4], 3)
  const L = tensor([
    [2, 0, 0],
    [0.3, 1.5, 0],
    [-0.4, 0.2, 1.8],
  ])
  it('solve, for matrix and vector right-hand sides', () => {
    checkGradient(solve, [A, B])
    checkGradient(solve, [A, b])
  })
  it('inverse, det and logDet', () => {
    checkGradient(inverse, [A])
    checkGradient((a) => det(a), [A])
    checkGradient((a) => logDet(a), [A])
  })
  it('solveTriangular in every mode (only the used triangle gets a cotangent)', () => {
    const rhs = data([3, 2], 4)
    for (const lower of [true, false]) {
      for (const transposed of [false, true]) {
        for (const unitDiagonal of [false, true]) {
          const T = lower ? L : transpose(L)
          const [gT] = checkGradient(
            (t, r) => solveTriangular(t, r, { lower, transpose: transposed, unitDiagonal }),
            [T, rhs],
          )
          const outside = lower ? [1, 2, 5] : [3, 6, 7]
          for (const k of outside) expect(Math.abs(gT[k])).toBe(0)
        }
      }
    }
    checkGradient((t, r) => solveTriangular(t, r), [L, data([3], 5)])
  })
  it('cholesky (lower triangle read), choleskySolve and choleskyLogDet', () => {
    const X = data([3, 3], 6)
    checkGradient((x) => cholesky(spdOf(x, 3)).L, [X])
    const S = spdOf(X, 3) as Tensor
    const [gA] = checkGradient((a) => cholesky(a).L, [S])
    for (const k of [1, 2, 5]) expect(Math.abs(gA[k])).toBe(0)
    checkGradient((x, r) => choleskySolve(cholesky(spdOf(x, 3)).L, r), [X, data([3, 2], 7)])
    checkGradient((x) => choleskyLogDet(cholesky(spdOf(x, 3)).L), [X])
  })
  it('choleskyLogDet agrees with logDet, and their gradients agree', () => {
    const S = spdOf(data([3, 3], 8), 3) as Tensor
    const g1 = checkGradient((x) => choleskyLogDet(cholesky(spdOf(x, 3)).L), [data([3, 3], 8)])
    const g2 = checkGradient((x) => logDet(spdOf(x, 3)), [data([3, 3], 8)])
    g1[0].forEach((v, k) => expect(v).toBeCloseTo(g2[0][k], 10))
    expect(choleskyLogDet(cholesky(S).L)).toBeCloseTo(logDet(S), 12)
  })
  it('kron, trace and normFrobenius', () => {
    checkGradient(kron, [data([2, 3], 9), data([2, 2], 10)])
    checkGradient((a) => matrixTrace(a), [A])
    checkGradient((a) => normFrobenius(a), [A])
  })
  it('the general eigenproblem refuses traced input (no derivative rule)', () => {
    expect(() => grad((x: Value) => sum(realPart(eig(x as never).values)))(A)).toThrow(NotDifferentiableError)
  })
})

describe('svd: the projection term at a zero singular value (review regression)', () => {
  it('a function of S alone of a rank-deficient tall matrix gives a finite gradient, not NaN', () => {
    const A = tensor([
      [1, 2],
      [2, 4],
      [3, 6],
    ])
    const g = toFlat(grad((a: Value) => sum(svd(a).S))(A) as Tensor)
    expect(Array.from(g).every(Number.isFinite)).toBe(true)
    // With s̄ = 1 the gradient is U Vᵀ; the projection term vanishes because Ū = 0.
    const { U, V } = svd(A)
    const u = toFlat(U)
    const v = toFlat(V)
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 2; j++)
        expect(g[i * 2 + j]).toBeCloseTo(u[i * 2] * v[j * 2] + u[i * 2 + 1] * v[j * 2 + 1], 12)
  })
})

describe('dense routines under grad (review maths 13)', () => {
  it('pinv, lstsq and conditionNumber throw NotDifferentiableError, not a TypeError', () => {
    const a = tensor([
      [2, 1],
      [1, 3],
    ])
    expect(() => grad((x: Value) => sum(pinv(x as Tensor)))(a)).toThrow(NotDifferentiableError)
    expect(() => grad((x: Value) => sum(lstsq(x as Tensor, tensor([1, 2])).x))(a)).toThrow(NotDifferentiableError)
    expect(() => grad((x: Value) => conditionNumber(x as Tensor))(a)).toThrow(NotDifferentiableError)
  })
})
