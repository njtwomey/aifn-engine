import { describe, expect, it } from 'vitest'
import {
  eig,
  expm,
  matrixTrace,
  pairwiseDistances,
  solve,
  solveDense,
  squaredDistances,
  symmetricInverseSqrt,
} from 'aifn-compute/numerics/linalg'
import { NumericalError, ShapeError } from 'aifn-compute/foundation/errors'
import { imagPart, realPart, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

// The general eigenproblem and the matrix exponential moved here from aifn-compute/ode; they are checked on ode's fixtures.
type OdeFixture = {
  expm: Record<string, { a: number[][]; expm: number[][] }>
  eig: Record<string, { a: number[][]; real: number[]; imag: number[] }>
}
const fx = fixture<OdeFixture>('dynamics/ode')

describe('eig and expm', () => {
  it.each(Object.entries(fx.expm))('expm matches scipy (%s)', (_k, { a, expm: ref }) => {
    const e = toRows(expm(a).value)
    const scale = Math.max(...ref.flat().map(Math.abs))
    ref.forEach((row, i) => row.forEach((v, j) => expect(Math.abs(e[i][j] - v) / scale).toBeLessThan(1e-12)))
  })

  it.each(Object.entries(fx.eig))('eig matches numpy (%s)', (_k, { a, real, imag }) => {
    const e = eig(a, { vectors: false })
    expect(e.converged).toBe(true)
    expect(e.values.dtype).toBe('complex128')
    toFlat(realPart(e.values)).forEach((v, i) => expect(v).toBeCloseTo(real[i], 9))
    toFlat(imagPart(e.values)).forEach((v, i) => expect(v).toBeCloseTo(imag[i], 9))
  })
})

describe('matrixTrace', () => {
  it('sums the diagonal and rejects non-square matrices', () => {
    expect(matrixTrace(tensor([1, 2, 3, 4], [2, 2]))).toBe(5)
    expect(() => matrixTrace(tensor([1, 2, 3, 4, 5, 6], [2, 3]))).toThrow(/square/)
  })
})

describe('pairwise and squared distances', () => {
  const X = [
    [0, 0],
    [3, 4],
  ]
  const Y = [
    [1, 0],
    [0, -2],
    [3, 4],
  ]
  it('euclidean, squared, manhattan, chebyshev, minkowski and cosine', () => {
    expect(toRows(pairwiseDistances(X, Y))).toEqual([
      [1, 2, 5],
      [Math.sqrt(20), Math.sqrt(45), 0],
    ])
    expect(toRows(squaredDistances(X, Y))).toEqual([
      [1, 4, 25],
      [20, 45, 0],
    ])
    expect(toRows(pairwiseDistances(X, Y, { metric: 'manhattan' }))[1]).toEqual([6, 9, 0])
    expect(toRows(pairwiseDistances(X, Y, { metric: 'chebyshev' }))[1]).toEqual([4, 6, 0])
    expect(toRows(pairwiseDistances(X, Y, { metric: 'minkowski', p: 3 }))[1][0]).toBeCloseTo((8 + 64) ** (1 / 3), 14)
    const cosine = toRows(pairwiseDistances(X, Y, { metric: 'cosine' }))
    expect(cosine[0][0]).toBeNaN()
    expect(cosine[1][2]).toBeCloseTo(0, 15)
  })
  it('defaults Y to X, reads a vector as points on a line and checks dimensions', () => {
    expect(toRows(squaredDistances(tensor([0, 1, 3])))).toEqual([
      [0, 1, 9],
      [1, 0, 4],
      [9, 4, 0],
    ])
    expect(() => pairwiseDistances(X, [[1, 2, 3]])).toThrow(/dimension/)
  })
})

describe('solveDense', () => {
  it('matches linalg.solve for one and several right-hand sides, with log|det A|', () => {
    const a = [4, 1, 2, 0, 3, 1, 2, 1, 5]
    const b = [1, 2, 3, 4, 5, 6] // 3×2
    const r = solveDense(a, b, 3)
    expect(r.singular).toBe(false)
    const want = toFlat(solve(tensor(a, [3, 3]), tensor(b, [3, 2])))
    Array.from(r.x!).forEach((v, i) => expect(v).toBeCloseTo(want[i], 13))
    expect(r.logAbsDet).toBeCloseTo(Math.log(46), 13)
    expect(Array.from(solveDense([2, 0, 0, 4], [2, 4], 2).x!)).toEqual([1, 1])
  })
  it('reports a singular matrix instead of throwing', () => {
    const r = solveDense([1, 2, 2, 4], [1, 1], 2)
    expect(r.singular).toBe(true)
    expect(r.x).toBeNull()
    expect(solveDense([1, NaN, 0, 1], [1, 1], 2).singular).toBe(true)
  })
})

describe('symmetricInverseSqrt', () => {
  it('is symmetric and squares to the inverse: W S W = I', () => {
    const S = [
      [4, 1, 0.5],
      [1, 3, 0.2],
      [0.5, 0.2, 2],
    ]
    const W = toRows(symmetricInverseSqrt(S))
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(W[i][j]).toBeCloseTo(W[j][i], 12)
    const WS = W.map((row) => S[0].map((_, j) => row.reduce((a, w, k) => a + w * S[k][j], 0)))
    const WSW = WS.map((row) => W[0].map((_, j) => row.reduce((a, v, k) => a + v * W[k][j], 0)))
    WSW.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(i === j ? 1 : 0, 12)))
  })

  it('refuses a singular matrix unless a floor is given, and a non-square one', () => {
    const S = [
      [1, 1],
      [1, 1],
    ]
    expect(() => symmetricInverseSqrt(S)).toThrow(NumericalError)
    expect(toFlat(symmetricInverseSqrt(S, { floor: 1e-6 })).every(Number.isFinite)).toBe(true)
    expect(() => symmetricInverseSqrt([[1, 2, 3]])).toThrow(ShapeError)
  })
})
