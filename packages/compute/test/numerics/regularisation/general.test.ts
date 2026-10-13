import { describe, expect, it } from 'vitest'
import { convolutionMatrix, differenceOperator, tikhonov, tikhonovPath } from 'aifn-compute/numerics/regularisation'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'

type Path = { L: number[][]; lambdas: number[]; x: number[][]; residualNorm: number[]; solutionNorm: number[] }
type Fixture = {
  A: number[][]
  b: number[]
  generalPath: { second: Path; square: Path & { gammas: number[] }; tall: Path }
  convolution: { kernel: number[]; n: number; boundary: 'zero' | 'periodic' | 'reflect'; matrix: number[][] }[]
}
const F = fixture<Fixture>('numerics/regularisation')

/** Every entry within `rel` of the expected, relative to the largest expected magnitude. */
const close = (actual: ArrayLike<number>, expected: ArrayLike<number>, rel: number) => {
  expect(actual.length).toBe(expected.length)
  const scale = Math.max(...Array.from(expected, Math.abs), 1e-300)
  for (let i = 0; i < expected.length; i++) expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(rel * scale)
}
const relClose = (actual: number, expected: number, rel: number) =>
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected))

describe('tikhonovPath with a regularisation matrix', () => {
  it.each(['second', 'square', 'tall'] as const)('%s L: matches the stacked least-squares solutions', (name) => {
    const c = F.generalPath[name]
    const path = tikhonovPath(F.A, F.b, c.lambdas, { L: c.L })
    const rows = toRows(path.x) as number[][]
    c.lambdas.forEach((lambda, j) => {
      // The stacked system is solved to its own conditioning, worst at the smallest weight.
      close(rows[j], c.x[j], lambda < 1e-9 ? 1e-7 : 1e-9)
      relClose(toFlat(path.residualNorm)[j], c.residualNorm[j], 1e-8)
      relClose(toFlat(path.solutionNorm)[j], c.solutionNorm[j], 1e-7)
      // A single solve (QR of the stacked system) agrees too.
      const one = tikhonov(F.A, F.b, { lambda, L: c.L })
      close(toFlat(one.x), c.x[j], lambda < 1e-9 ? 1e-7 : 1e-9)
    })
    const f = toRows(path.filterFactors) as number[][]
    const g = toFlat(path.singularValues)
    f.forEach((row, j) =>
      close(
        row,
        Array.from(g, (s) => (s * s) / (s * s + c.lambdas[j])),
        1e-14,
      ),
    )
  })

  it('reports the generalised singular values: those of A L^-1 for a square L, and of A for L = I', () => {
    const c = F.generalPath.square
    close(toFlat(tikhonovPath(F.A, F.b, [1], { L: c.L }).singularValues), c.gammas, 1e-12)
    const n = F.A[0].length
    const I = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)))
    const plain = tikhonovPath(F.A, F.b, [0, 1e-6, 1e-2])
    const viaI = tikhonovPath(F.A, F.b, [0, 1e-6, 1e-2], { L: I })
    close(toFlat(viaI.singularValues), toFlat(plain.singularValues), 1e-13)
    close(toFlat(viaI.x).slice(n), toFlat(plain.x).slice(n), 1e-11)
  })

  it('leaves the null space of L unpenalised and checks its arguments', () => {
    // With a huge weight and first differences the solution is the best constant.
    const path = tikhonovPath(F.A, F.b, [1e12], { L: differenceOperator(F.A[0].length) })
    const x = toFlat(path.x)
    expect(Math.max(...x) - Math.min(...x)).toBeLessThan(1e-8)
    const ones = F.A.map((row) => row.reduce((a, v) => a + v, 0))
    const best = ones.reduce((a, v, i) => a + v * F.b[i], 0) / ones.reduce((a, v) => a + v * v, 0)
    relClose(x[0], best, 1e-8)
    expect(() => tikhonovPath(F.A, F.b, [1], { L: [[1, 2]] })).toThrow(ShapeError)
    expect(() => tikhonovPath(F.A, F.b, [1], { L: [Array(F.A[0].length).fill(0)] })).toThrow(DomainError)
    expect(() => tikhonovPath(F.A, F.b, [-1], { L: differenceOperator(F.A[0].length) })).toThrow(DomainError)
  })
})

describe('convolutionMatrix', () => {
  it.each(F.convolution.map((c) => [`${c.kernel.length} taps, n = ${c.n}, ${c.boundary}`, c] as const))(
    '%s matches scipy.ndimage.convolve',
    (_, c) => {
      const K = convolutionMatrix(c.kernel, c.n, { boundary: c.boundary })
      expect(K.shape).toEqual([c.n, c.n])
      close(toFlat(K), c.matrix.flat(), 1e-15)
    },
  )

  it('checks its arguments', () => {
    expect(() => convolutionMatrix([], 4)).toThrow(DomainError)
    expect(() => convolutionMatrix([1, NaN], 4)).toThrow(DomainError)
    expect(() => convolutionMatrix([1], 0)).toThrow(DomainError)
    expect(() => convolutionMatrix([1], 3, { boundary: 'mirror' as 'reflect' })).toThrow(DomainError)
  })
})
