/**
 * aifn-compute/learning/selection, linear smoothers: `ridgeCv` against scikit-learn's RidgeCV (`cv_results_` in both
 * `gcv_mode`s, `alpha_`, and `alpha_per_target` column by column) and brute-force refits, GCV and degrees of freedom
 * against the explicit hat matrix, `ridgeSmoother` against the normal equations, and `linearSmootherCv` on a
 * Nadaraya-Watson smoother against leave-one-out by refitting.
 */
import { describe, expect, it } from 'vitest'
import { tensor, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { linearSmootherCv, ridgeCv, ridgeSmoother } from 'aifn-compute/learning/selection'
import { fixture } from '../../fixtures'

interface RidgeCase {
  x: number[][]
  y: number[]
  intercept: boolean
  lambdas: number[]
  looResiduals: number[][]
  sklearn: Record<'svd' | 'eigen', { cvResults: number[][]; alpha: number; bestScore: number }>
  gcv: number[]
  df: number[]
  smoother: number[][]
}

interface Fixture {
  ridgeCv: RidgeCase[]
  alphaPerTarget: { x: number[][]; y: number[][]; lambdas: number[]; alpha: number[]; cvResults: number[][][] }
  nadarayaWatson: { smoother: number[][]; y: number[]; looResiduals: number[]; loo: number; gcv: number; df: number }
}

const F = fixture<Fixture>('learning/selection')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
}

describe('ridgeCv', () => {
  it.each(F.ridgeCv.map((c, i) => [i, c] as const))('case %i matches RidgeCV and brute-force refits', (_, c) => {
    const cv = ridgeCv(tensor(c.x), tensor(c.y), c.lambdas, { intercept: c.intercept })
    const residuals = toArray(cv.looResiduals) as number[][]
    c.lambdas.forEach((_, t) => {
      close(residuals[t], c.looResiduals[t], 1e-8)
      for (const mode of ['svd', 'eigen'] as const)
        close(
          residuals[t].map((r) => r * r),
          c.sklearn[mode].cvResults.map((row) => row[t]),
          1e-8,
        )
    })
    close(toFlat(cv.gcv), c.gcv, 1e-9)
    close(toFlat(cv.df), c.df, 1e-9)
    expect(c.lambdas[cv.bestLoo]).toBe(c.sklearn.svd.alpha)
    expect(-toFlat(cv.loo)[cv.bestLoo]).toBeCloseTo(c.sklearn.svd.bestScore, 9)
  })

  it('alpha_per_target: each target cross-validated alone picks the same alpha', () => {
    const c = F.alphaPerTarget
    c.y.forEach((y, j) => {
      const cv = ridgeCv(tensor(c.x), tensor(y), c.lambdas)
      expect(c.lambdas[cv.bestLoo]).toBe(c.alpha[j])
      const residuals = toArray(cv.looResiduals) as number[][]
      residuals.forEach((r, t) =>
        close(
          r.map((v) => v * v),
          c.cvResults[j][t],
          1e-8,
        ),
      )
    })
  })

  it('throws for a negative penalty or mismatched rows', () => {
    expect(() => ridgeCv(tensor([[1], [2]]), tensor([1, 2]), [-1])).toThrow(/non-negative/)
    expect(() => ridgeCv(tensor([[1], [2]]), tensor([1, 2, 3]), [1])).toThrow(/rows/)
  })
})

describe('ridgeSmoother', () => {
  it.each(F.ridgeCv.map((c, i) => [i, c] as const))('case %i equals the normal-equations hat matrix', (_, c) => {
    const S = ridgeSmoother(tensor(c.x), c.lambdas[4], { intercept: c.intercept })
    close(toFlat(S), c.smoother.flat(), 1e-9)
  })

  it('agrees with ridgeCv through linearSmootherCv', () => {
    const c = F.ridgeCv[0]
    const cv = ridgeCv(tensor(c.x), tensor(c.y), [c.lambdas[4]])
    const sm = linearSmootherCv(ridgeSmoother(tensor(c.x), c.lambdas[4]), tensor(c.y))
    close(toFlat(sm.looResiduals), toFlat(cv.looResiduals), 1e-9)
    expect(sm.gcv).toBeCloseTo(toFlat(cv.gcv)[0], 10)
    expect(sm.df).toBeCloseTo(toFlat(cv.df)[0], 10)
  })

  it('at lambda = 0 is a projection', () => {
    const x = tensor([
      [0, 1],
      [1, 0],
      [2, 2],
      [3, 1],
      [4, 3],
    ])
    const S = toArray(ridgeSmoother(x, 0)) as number[][]
    const S2 = S.map((row) => S[0].map((_, j) => row.reduce((a, v, k) => a + v * S[k][j], 0)))
    close(S2.flat(), S.flat(), 1e-12)
    expect(S.reduce((a, row, i) => a + row[i], 0)).toBeCloseTo(3, 12)
  })
})

describe('linearSmootherCv', () => {
  it('Nadaraya-Watson: leave-one-out by formula equals leave-one-out by refitting', () => {
    const c = F.nadarayaWatson
    const cv = linearSmootherCv(tensor(c.smoother), tensor(c.y))
    close(toFlat(cv.looResiduals), c.looResiduals, 1e-10)
    expect(cv.loo).toBeCloseTo(c.loo, 12)
    expect(cv.gcv).toBeCloseTo(c.gcv, 12)
    expect(cv.df).toBeCloseTo(c.df, 12)
    close(
      toFlat(cv.leverage),
      c.smoother.map((r, i) => r[i]),
      1e-15,
    )
  })

  it('reports an infinite residual for a leverage of 1, and throws for a smoother of the wrong size', () => {
    const cv = linearSmootherCv(
      [
        [1, 0],
        [0, 0.5],
      ],
      [2, 4],
    )
    expect(toFlat(cv.looResiduals)[0]).toBeNaN()
    expect(toFlat(cv.looResiduals)[1]).toBe(4)
    expect(() => linearSmootherCv([[1]], [1, 2])).toThrow()
  })
})
