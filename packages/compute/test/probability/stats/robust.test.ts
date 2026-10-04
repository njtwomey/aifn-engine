/**
 * The minimum covariance determinant against scikit-learn's MinCovDet (`fixtures/probability/stats.json`, key `mcd`): the raw
 * support, the corrected raw and reweighted estimates and the distances; and the laws (affine equivariance of the
 * distances, robustness to the planted outliers).
 */
import { describe, expect, it } from 'vitest'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { minimumCovarianceDeterminant, squaredMahalanobis } from 'aifn-compute/probability/stats'
import { fixture } from '../../fixtures'

type Case = {
  name: string
  x: number[][]
  location: number[]
  covariance: number[][]
  rawLocation: number[]
  rawCovariance: number[][]
  support: number[]
  distances: number[]
  rawLogDeterminant: number
}
const F = fixture<{ mcd: Case[] }>('probability/stats')

const close = (got: ArrayLike<number>, want: number[], tol: number) =>
  want.forEach((w, i) =>
    expect(Math.abs(got[i] - w), `${i}: ${got[i]} vs ${w}`).toBeLessThanOrEqual(tol * (1 + Math.abs(w))),
  )

describe('minimumCovarianceDeterminant', () => {
  for (const c of F.mcd)
    it(`matches scikit-learn's MinCovDet (${c.name})`, () => {
      const r = minimumCovarianceDeterminant(c.x)
      // Both search from random starts: aifn's support is at least as good (determinant no larger).
      expect(r.logDeterminant).toBeLessThanOrEqual(c.rawLogDeterminant + 1e-9)
      if (r.logDeterminant < c.rawLogDeterminant - 1e-9) return
      expect(Array.from(toFlat(r.support))).toEqual(c.support)
      close(toFlat(r.rawLocation), c.rawLocation, 1e-10)
      close(toFlat(r.rawCovariance), c.rawCovariance.flat(), 1e-10)
      close(toFlat(r.location), c.location, 1e-10)
      close(toFlat(r.covariance), c.covariance.flat(), 1e-10)
      close(toFlat(r.distances), c.distances, 1e-8)
    })

  it('flags the planted outliers and keeps them out of the support', () => {
    const c = F.mcd[0]
    const r = minimumCovarianceDeterminant(c.x)
    const support = new Set(toFlat(r.support))
    for (let i = 0; i < 8; i++) expect(support.has(i)).toBe(false)
    for (let i = 0; i < 8; i++) expect(toFlat(r.inliers)[i]).toBe(0)
  })

  it('squaredMahalanobis reproduces the distances of the reweighted estimate', () => {
    const c = F.mcd[1]
    const r = minimumCovarianceDeterminant(c.x)
    close(toFlat(squaredMahalanobis(c.x, r.location, r.covariance)), Array.from(toFlat(r.distances)), 1e-12)
  })
})
