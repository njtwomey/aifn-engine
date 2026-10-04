/**
 * CCA against a closed-form numpy reference (the SVD of the whitened cross-covariance, classical and ridge-regularised;
 * `fixtures/numerics/factorisation.json`) and scikit-learn's iterative `CCA` for the first canonical correlation; and
 * the laws: scores have unit variance, pairs are uncorrelated across components, and the score correlations equal the
 * canonical correlations.
 */
import { describe, expect, it } from 'vitest'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { canonicalCorrelation } from 'aifn-compute/numerics/factorisation'
import { fixture } from '../../fixtures'

type Fit = { rx: number; ry: number; A: number[][]; B: number[][]; correlations: number[] }
const F = fixture<{ cca: { x: number[][]; y: number[][]; fits: Fit[]; sklearnFirst: number } }>(
  'numerics/factorisation',
).cca

const close = (got: ArrayLike<number>, want: number[], tol: number) =>
  want.forEach((w, i) => expect(Math.abs(got[i] - w)).toBeLessThanOrEqual(tol * (1 + Math.abs(w))))

const column = (t: Tensor, c: number) => {
  const [n, r] = t.shape
  const v = toFlat(t)
  return Array.from({ length: n }, (_, i) => v[i * r + c])
}
const covariance = (a: number[], b: number[]) => {
  const n = a.length
  const ma = a.reduce((s, u) => s + u, 0) / n
  const mb = b.reduce((s, u) => s + u, 0) / n
  return a.reduce((s, u, i) => s + (u - ma) * (b[i] - mb), 0) / (n - 1)
}

describe('canonicalCorrelation', () => {
  for (const f of F.fits)
    it(`matches the closed form (r_x = ${f.rx}, r_y = ${f.ry})`, () => {
      const c = canonicalCorrelation(F.x, F.y, { regularisation: [f.rx, f.ry] })
      close(toFlat(c.correlations), f.correlations, 1e-9)
      close(toFlat(c.xWeights), f.A.flat(), 1e-8)
      close(toFlat(c.yWeights), f.B.flat(), 1e-8)
    })

  it("matches scikit-learn's first canonical correlation", () => {
    const c = canonicalCorrelation(F.x, F.y)
    expect(toFlat(c.correlations)[0]).toBeCloseTo(F.sklearnFirst, 8)
  })

  it('gives unit-variance, mutually uncorrelated scores whose pair correlations are the canonical ones', () => {
    const c = canonicalCorrelation(F.x, F.y)
    const U = c.transformX(F.x)
    const V = c.transformY(F.y)
    const rho = toFlat(c.correlations)
    for (let a = 0; a < rho.length; a++) {
      expect(covariance(column(U, a), column(U, a))).toBeCloseTo(1, 9)
      expect(covariance(column(V, a), column(V, a))).toBeCloseTo(1, 9)
      expect(covariance(column(U, a), column(V, a))).toBeCloseTo(rho[a], 9)
      for (let b = a + 1; b < rho.length; b++) {
        expect(covariance(column(U, a), column(U, b))).toBeCloseTo(0, 9)
        expect(covariance(column(U, a), column(V, b))).toBeCloseTo(0, 9)
      }
    }
  })

  it('refuses a singular block without regularisation and accepts it with', () => {
    const x = F.x.slice(0, 3)
    const y = F.y.slice(0, 3)
    expect(() => canonicalCorrelation(x, y)).toThrow(/singular/)
    expect(toFlat(canonicalCorrelation(x, y, { regularisation: 0.1 }).correlations).every(Number.isFinite)).toBe(true)
  })
})
