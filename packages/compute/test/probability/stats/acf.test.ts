import { describe, expect, test } from 'vitest'
import { sampleAcf, samplePacf } from 'aifn-compute/probability/stats'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

// Golden values from numpy/statsmodels, generated with the Kalman checks (fixtures/gen/inference/filtering.py).
const fx = fixture<{
  ar2: number[]
  acf: number[]
  acov: number[]
  pacf: number[]
  yuleWalker: { ar: number[]; sigma2: number }
  burg: { ar: number[]; sigma2: number }
}>('inference/filtering')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-10) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
}

describe('sample autocorrelation', () => {
  test('sample ACF and PACF match numpy and Toeplitz solves', () => {
    const a = sampleAcf(fx.ar2, 12)
    close(toFlat(a.acf), fx.acf)
    expect(a.band).toBeCloseTo(1.959963984540054 / Math.sqrt(fx.ar2.length), 12)
    close(toFlat(samplePacf(fx.ar2, 10)), fx.pacf, 1e-9)
  })
})
