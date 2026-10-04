import { describe, expect, test } from 'vitest'
import { burg, yuleWalker } from 'aifn-compute/signal/statistical'
import { levinsonDurbin } from 'aifn-compute/numerics/linalg'
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

describe('AR estimation', () => {
  test('Levinson–Durbin solves the Yule–Walker Toeplitz system', () => {
    const ld = levinsonDurbin(fx.acov, 3)
    close(toFlat(ld.ar), fx.yuleWalker.ar, 1e-10)
    expect(ld.singular).toBe(false)
    const yw = yuleWalker(fx.ar2, 3)
    close(toFlat(yw.ar), fx.yuleWalker.ar, 1e-10)
    expect(yw.sigma2).toBeCloseTo(fx.yuleWalker.sigma2, 10)
  })
  test('Burg matches the reference recursion and recovers the AR(2)', () => {
    const b = burg(fx.ar2, 3)
    close(toFlat(b.ar), fx.burg.ar, 1e-10)
    expect(b.sigma2).toBeCloseTo(fx.burg.sigma2, 10)
    const b2 = toFlat(burg(fx.ar2, 2).ar)
    expect(b2[0]).toBeCloseTo(0.6, 1)
    expect(b2[1]).toBeCloseTo(-0.3, 1)
  })
})
