/**
 * aifn-methods/neural/self-supervised: the most informative linear predictive code (no site check; tests written from
 * the code: horizon 1 is the Levinson–Durbin AR fit, and an AR(1) series has $\rho^2 = \phi^{2k}$).
 */
import { describe, expect, it } from 'vitest'
import { applyPredictiveCode, codeCorrelation, linearPredictiveCode } from 'aifn-methods/neural/self-supervised'
import { NumericalError } from 'aifn-compute/foundation/errors'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { levinsonDurbin } from 'aifn-compute/numerics/linalg'

// Autocovariances of an AR(2) process x_t = 0.6 x_{t-1} - 0.3 x_{t-2} + e_t, by the Yule–Walker recursion.
const ar2 = (() => {
  const [p1, p2] = [0.6, -0.3]
  const r1 = p1 / (1 - p2)
  const r = [1, r1]
  for (let h = 2; h < 12; h++) r.push(p1 * r[h - 1] + p2 * r[h - 2])
  return r
})()

describe('linearPredictiveCode', () => {
  it('horizon 1 gives the AR coefficients of levinsonDurbin', () => {
    for (const L of [1, 2, 3, 5]) {
      const code = linearPredictiveCode(ar2, L, 1)
      const ar = Array.from(toFlat(levinsonDurbin(ar2, L).ar))
      ar.forEach((v, i) => expect(code.weights[i]).toBeCloseTo(v, 12))
    }
    // The AR(2) process itself: a window of 2 or more recovers (0.6, -0.3) and nothing else.
    const w = linearPredictiveCode(ar2, 4, 1).weights
    expect(Array.from(w).map((v) => Math.round(v * 1e10) / 1e10 + 0)).toEqual([0.6, -0.3, 0, 0])
  })

  it('an AR(1) series has ρ² = φ^{2k} and information −½ log(1 − ρ²)', () => {
    const phi = 0.8
    const acov = Array.from({ length: 10 }, (_, h) => phi ** h / (1 - phi * phi))
    for (const k of [1, 2, 4]) {
      const code = linearPredictiveCode(acov, 3, k)
      expect(code.r2).toBeCloseTo(phi ** (2 * k), 12)
      expect(code.information).toBeCloseTo(-0.5 * Math.log(1 - phi ** (2 * k)), 12)
      expect(code.weights[0]).toBeCloseTo(phi ** k, 12)
    }
  })

  it('its correlation with the series at the horizon is ρ', () => {
    const code = linearPredictiveCode(ar2, 3, 2)
    expect(codeCorrelation(code, ar2, 2)).toBeCloseTo(Math.sqrt(code.r2), 12)
  })

  it('throws on a singular window instead of returning zero weights', () => {
    // A constant series: every autocovariance equals the variance, so the window's matrix has rank 1.
    expect(() => linearPredictiveCode([1, 1, 1, 1, 1], 2, 1)).toThrow(NumericalError)
    expect(() => linearPredictiveCode(ar2, 0, 1)).toThrow(/at least 1/)
    expect(() => linearPredictiveCode([1, 0.5], 2, 1)).toThrow(/autocovariances/)
  })
})

describe('applyPredictiveCode', () => {
  it('is NaN until the window is full, then the weighted window', () => {
    const code = { weights: Float64Array.from([0.5, 0.25]), variance: 1, r2: 0.5, information: 0 }
    const c = applyPredictiveCode(code, [1, 2, 3, 4])
    expect(Number.isNaN(c[0])).toBe(true)
    expect(Array.from(c.slice(1))).toEqual([1.25, 2, 2.75])
  })
})
