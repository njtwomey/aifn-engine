import { describe, expect, it } from 'vitest'
import {
  armaAutocovariance,
  armaLogLikelihood,
  difference,
  expandSarima,
  fitSarima,
  forecastSarima,
  sarimaFitSteps,
  sarimaLogLikelihood,
  simulateSarima,
} from 'aifn-methods/timeseries'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { expectProtocol } from '../protocol'

/** The exact log N(x; μ1, Γ) with Γ the Toeplitz autocovariance, by a dense Cholesky factor. */
function denseLogLikelihood(x: number[], gamma: number[], mean = 0): number {
  const n = x.length
  const L = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = gamma[i - j]
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]
      L[i][j] = i === j ? Math.sqrt(s) : s / L[j][j]
    }
  const z = new Array<number>(n).fill(0)
  let logDet = 0
  let quad = 0
  for (let i = 0; i < n; i++) {
    let s = x[i] - mean
    for (let k = 0; k < i; k++) s -= L[i][k] * z[k]
    z[i] = s / L[i][i]
    logDet += Math.log(L[i][i])
    quad += z[i] * z[i]
  }
  return -0.5 * n * Math.log(2 * Math.PI) - logDet - 0.5 * quad
}

describe('seasonal ARIMA', () => {
  it('expands φ(z)Φ(zˢ) and θ(z)Θ(zˢ), and the integrated polynomial with (1 − z)(1 − zˢ)', () => {
    const e = expandSarima({ ar: [0.5], seasonalAr: [0.3], ma: [0.2], seasonalMa: [-0.6], period: 4 })
    // (1 − 0.5z)(1 − 0.3z⁴) = 1 − 0.5z − 0.3z⁴ + 0.15z⁵ → φ* = (0.5, 0, 0, 0.3, −0.15).
    toFlat(e.ar).forEach((v, i) => expect(v).toBeCloseTo([0.5, 0, 0, 0.3, -0.15][i], 14))
    // (1 + 0.2z)(1 − 0.6z⁴) = 1 + 0.2z − 0.6z⁴ − 0.12z⁵.
    toFlat(e.ma).forEach((v, i) => expect(v).toBeCloseTo([0.2, 0, 0, -0.6, -0.12][i], 14))
    // Airline model: (1 − z)(1 − z¹²) = 1 − z − z¹² + z¹³ → integrated (1, 0 … 0, 1, −1).
    const airline = toFlat(expandSarima({ diff: 1, seasonalDiff: 1, period: 12 }).integrated)
    expect(airline.length).toBe(13)
    expect([airline[0], airline[11], airline[12]]).toEqual([1, 1, -1])
    expect(airline.slice(1, 11).every((v) => v === 0)).toBe(true)
  })

  it('its exact likelihood is the dense Gaussian likelihood of the differenced series', () => {
    const model = { ar: [0.4], seasonalAr: [0.5], ma: [0.3], seasonalMa: [-0.4], period: 4, sigma: 1.3 }
    const sim = simulateSarima(stream('sarima-dense'), { ...model, diff: 1 }, 41)
    const x = toFlat(sim.x)
    const w = toFlat(difference(x))
    const e = expandSarima(model)
    const gamma = toFlat(armaAutocovariance({ ar: e.ar, ma: e.ma, sigma: 1.3 }, w.length))
    const want = denseLogLikelihood(w, gamma)
    expect(sarimaLogLikelihood(x, { ...model, diff: 1 }).logLikelihood).toBeCloseTo(want, 8)
    // Without seasonal terms it is the ARMA likelihood of ∇ᵈx.
    expect(sarimaLogLikelihood(x, { ar: [0.4], ma: [0.3], diff: 1, sigma: 1.3 }).logLikelihood).toBeCloseTo(
      armaLogLikelihood(w, { ar: [0.4], ma: [0.3], sigma: 1.3 }).logLikelihood,
      12,
    )
    // A non-stationary seasonal factor has no likelihood.
    expect(sarimaLogLikelihood(x, { seasonalAr: [1.1], period: 4 }).logLikelihood).toBe(-Infinity)
  })

  // The airline model's maximum-likelihood fit is checked against scipy on a 144-point series in reference.test.ts
  // (it replaced a 600-point recovery here that took about 3 s and timed out under parallel load).

  it('recovers a stationary seasonal AR with a mean, SARIMA(1,0,0)(1,0,0)₄', () => {
    const x = toFlat(
      simulateSarima(stream('sar'), { ar: [0.5], seasonalAr: [0.4], period: 4, mean: 3, sigma: 1 }, 800).x,
    )
    for (const method of ['exact', 'css'] as const) {
      const fit = fitSarima(x, { p: 1, q: 0, P: 1, Q: 0, period: 4, method })
      // Within about three standard errors, √((1 − φ²)/n) ≈ 0.03.
      expect(Math.abs(toFlat(fit.ar)[0] - 0.5)).toBeLessThan(0.1)
      expect(Math.abs(toFlat(fit.seasonalAr)[0] - 0.4)).toBeLessThan(0.1)
      expect(fit.mean).toBeCloseTo(3, 0)
      expect(fit.aic).toBeCloseTo(-2 * fit.logLikelihood + 2 * 4, 10)
    }
  }, 20_000)

  it('forecasts a seasonal random walk by repeating the last season, with se growing by whole seasons', () => {
    const x = [1, 5, 2, 8, 1.5, 5.5, 2.5, 7.5]
    const f = forecastSarima(x, { seasonalDiff: 1, period: 4, sigma: 2 }, 8)
    expect(toFlat(f.mean)).toEqual([1.5, 5.5, 2.5, 7.5, 1.5, 5.5, 2.5, 7.5])
    const se = toFlat(f.se)
    se.forEach((v, h) => expect(v).toBeCloseTo(2 * Math.sqrt(Math.floor(h / 4) + 1), 12))
  })

  it('rejects a seasonal term without a period and a mean with differencing', () => {
    expect(() => expandSarima({ seasonalAr: [0.5] })).toThrow(/period/)
    expect(() => sarimaLogLikelihood([1, 2, 3], { diff: 1, mean: 1 })).toThrow(/mean/)
  })

  it('its fitter follows the trace protocol', () => {
    const x = toFlat(simulateSarima(stream('sarima-protocol'), { ar: [0.3], seasonalMa: [0.5], period: 4 }, 200).x)
    for (const method of ['exact', 'css'] as const)
      expectProtocol(sarimaFitSteps(x, { p: 1, q: 0, Q: 1, period: 4, method }), undefined, {
        n: 8,
        record: { v: (s) => s.objective },
      })
  })
})
