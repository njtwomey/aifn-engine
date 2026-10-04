/**
 * `aifn-methods/timeseries` against direct numpy/scipy references (`fixtures/timeseries.json`, written by
 * `fixtures/gen/timeseries.py`): exact ARMA and seasonal ARIMA log-likelihoods as dense Gaussian densities, their
 * maximum-likelihood fits by scipy.optimize, the GARCH(1,1) likelihood and fit, and differencing.
 */
import { describe, expect, it } from 'vitest'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  armaLogLikelihood,
  difference,
  fitArma,
  fitGarch,
  fitSarima,
  garchLogLikelihood,
  sarimaLogLikelihood,
} from 'aifn-methods/timeseries'
import { fixture } from '../fixtures'

type ArmaCase = {
  x: number[]
  p: number
  q: number
  spec: { ar: number[]; ma: number[]; sigma: number; mean: number }
  logLikelihoodAtSpec: number
  fit: { ar: number[]; ma: number[]; logLikelihood: number; mean: number }
}
type SarimaCase = {
  x: number[]
  period: number
  spec: { ma: number[]; seasonalMa: number[]; sigma: number }
  logLikelihoodAtSpec: number
  fit: { ma: number; seasonalMa: number; logLikelihood: number }
}
type GarchCase = {
  r: number[]
  spec: { omega: number; alpha: number; beta: number; mean: number }
  logLikelihoodAtSpec: number
  fit: { omega: number; alpha: number; beta: number; mean: number; logLikelihood: number }
}
type DifferenceCase = { x: number[]; lag: number; order: number; value: number[] }
const F = fixture<{ arma: ArmaCase[]; sarima: SarimaCase[]; garch: GarchCase[]; difference: DifferenceCase[] }>(
  'timeseries',
)

describe('ARMA against scipy', () => {
  it.each(F.arma.map((c) => [`ARMA(${c.p}, ${c.q})`, c] as const))('%s: exact log-likelihood', (_, c) => {
    expect(armaLogLikelihood(c.x, c.spec).logLikelihood).toBeCloseTo(c.logLikelihoodAtSpec, 6)
  })

  it.each(F.arma.map((c) => [`ARMA(${c.p}, ${c.q})`, c] as const))('%s: maximum-likelihood fit', (_, c) => {
    const fit = fitArma(c.x, { p: c.p, q: c.q })
    expect(fit.converged).toBe(true)
    expect(fit.mean).toBeCloseTo(c.fit.mean, 12)
    toFlat(fit.ar).forEach((v, i) => expect(Math.abs(v - c.fit.ar[i])).toBeLessThan(2e-3))
    toFlat(fit.ma).forEach((v, i) => expect(Math.abs(v - c.fit.ma[i])).toBeLessThan(2e-3))
    expect(fit.logLikelihood).toBeGreaterThan(c.fit.logLikelihood - 1e-5)
    expect(fit.logLikelihood).toBeLessThan(c.fit.logLikelihood + 1e-5)
  })
})

describe('seasonal ARIMA against scipy', () => {
  const c = F.sarima[0]
  const airline = { diff: 1, seasonalDiff: 1, period: c.period }

  it('the airline model: exact log-likelihood of the differenced series', () => {
    expect(sarimaLogLikelihood(c.x, { ...airline, ...c.spec }).logLikelihood).toBeCloseTo(c.logLikelihoodAtSpec, 6)
  })

  it('the airline model: maximum-likelihood fit', () => {
    const fit = fitSarima(c.x, { p: 0, d: 1, q: 1, P: 0, D: 1, Q: 1, period: c.period })
    expect(fit.converged).toBe(true)
    expect(Math.abs(toFlat(fit.ma)[0] - c.fit.ma)).toBeLessThan(2e-3)
    expect(Math.abs(toFlat(fit.seasonalMa)[0] - c.fit.seasonalMa)).toBeLessThan(2e-3)
    expect(Math.abs(fit.logLikelihood - c.fit.logLikelihood)).toBeLessThan(1e-5)
    // The series was simulated with θ = −0.4, Θ = −0.6, σ = 0.5; the fit recovers them, and nudging a coefficient
    // lowers the likelihood.
    expect(Math.abs(toFlat(fit.ma)[0] + 0.4)).toBeLessThan(0.1)
    expect(Math.abs(toFlat(fit.seasonalMa)[0] + 0.6)).toBeLessThan(0.15)
    expect(Math.abs(Math.sqrt(fit.sigma2) - 0.5)).toBeLessThan(0.1)
    const at = (ma: number, sma: number) =>
      sarimaLogLikelihood(c.x, { ...airline, ma: [ma], seasonalMa: [sma] }).logLikelihood
    const [ma, sma] = [toFlat(fit.ma)[0], toFlat(fit.seasonalMa)[0]]
    expect(at(ma, sma)).toBeCloseTo(fit.logLikelihood, 8)
    for (const [a, b] of [
      [0.02, 0],
      [-0.02, 0],
      [0, 0.02],
      [0, -0.02],
    ])
      expect(at(ma + a, sma + b)).toBeLessThan(fit.logLikelihood)
  }, 20_000)
})

describe('GARCH(1, 1) against scipy', () => {
  const c = F.garch[0]
  it('log-likelihood (variance backcast at the sample variance)', () => {
    expect(garchLogLikelihood(c.r, c.spec).logLikelihood).toBeCloseTo(c.logLikelihoodAtSpec, 6)
  })
  it('maximum-likelihood fit', () => {
    const fit = fitGarch(c.r)
    expect(fit.mean).toBeCloseTo(c.fit.mean, 12)
    expect(Math.abs(fit.omega - c.fit.omega)).toBeLessThan(2e-3)
    expect(Math.abs(fit.alpha - c.fit.alpha)).toBeLessThan(2e-3)
    expect(Math.abs(fit.beta - c.fit.beta)).toBeLessThan(2e-3)
    expect(Math.abs(fit.logLikelihood - c.fit.logLikelihood)).toBeLessThan(1e-5)
  })
})

describe('differencing against numpy', () => {
  it.each(F.difference.map((c) => [`lag ${c.lag}, order ${c.order}`, c] as const))('%s', (_, c) => {
    const d = toFlat(difference(c.x, { lag: c.lag, order: c.order }))
    expect(d.length).toBe(c.value.length)
    d.forEach((v, i) => expect(v).toBeCloseTo(c.value[i], 12))
  })
})
