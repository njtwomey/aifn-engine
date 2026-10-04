import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { aftModel, aftSurvival, coxPh, coxSurvival, harrellConcordance } from 'aifn-methods/learning/survival'
import { censoredSurvival } from 'aifn-methods/data/synthetic'
import { fixture } from '../../fixtures'

type Fit = { coefficients: number[]; se: number[]; loglik: number }
type Aft = { intercept: number; coefficients: number[]; scale: number; loglik: number }
type F = { x: number[][]; time: number[]; event: number[]; efron: Fit; breslow: Fit; weibull: Aft; lognormal: Aft }
const f = fixture<F>('learning/survival')
const x = fromData(Float64Array.from(f.x.flat()), [f.x.length, 2])

describe('Cox proportional hazards', () => {
  for (const ties of ['efron', 'breslow'] as const) {
    it(`matches ${ties === 'efron' ? 'lifelines' : 'statsmodels'} with ${ties} ties`, () => {
      const fit = coxPh(x, f.time, f.event, { ties })
      const ref = f[ties]
      expect(fit.converged).toBe(true)
      fit.coefficients.forEach((b, k) => expect(b).toBeCloseTo(ref.coefficients[k], 5))
      fit.standardErrors.forEach((s, k) => expect(s).toBeCloseTo(ref.se[k], 4))
      expect(fit.logPartialLikelihood).toBeCloseTo(ref.loglik, 5)
      for (let i = 1; i < fit.path.length; i++) expect(fit.path[i]).toBeGreaterThanOrEqual(fit.path[i - 1] - 1e-9)
    })
  }
  it('recovers the true log hazard ratios and gives monotone survival curves', () => {
    const d = censoredSurvival(stream(2), { n: 2000 })
    const fit = coxPh(d.x, d.time, d.event)
    expect(fit.coefficients[0]).toBeCloseTo(-0.7, 1)
    expect(fit.coefficients[1]).toBeCloseTo(0.5, 1)
    const s = coxSurvival(fit, [1, 0])
    for (let i = 1; i < s.length; i++) expect(s[i]).toBeLessThanOrEqual(s[i - 1])
    const risk = Float64Array.from({ length: 2000 }, (_, i) => {
      const r = toFlat(d.x)
      return r[2 * i] * fit.coefficients[0] + r[2 * i + 1] * fit.coefficients[1]
    })
    expect(harrellConcordance(d.time, d.event, risk)).toBeGreaterThan(0.6)
  })

  it('is unchanged by a location shift of a covariate, however large (review G2)', () => {
    // The partial likelihood ignores a constant added to xᵀβ; exp(xᵀβ) of x ≈ −1100 (β ≈ −0.7) used to overflow and the fit threw.
    const d = censoredSurvival(stream(3), { n: 300 })
    const x = toFlat(d.x)
    const shifted = fromData(
      Float64Array.from(x, (v, i) => (i % 2 === 0 ? v - 1100 : v)),
      [300, 2],
    )
    const a = coxPh(d.x, d.time, d.event)
    const b = coxPh(shifted, d.time, d.event)
    expect(b.converged).toBe(true)
    b.coefficients.forEach((v, k) => expect(v).toBeCloseTo(a.coefficients[k], 8))
    expect(b.logPartialLikelihood).toBeCloseTo(a.logPartialLikelihood, 8)
  })
})

describe('accelerated failure time', () => {
  for (const [family, ref] of [
    ['weibull', f.weibull],
    ['log-normal', f.lognormal],
  ] as const) {
    it(`matches lifelines' ${family} AFT`, () => {
      const fit = aftModel(x, f.time, f.event, { family })
      expect(fit.intercept).toBeCloseTo(ref.intercept, 4)
      fit.coefficients.forEach((b, k) => expect(b).toBeCloseTo(ref.coefficients[k], 4))
      expect(fit.scale).toBeCloseTo(ref.scale, 4)
      expect(fit.logLikelihood).toBeCloseTo(ref.loglik, 4)
      const s = aftSurvival(fit, [0, 0], [1, 5, 10, 20])
      for (let i = 1; i < s.length; i++) expect(s[i]).toBeLessThan(s[i - 1])
    })
  }
  it('recovers the Weibull PH truth as time ratios −β/k', () => {
    const d = censoredSurvival(stream(3), { n: 3000, shape: 2 })
    const fit = aftModel(d.x, d.time, d.event)
    expect(fit.coefficients[0]).toBeCloseTo(0.35, 1)
    expect(fit.coefficients[1]).toBeCloseTo(-0.25, 1)
    expect(fit.scale).toBeCloseTo(0.5, 1)
  })
})
