/**
 * Peaks over threshold against scipy.stats (`fixtures/probability/extremes.json`: genpareto.fit with floc = 0, tail
 * probabilities and quantiles of the fitted law, mean excesses) and laws (the exponential limit, inverse functions).
 */
import { describe, expect, it } from 'vitest'
import {
  fitGeneralisedPareto,
  meanExcess,
  peaksOverThreshold,
  tailProbability,
  tailQuantile,
} from 'aifn-compute/probability/extremes'
import { fixture } from '../../fixtures'

type Fit = { name: string; excesses: number[]; shape: number; scale: number; logLikelihood: number }
const F = fixture<{
  fits: Fit[]
  pot: {
    values: number[]
    threshold: number
    shape: number
    scale: number
    rate: number
    probes: number[]
    tailProbability: number[]
    risks: number[]
    tailQuantile: number[]
  }
  meanExcess: { thresholds: number[]; values: number[] }
}>('probability/extremes')

describe('fitGeneralisedPareto', () => {
  for (const f of F.fits)
    it(`matches scipy's maximum-likelihood fit (${f.name} tail)`, () => {
      const r = fitGeneralisedPareto(f.excesses)
      // scipy's optimiser stops near the maximum; the log-likelihood is at least as high and the parameters agree.
      expect(r.logLikelihood).toBeGreaterThanOrEqual(f.logLikelihood - 1e-6)
      expect(r.shape).toBeCloseTo(f.shape, 3)
      expect(r.scale).toBeCloseTo(f.scale, 3)
      expect(r.n).toBe(f.excesses.length)
    })

  it('fits ξ ≈ 0 and σ ≈ the mean to exponential quantiles', () => {
    const n = 2000
    const y = Array.from({ length: n }, (_, i) => -Math.log(1 - (i + 0.5) / n))
    const r = fitGeneralisedPareto(y)
    expect(Math.abs(r.shape)).toBeLessThan(0.05)
    expect(r.scale).toBeCloseTo(1, 1)
  })

  it('keeps ξ ≥ −1 where the likelihood is unbounded', () => {
    const r = fitGeneralisedPareto([2, 6, 2, 6])
    expect(r.shape).toBeGreaterThanOrEqual(-1)
    expect(Number.isFinite(r.logLikelihood)).toBe(true)
  })

  it('rejects non-positive excesses', () => {
    expect(() => fitGeneralisedPareto([1, 0, 2])).toThrow()
  })
})

describe('peaksOverThreshold', () => {
  const pot = peaksOverThreshold(F.pot.values, { quantile: 0.9 })
  it('takes the 0.9 quantile as threshold and the exceedance rate', () => {
    expect(pot.threshold).toBeCloseTo(F.pot.threshold, 12)
    expect(pot.rate).toBeCloseTo(F.pot.rate, 12)
    expect(pot.shape).toBeCloseTo(F.pot.shape, 3)
    expect(pot.scale).toBeCloseTo(F.pot.scale, 3)
  })
  it('tail probabilities and quantiles agree with scipy on its fitted law', () => {
    F.pot.probes.forEach((x, i) => expect(tailProbability(pot, x)).toBeCloseTo(F.pot.tailProbability[i], 3))
    F.pot.risks.forEach((q, i) => {
      const x = tailQuantile(pot, 1 - q)
      expect(Math.abs(x - F.pot.tailQuantile[i]) / F.pot.tailQuantile[i]).toBeLessThan(1e-2)
      // The quantile inverts the tail probability.
      expect(tailProbability(pot, x)).toBeCloseTo(q, 10)
    })
  })
})

describe('meanExcess', () => {
  it('matches the empirical mean of x − u above each u', () => {
    const e = meanExcess(F.pot.values, F.meanExcess.thresholds)
    F.meanExcess.values.forEach((v, i) => expect(e[i]).toBeCloseTo(v, 12))
  })
})
