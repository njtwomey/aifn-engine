/**
 * The James–Stein estimator and its risk: the closed form against scipy quadrature of the noncentral chi-squared
 * density (fixture `theory/shrinkage`), and the simulation against the closed form.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { jamesStein, jamesSteinExactRisk, jamesSteinRisk, shrinkageFunctions } from 'aifn-methods/theory/shrinkage'
import { fixture } from '../fixtures'

type Fixture = {
  exact_risk: { p: number; target: 'fixed' | 'mean'; sigma2: number; distances: number[]; risk: number[] }[]
}
const fx = fixture<Fixture>('theory/shrinkage')

describe('jamesStein', () => {
  it('shrinks by 1 - (p - 2) sigma^2 / |x - t|^2 towards a fixed target', () => {
    const x = [1, 2, -1, 0.5, 3]
    const s = x.reduce((a, v) => a + v * v, 0)
    const js = jamesStein(x, { sigma2: 0.8 })
    expect(js.k).toBe(3)
    expect(js.factor).toBeCloseTo(1 - (3 * 0.8) / s, 14)
    js.estimate.forEach((v, i) => expect(v).toBeCloseTo(js.factor * x[i], 14))
    // A vector target, and a constant one.
    const t = [1, 1, 1, 1, 1]
    const towards = jamesStein(x, { target: t })
    const d = x.reduce((a, v) => a + (v - 1) ** 2, 0)
    towards.estimate.forEach((v, i) => expect(v).toBeCloseTo(1 + (1 - 3 / d) * (x[i] - 1), 14))
    expect(Array.from(jamesStein(x, { target: 1 }).estimate)).toEqual(Array.from(towards.estimate))
  })

  it('shrinks towards the grand mean with p - 3, and clips the factor with the positive part', () => {
    const x = [0.2, 0.5, -0.1, 0.4, 0.1, 0.3]
    const m = x.reduce((a, v) => a + v, 0) / 6
    const s = x.reduce((a, v) => a + (v - m) ** 2, 0)
    const js = jamesStein(x, { target: 'mean' })
    expect(js.k).toBe(3)
    expect(js.factor).toBeCloseTo(1 - 3 / s, 12)
    expect(js.factor).toBeLessThan(0)
    const pp = jamesStein(x, { target: 'mean', positivePart: true })
    expect(pp.factor).toBe(0)
    pp.estimate.forEach((v) => expect(v).toBeCloseTo(m, 14))
    // At the target itself the estimate is the target.
    expect(Array.from(jamesStein([2, 2, 2, 2], { target: 2 }).estimate)).toEqual([2, 2, 2, 2])
  })

  it('throws for too few dimensions, a bad variance or a target of the wrong length', () => {
    expect(() => jamesStein([1, 2])).toThrow(DomainError)
    expect(() => jamesStein([1, 2, 3], { target: 'mean' })).toThrow(DomainError)
    expect(() => jamesStein([1, 2, 3], { sigma2: 0 })).toThrow(DomainError)
    expect(() => jamesStein([1, 2, 3], { target: [1, 2] })).toThrow(ShapeError)
  })
})

describe('jamesSteinExactRisk', () => {
  it('matches scipy quadrature of E[1 / chi-squared]', () => {
    for (const c of fx.exact_risk) {
      c.distances.forEach((dist, i) => {
        const r = jamesSteinExactRisk(c.p, dist, { sigma2: c.sigma2, target: c.target })
        expect(Math.abs(r - c.risk[i]), `p ${c.p} ${c.target} at ${dist}`).toBeLessThan(1e-10 * c.risk[i])
      })
    }
  })

  it('is 2 sigma^2 at the target (3 sigma^2 towards the mean) and below p sigma^2 everywhere', () => {
    expect(jamesSteinExactRisk(10, 0)).toBeCloseTo(2, 14)
    expect(jamesSteinExactRisk(10, 0, { sigma2: 3 })).toBeCloseTo(6, 13)
    expect(jamesSteinExactRisk(10, 0, { target: 'mean' })).toBeCloseTo(3, 14)
    for (const d of [0.1, 1, 10, 100, 1000]) expect(jamesSteinExactRisk(5, d)).toBeLessThan(5)
    expect(() => jamesSteinExactRisk(2, 1)).toThrow(DomainError)
  })
})

describe('jamesSteinRisk', () => {
  it('agrees with the closed form within Monte Carlo error, with the positive part better still', () => {
    for (const target of ['zero', 'mean'] as const) {
      const r = jamesSteinRisk(stream(11), { p: 8, norms: [0, 1, 2.5, 5, 10], repeats: 20_000, target })
      for (let a = 0; a < r.norms.length; a++) {
        expect(Math.abs(r.jamesStein[a] - r.exact[a]), `${target} ${r.norms[a]}`).toBeLessThan(4 * r.se.jamesStein[a])
        expect(Math.abs(r.mle[a] - 8)).toBeLessThan(4 * r.se.mle[a])
        expect(r.positivePart[a]).toBeLessThanOrEqual(r.jamesStein[a] + 1e-12)
        expect(r.jamesStein[a]).toBeLessThan(r.mle[a])
      }
    }
    // At theta = 0 with sigma^2 = 1, the risk of James-Stein is 2.
    const zero = jamesSteinRisk(stream(12), { p: 10, norms: [0], repeats: 40_000 })
    expect(Math.abs(zero.jamesStein[0] - 2)).toBeLessThan(4 * zero.se.jamesStein[0])
  })

  it('uses the same noise at every distance, is reproducible, and has the default grid', () => {
    const a = jamesSteinRisk(stream(3), { repeats: 50 })
    const b = jamesSteinRisk(stream(3), { repeats: 50 })
    expect(Array.from(a.jamesStein)).toEqual(Array.from(b.jamesStein))
    expect(a.norms).toHaveLength(41)
    expect(a.norms[40]).toBeCloseTo(2 * Math.sqrt(10), 12)
    // Common random numbers: the MLE's loss does not depend on the mean, so its curve is flat.
    for (const v of a.mle) expect(v).toBeCloseTo(a.mle[0], 10)
    expect(() => jamesSteinRisk(stream(0), { p: 3, target: 'mean' })).toThrow(DomainError)
    expect(() => jamesSteinRisk(stream(0), { repeats: 0 })).toThrow(DomainError)
  })

  it('is registered', () => {
    expect(shrinkageFunctions.jamesStein).toBe(jamesStein)
    expect(shrinkageFunctions.jamesSteinRisk).toBe(jamesSteinRisk)
    expect(shrinkageFunctions.jamesSteinExactRisk).toBe(jamesSteinExactRisk)
  })
})
