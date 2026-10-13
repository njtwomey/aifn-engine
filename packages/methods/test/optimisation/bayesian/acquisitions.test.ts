/**
 * The acquisition functions against their closed forms from scipy.stats.norm (`fixtures/gen/optimisation/bayesian.py`):
 * PI, EI and the confidence bound in both directions and at sd = 0, the GP-UCB schedule, max-value entropy search for
 * given samples, and the Gumbel fit of `maxValueSamples`.
 */
import { describe, expect, it } from 'vitest'
import {
  bayesianFunctions,
  expectedImprovement,
  maxValueEntropySearch,
  maxValueSamples,
  probabilityOfImprovement,
  ucbBeta,
  upperConfidenceBound,
} from 'aifn-methods/optimisation/bayesian'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { normalCdf } from 'aifn-compute/numerics/special'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type Fixture = {
  mu: number[]
  sd: number[]
  best: number
  acquisitions: { minimise: boolean; xi: number; pi: number[]; ei: number[] }[]
  bounds: { minimise: boolean; beta: number; ucb: number[] }[]
  betas: { t: number; candidates: number; delta: number; beta: number }[]
  entropy: { minimise: boolean; samples: number[]; mes: number[] }[]
  gumbel: {
    mu: number[]
    sd: number[]
    fits: { minimise: boolean; location: number; scale: number; quantiles: number[] }[]
  }
}

const F = fixture<Fixture>('optimisation/bayesian')
const mu = tensor(F.mu)
const sd = tensor(F.sd)

/** Every entry of `actual` within `tol` (absolute plus relative) of `expected`. */
const close = (actual: Tensor | number[], expected: number[], tol: number) => {
  const a = Array.isArray(actual) ? actual : toFlat(actual)
  expect(a.length).toBe(expected.length)
  a.forEach((v, i) =>
    expect(Math.abs(v - expected[i]), `entry ${i}: ${v} vs ${expected[i]}`).toBeLessThanOrEqual(
      tol * (1 + Math.abs(expected[i])),
    ),
  )
}

describe('probability and expected improvement', () => {
  it.each(F.acquisitions)('match scipy (minimise $minimise, xi $xi)', ({ minimise, xi, pi, ei }) => {
    close(probabilityOfImprovement(mu, sd, F.best, { xi, minimise }), pi, 1e-14)
    close(expectedImprovement(mu, sd, F.best, { xi, minimise }), ei, 1e-14)
  })

  it('take numbers to numbers and keep a tensor shape', () => {
    expect(typeof expectedImprovement(0, 1, 0)).toBe('number')
    expect(expectedImprovement(0, 1, 0)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 15)
    expect(probabilityOfImprovement(0.5, 2, 0)).toBeCloseTo(normalCdf(-0.25), 15)
    const m = tensor([
      [0, 1],
      [2, 3],
    ])
    expect(expectedImprovement(m, 1, 1).shape).toEqual([2, 2])
  })

  it('EI grows with the spread at a fixed mean and is never below the mean improvement', () => {
    const e = toFlat(expectedImprovement(tensor([0, 0, 0, 0]), tensor([0, 0.5, 1, 2]), 0.3))
    for (let i = 1; i < e.length; i++) expect(e[i]).toBeGreaterThan(e[i - 1])
    for (const v of e) expect(v).toBeGreaterThanOrEqual(0.3)
  })

  it('report bad inputs', () => {
    expect(() => expectedImprovement(mu, tensor([1, 2]), 0)).toThrow(ShapeError)
    expect(() => expectedImprovement(0, -1, 0)).toThrow(DomainError)
    expect(() => probabilityOfImprovement(0, 1, 0, { xi: -0.1 })).toThrow(DomainError)
  })
})

describe('upperConfidenceBound and ucbBeta', () => {
  it.each(F.bounds)('match the closed form (minimise $minimise, beta $beta)', ({ minimise, beta, ucb }) => {
    close(upperConfidenceBound(mu, sd, { beta, minimise }), ucb, 1e-15)
  })

  it.each(F.betas)('beta_t at t = $t over $candidates candidates', ({ t, candidates, delta, beta }) => {
    expect(ucbBeta(t, { candidates, delta })).toBeCloseTo(beta, 12)
  })

  it('reports bad inputs', () => {
    expect(() => upperConfidenceBound(0, 1, { beta: -1 })).toThrow(DomainError)
    expect(() => ucbBeta(0, { candidates: 10 })).toThrow(DomainError)
    expect(() => ucbBeta(1, { candidates: 10, delta: 1 })).toThrow(DomainError)
  })
})

describe('max-value entropy search', () => {
  it.each(F.entropy)('matches eq. 6 for given samples (minimise $minimise)', ({ minimise, samples, mes }) => {
    close(maxValueEntropySearch(mu, sd, { samples, minimise }), mes, 1e-13)
    close(maxValueEntropySearch(mu, sd, { samples: tensor(samples), minimise }), mes, 1e-13)
  })

  it.each(F.gumbel.fits)(
    'fits the Gumbel law at the quartiles (minimise $minimise)',
    ({ minimise, location, scale }) => {
      const fit = maxValueSamples(tensor(F.gumbel.mu), tensor(F.gumbel.sd), { count: 0, minimise })
      expect(fit.location).toBeCloseTo(location, 10)
      expect(fit.scale).toBeCloseTo(scale, 10)
    },
  )

  it('draws samples reproducibly by stream, clipped at the incumbent', () => {
    const m = tensor(F.gumbel.mu)
    const s = tensor(F.gumbel.sd)
    const a = maxValueSamples(m, s, { count: 200, stream: stream(3) })
    const b = maxValueSamples(m, s, { count: 200, stream: stream(3) })
    expect(toFlat(a.samples)).toEqual(toFlat(b.samples))
    // Samples of the minimum of f lie at or below the smallest mean, mostly; the median matches the Gumbel law's.
    const sorted = toFlat(a.samples).sort((x, y) => x - y)
    const median = -(a.location - a.scale * Math.log(Math.log(2)))
    expect(Math.abs(sorted[100] - median)).toBeLessThan(0.1)
    const clipped = toFlat(maxValueSamples(m, s, { count: 200, stream: stream(3), best: median }).samples)
    for (const v of clipped) expect(v).toBeLessThanOrEqual(median)
    expect(clipped.filter((v) => v === median).length).toBeGreaterThan(50)
  })

  it('is zero where sd = 0, non-negative elsewhere, and needs a stream to draw', () => {
    const v = toFlat(maxValueEntropySearch(mu, sd, { samples: 16, stream: stream(0) }))
    expect(v[7]).toBe(0)
    expect(v[8]).toBe(0)
    for (const x of v) expect(x).toBeGreaterThanOrEqual(0)
    expect(() => maxValueEntropySearch(mu, sd, { samples: 4 })).toThrow(DomainError)
    expect(() => maxValueEntropySearch(mu, sd, { samples: [] })).toThrow(DomainError)
  })

  it('with every sd zero, samples equal the best mean', () => {
    const fit = maxValueSamples(tensor([3, 1, 2]), 0, { count: 3, stream: stream(0) })
    expect(fit.scale).toBe(0)
    expect(toFlat(fit.samples)).toEqual([1, 1, 1])
  })
})

describe('registry', () => {
  it('registers the acquisitions as functions', () => {
    expectInfo(bayesianFunctions, 'function')
    expect(Object.keys(bayesianFunctions).sort()).toEqual(
      [
        'bayesianOptimisationStudy',
        'expectedImprovement',
        'expectedImprovementLogNormal',
        'localPenalisation',
        'maxValueEntropySearch',
        'multiPointExpectedImprovement',
        'maxValueSamples',
        'probabilityOfImprovement',
        'ucbBeta',
        'upperConfidenceBound',
      ].sort(),
    )
  })
})
