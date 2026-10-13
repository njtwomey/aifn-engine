/**
 * The log-normal expected improvement and the batch acquisitions against scipy
 * (`fixtures/gen/optimisation/bayesian.py`): `expectedImprovementLogNormal` against quadrature of
 * $\expect[\max(f^+ - e^Y, 0)]$, `multiPointExpectedImprovement` against quadrature of the tail probability of the
 * best of $q = 2, 3$ Gaussian values (within four standard errors) and exactly against the expected improvement for
 * $q = 1$, and `localPenalisation` against its closed form.
 */
import { describe, expect, it } from 'vitest'
import {
  bayesianFunctions,
  expectedImprovement,
  expectedImprovementLogNormal,
  localPenalisation,
  multiPointExpectedImprovement,
} from 'aifn-methods/optimisation/bayesian'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type Fixture = {
  log_normal: { mu: number[]; sd: number[]; best: number; ei: number[] }
  qei: { mean: number[]; cov: number[][]; best: number; minimise: boolean; value: number }[]
  penalisation: {
    grid: number[]
    mean: number[]
    sd: number[]
    acq: number[]
    pending: number[]
    lipschitz: number
    minimum: number
    value: number[]
  }
}

const F = fixture<Fixture>('optimisation/bayesian')

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

describe('expectedImprovementLogNormal', () => {
  const { mu, sd, best, ei } = F.log_normal
  it('matches quadrature of the improvement of the cost', () => {
    close(expectedImprovementLogNormal(tensor(mu), tensor(sd), best), ei, 1e-9)
  })
  it('is the point-mass limit where sd = 0, and checks its incumbent', () => {
    expect(expectedImprovementLogNormal(Math.log(0.5), 0, 1)).toBeCloseTo(0.5, 15)
    expect(expectedImprovementLogNormal(Math.log(2), 0, 1)).toBe(0)
    expect(() => expectedImprovementLogNormal(0, 1, 0)).toThrow(DomainError)
  })
})

describe('multiPointExpectedImprovement', () => {
  it.each(F.qei)('matches quadrature within four standard errors (q = $mean.length, minimise $minimise)', (c) => {
    const est = multiPointExpectedImprovement(tensor(c.mean), tensor(c.cov), c.best, {
      samples: 40000,
      stream: stream(1),
      minimise: c.minimise,
    })
    expect(est.se).toBeGreaterThan(0)
    expect(est.se).toBeLessThan(0.01)
    expect(Math.abs(est.value - c.value)).toBeLessThan(4 * est.se + 1e-6)
  })
  it('is the closed-form expected improvement for q = 1, exactly', () => {
    for (const minimise of [true, false]) {
      const est = multiPointExpectedImprovement(tensor([0.3]), tensor([[0.49]]), 0.1, { minimise })
      expect(est.value).toBe(expectedImprovement(0.3, 0.7, 0.1, { minimise }))
      expect(est.se).toBe(0)
    }
  })
  it('is reproducible from its stream, and no more than one EI for two copies of one point', () => {
    const opts = { samples: 20000, stream: stream(3) }
    const a = multiPointExpectedImprovement(
      tensor([0, 0]),
      tensor([
        [1, 1],
        [1, 1],
      ]),
      0,
      opts,
    )
    const b = multiPointExpectedImprovement(
      tensor([0, 0]),
      tensor([
        [1, 1],
        [1, 1],
      ]),
      0,
      { ...opts, stream: stream(3) },
    )
    expect(a).toEqual(b)
    expect(Math.abs(a.value - expectedImprovement(0, 1, 0))).toBeLessThan(4 * a.se + 1e-6)
  })
  it('throws for a wrong covariance, too few samples or no stream', () => {
    expect(() => multiPointExpectedImprovement(tensor([0, 0]), tensor([[1]]), 0, { stream: stream(0) })).toThrow(
      ShapeError,
    )
    expect(() =>
      multiPointExpectedImprovement(
        tensor([0, 0]),
        tensor([
          [1, 0],
          [0, 1],
        ]),
        0,
      ),
    ).toThrow(DomainError)
    expect(() =>
      multiPointExpectedImprovement(
        tensor([0, 0]),
        tensor([
          [1, 0],
          [0, 1],
        ]),
        0,
        { samples: 1, stream: stream(0) },
      ),
    ).toThrow(DomainError)
    expect(() =>
      multiPointExpectedImprovement(
        tensor([0, 0]),
        tensor([
          [1, 2],
          [2, 1],
        ]),
        0,
        { stream: stream(0) },
      ),
    ).toThrow(DomainError)
  })
})

describe('localPenalisation', () => {
  const P = F.penalisation
  const opts = { mean: tensor(P.mean), sd: tensor(P.sd), lipschitz: P.lipschitz, minimum: P.minimum }
  it('matches the closed form of González et al. (2016)', () => {
    close(localPenalisation(tensor(P.acq), tensor(P.grid), tensor(P.pending), opts), P.value, 1e-14)
    // The same points as an [m, 1] grid and [p, 1] pending points.
    const grid = tensor(P.grid.map((g) => [g]))
    close(localPenalisation(tensor(P.acq), grid, tensor(P.pending.map((p) => [p])), opts), P.value, 1e-14)
  })
  it('leaves the acquisition without pending points, and soft-plus transforms a negative one', () => {
    close(localPenalisation(tensor(P.acq), tensor(P.grid), tensor([]), opts), P.acq, 0)
    const neg = P.acq.map((a) => a - 0.5)
    close(
      localPenalisation(tensor(neg), tensor(P.grid), tensor([]), opts),
      neg.map((a) => Math.log1p(Math.exp(a))),
      1e-15,
    )
  })
  it('defaults L to the largest slope of the mean and M to its minimum', () => {
    let L = 0
    for (let i = 0; i < P.grid.length; i++)
      for (let j = i + 1; j < P.grid.length; j++)
        L = Math.max(L, Math.abs(P.mean[i] - P.mean[j]) / Math.abs(P.grid[i] - P.grid[j]))
    const M = Math.min(...P.mean)
    const a = localPenalisation(tensor(P.acq), tensor(P.grid), tensor(P.pending), { mean: opts.mean, sd: opts.sd })
    const b = localPenalisation(tensor(P.acq), tensor(P.grid), tensor(P.pending), { ...opts, lipschitz: L, minimum: M })
    close(a, Array.from(toFlat(b)), 1e-14)
  })
  it('throws for mismatched sizes', () => {
    expect(() => localPenalisation(tensor([1, 2]), tensor(P.grid), tensor(P.pending), opts)).toThrow(ShapeError)
    expect(() => localPenalisation(tensor(P.acq), tensor(P.grid), tensor([[0.1, 0.2]]), opts)).toThrow(ShapeError)
  })
})

describe('registry', () => {
  it('registers the new functions', () => {
    expectInfo(bayesianFunctions, 'function')
    for (const key of [
      'expectedImprovementLogNormal',
      'multiPointExpectedImprovement',
      'localPenalisation',
      'bayesianOptimisationStudy',
    ])
      expect(bayesianFunctions[key]?.info.key).toBe(key)
  })
})
