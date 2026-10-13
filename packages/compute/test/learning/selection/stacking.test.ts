/**
 * aifn-compute/learning/selection, model averaging and nested cross-validation: `stackingWeights` against a root of
 * the one-dimensional gradient (two models) and scipy's SLSQP (three models, one weight at 0), with arviz-stats'
 * `compare` as a looser check; `pseudoBmaWeights` against `compare` (`pseudo-bma`) and `pseudoBmaPlusWeights` against
 * numpy on fixed Dirichlet draws; `nestedCvInterval` against a direct numpy implementation of Bates, Hastie and
 * Tibshirani's Algorithm 1 for least squares on fixed fold assignments.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset, type Decides, type Estimator, type Supervised } from 'aifn-compute/learning/estimators'
import {
  nestedCvInterval,
  pseudoBmaPlusWeights,
  pseudoBmaWeights,
  stackingWeights,
} from 'aifn-compute/learning/selection'
import { solve } from 'aifn-compute/numerics/linalg'
import { fixture } from '../../fixtures'

interface Fixture {
  stacking: {
    two: { lpd: number[][]; weights: number[]; objective: number }
    three: {
      lpd: number[][]
      weights: number[]
      objective: number
      arviz: Record<'stacking' | 'pseudo-bma' | 'bb-pseudo-bma', number[]>
      pseudoBma: number[]
      elpd: number[]
      draws: number[][]
      pseudoBmaPlus: number[]
      pseudoBmaPlusSe: number[]
    }
  }
  nestedCv: {
    x: number[][]
    y: number[]
    foldIds: number[][]
    level: number
    a: number[]
    b: number[]
    ncv: number
    cv: number
    mseNested: number
    se: number
    bias: number
    estimate: number
    interval: [number, number]
    naiveSe: number
    naiveInterval: [number, number]
  }
}

const F = fixture<Fixture>('learning/selection')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
}

describe('stacking and pseudo-BMA weights', () => {
  it('two models: the interior optimum, a root of the gradient', () => {
    const { lpd, weights, objective } = F.stacking.two
    const s = stackingWeights(lpd)
    expect(s.converged).toBe(true)
    close(toFlat(s.weights), weights, 1e-12)
    expect(s.objective).toBeCloseTo(objective, 11)
  })

  it('three models: SLSQP to its tolerance, one model at weight 0, arviz-stats loosely', () => {
    const c = F.stacking.three
    const s = stackingWeights(c.lpd)
    expect(s.converged).toBe(true)
    const w = toFlat(s.weights)
    expect(w[1]).toBe(0)
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
    close(w, c.weights, 1e-7)
    // The optimum is at least as good as SLSQP's, and its optimality conditions hold.
    expect(s.objective).toBeGreaterThanOrEqual(c.objective - 1e-12)
    expect(s.objective - c.objective).toBeLessThan(1e-9)
    close(w, c.arviz.stacking, 1e-4)
  })

  it('pseudo-BMA matches arviz-stats compare; pseudo-BMA+ matches numpy on fixed draws', () => {
    const c = F.stacking.three
    const p = pseudoBmaWeights(c.lpd)
    close(toFlat(p.weights), c.arviz['pseudo-bma'], 1e-12)
    close(toFlat(p.elpd), c.elpd, 1e-13)
    const plus = pseudoBmaPlusWeights(c.lpd, { draws: c.draws })
    expect(plus.B).toBe(40)
    close(toFlat(plus.weights), c.pseudoBmaPlus, 1e-12)
    close(toFlat(plus.se), c.pseudoBmaPlusSe, 1e-12)
    const drawn = pseudoBmaPlusWeights(c.lpd, { B: 200, stream: stream('bb') })
    expect(toFlat(drawn.weights).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
    expect(toFlat(drawn.weights)).toEqual(toFlat(pseudoBmaPlusWeights(c.lpd, { B: 200, stream: stream('bb') }).weights))
    expect(() => pseudoBmaPlusWeights(c.lpd)).toThrow(/stream/)
  })
})

/** Least squares with an intercept, by the normal equations. */
const leastSquares: Estimator<Supervised<Tensor, Tensor>, Decides<Tensor, Tensor>> = {
  name: 'least squares',
  fit({ x, y }) {
    const rows = toRows(x).map((r) => [1, ...r])
    const d = rows[0].length
    const yv = toFlat(y)
    const a = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r[i] * r[j], 0)),
    )
    const b = Array.from({ length: d }, (_, i) => rows.reduce((s, r, k) => s + r[i] * yv[k], 0))
    const w = toFlat(solve(tensor(a), tensor(b)) as Tensor)
    return { decide: (z) => tensor(toRows(z).map((r) => r.reduce((s, v, j) => s + v * w[j + 1], w[0]))) }
  },
}

describe('nestedCvInterval', () => {
  const c = F.nestedCv
  const data = dataset(tensor(c.x), tensor(c.y))

  it('matches Algorithm 1 on fixed fold assignments', () => {
    const r = nestedCvInterval(leastSquares, data, { foldIds: c.foldIds, level: c.level })
    expect(r.folds).toBe(4)
    expect(r.repetitions).toBe(3)
    close(toFlat(r.a), c.a, 1e-10)
    close(toFlat(r.b), c.b, 1e-10)
    expect(r.ncv).toBeCloseTo(c.ncv, 11)
    expect(r.cv).toBeCloseTo(c.cv, 11)
    expect(r.mseNested).toBeCloseTo(c.mseNested, 11)
    expect(r.se).toBeCloseTo(c.se, 11)
    expect(r.mse).toBeCloseTo(c.se ** 2, 11)
    expect(r.bias).toBeCloseTo(c.bias, 11)
    expect(r.estimate).toBeCloseTo(c.estimate, 11)
    close(r.interval, c.interval, 1e-10)
    expect(r.naive.se).toBeCloseTo(c.naiveSe, 11)
    close(r.naive.interval, c.naiveInterval, 1e-10)
    expect(r.inflation).toBeGreaterThanOrEqual(1)
    expect(r.inflation).toBeLessThanOrEqual(2)
  })

  it('draws its folds from the stream, reproducibly', () => {
    const r = nestedCvInterval(leastSquares, data, { folds: 3, repetitions: 2, stream: stream('ncv') })
    const again = nestedCvInterval(leastSquares, data, { folds: 3, repetitions: 2, stream: stream('ncv') })
    expect(r.interval).toEqual(again.interval)
    expect(r.a.shape).toEqual([6])
    expect(r.interval[0]).toBeLessThan(r.estimate)
    expect(() => nestedCvInterval(leastSquares, data)).toThrow(/stream/)
    expect(() => nestedCvInterval(leastSquares, data, { folds: 2, stream: stream('x') })).toThrow(/folds/)
  })
})
