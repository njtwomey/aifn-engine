import { describe, expect, it } from 'vitest'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { toArray, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { tuningTables } from 'aifn-methods/data/objectives'
import {
  averageRank,
  greedyPortfolio,
  leaveOneTaskOut,
  portfolioFunctions,
  portfolioLoss,
} from 'aifn-methods/optimisation/portfolio'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type Case = {
  table: number[][]
  k: number
  worst: number
  order: number[]
  loss: number[]
  gain: number[]
  perTask: number[][]
  optimal: number[]
  optimalLoss: number
  lotoOrder: number[][]
  lotoLoss: number[][]
  lotoMean: number[]
  rank: number[]
}
const F = fixture<{ random: Case; coarse: Case }>('optimisation/portfolio')
const flat = (t: Tensor) => Array.from(toFlat(t) as ArrayLike<number>)

describe('greedyPortfolio', () => {
  for (const name of ['random', 'coarse'] as const) {
    it(`matches the plain loop and the exhaustive search on the ${name} table`, () => {
      const c = F[name]
      const p = greedyPortfolio(c.table, c.k, { worst: c.worst })
      expect(p.order).toEqual(c.order)
      p.loss.forEach((v, i) => expect(v).toBeCloseTo(c.loss[i], 14))
      p.gain.forEach((v, i) => expect(v).toBeCloseTo(c.gain[i], 14))
      expect(toArray(p.perTask)).toEqual(c.perTask)
      expect(p.optimal).toEqual(c.optimal)
      expect(p.optimalLoss).toBeCloseTo(c.optimalLoss, 14)
      expect(p.worst).toBe(c.worst)
    })
  }

  it('gains do not increase, and the greedy reduction is at least 1 - 1/e of the optimum', () => {
    const t = tuningTables()
    const p = greedyPortfolio(t.table, 2, { worst: 1 })
    for (let i = 1; i < p.gain.length; i++) expect(p.gain[i]).toBeLessThanOrEqual(p.gain[i - 1] + 1e-15)
    expect(1 - p.loss[1]).toBeGreaterThanOrEqual((1 - 1 / Math.E) * (1 - (p.optimalLoss as number)))
    expect(p.loss[1]).toBeGreaterThanOrEqual(p.optimalLoss as number)
  })

  it('can be suboptimal, as in its example', () => {
    const table = [
      [0.35, 0.35, 0.35],
      [0.1, 0.6, 0.5],
      [0.5, 0.1, 0.6],
      [0.6, 0.5, 0.1],
    ]
    const p = greedyPortfolio(table, 2, { worst: 1 })
    expect(p.order).toEqual([0, 1])
    expect(p.optimal).toEqual([1, 2])
    expect(p.optimalLoss as number).toBeLessThan(p.loss[1])
  })

  it('skips the exhaustive search above its limit and defaults worst to the largest loss', () => {
    const c = F.random
    const p = greedyPortfolio(c.table, c.k, { exhaustive: 10 })
    expect(p.optimal).toBeUndefined()
    expect(p.worst).toBe(Math.max(...c.table.flat()))
    expect(() => greedyPortfolio(c.table, 0)).toThrow(DomainError)
    expect(() => greedyPortfolio(c.table, c.table.length + 1)).toThrow(DomainError)
    expect(() => greedyPortfolio([], 1)).toThrow(ShapeError)
    expect(() => greedyPortfolio([[NaN]], 1)).toThrow(DomainError)
  })
})

describe('portfolioLoss, leaveOneTaskOut and averageRank', () => {
  for (const name of ['random', 'coarse'] as const) {
    it(`match numpy on the ${name} table`, () => {
      const c = F[name]
      const r = portfolioLoss(c.table, c.optimal, { worst: c.worst })
      expect(r.loss).toBeCloseTo(c.optimalLoss, 14)
      const l = leaveOneTaskOut(c.table, c.k, { worst: c.worst })
      expect(l.order).toEqual(c.lotoOrder)
      flat(l.loss).forEach((v, i) => expect(v).toBeCloseTo(c.lotoLoss.flat()[i], 14))
      flat(l.mean).forEach((v, i) => expect(v).toBeCloseTo(c.lotoMean[i], 14))
      flat(averageRank(c.table).rank).forEach((v, i) => expect(v).toBeCloseTo(c.rank[i], 14))
    })
  }

  it('report bad input', () => {
    expect(() => portfolioLoss([[1, 2]], [1])).toThrow(DomainError)
    expect(portfolioLoss([[1, 2]], [], { worst: 3 }).loss).toBe(3)
    expect(() => leaveOneTaskOut([[1], [2]], 1)).toThrow(ShapeError)
    expect(averageRank([[1], [1], [0]]).order).toEqual([2, 0, 1])
  })
})

it('registers its functions', () => {
  expectInfo(portfolioFunctions, 'function')
  expect(Object.keys(portfolioFunctions).sort()).toEqual([
    'averageRank',
    'greedyPortfolio',
    'leaveOneTaskOut',
    'portfolioLoss',
  ])
})
