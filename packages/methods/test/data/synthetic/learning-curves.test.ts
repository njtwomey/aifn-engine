import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { learningCurveAt, learningCurves } from 'aifn-methods/data/synthetic'

const flat = (x: Parameters<typeof toFlat>[0]) => Array.from(toFlat(x) as ArrayLike<number>)

describe('learningCurves', () => {
  it('is seeded, decreasing, and its forms give the clean curves', () => {
    for (const shape of ['power', 'exponential', 'mixed'] as const) {
      const a = learningCurves(stream(4), { n: 12, shape })
      const b = learningCurves(stream(4), { n: 12, shape })
      expect(flat(a.values)).toEqual(flat(b.values))
      expect(a.values.shape).toEqual([12, 81])
      const clean = toArray(a.clean) as number[][]
      const budgets = flat(a.budgets)
      a.forms.forEach((f, i) => {
        if (shape !== 'mixed') expect(f.shape).toBe(shape)
        budgets.forEach((t, k) => expect(clean[i][k]).toBeCloseTo(learningCurveAt(f, t), 14))
        for (let k = 1; k < budgets.length; k++) expect(clean[i][k]).toBeLessThan(clean[i][k - 1])
        // Start in [0.6, 1], asymptote in [0.05, 0.3], final between them.
        expect(clean[i][0]).toBeGreaterThanOrEqual(0.6 - 1e-12)
        expect(clean[i][0]).toBeLessThanOrEqual(1 + 1e-12)
        expect(f.a).toBeGreaterThanOrEqual(0.05)
        expect(f.a).toBeLessThanOrEqual(0.3)
        expect(flat(a.final)[i]).toBeGreaterThan(f.a)
      })
      expect(flat(a.final)[a.best]).toBe(Math.min(...flat(a.final)))
    }
    const mixed = learningCurves(stream(4), { n: 40, shape: 'mixed' })
    expect(new Set(mixed.forms.map((f) => f.shape))).toEqual(new Set(['power', 'exponential']))
  })

  it('the noise has the requested size and leaves the clean curves alone', () => {
    const c = learningCurves(stream(1), { n: 50, noise: 0.05 })
    const quiet = learningCurves(stream(1), { n: 50, noise: 0 })
    expect(flat(c.clean)).toEqual(flat(quiet.clean))
    expect(flat(quiet.values)).toEqual(flat(quiet.clean))
    const r = flat(c.values).map((v, k) => v - flat(c.clean)[k])
    const sd = Math.sqrt(r.reduce((a, v) => a + v * v, 0) / r.length)
    expect(sd).toBeGreaterThan(0.047)
    expect(sd).toBeLessThan(0.053)
  })

  it('crossing turns how a curve starts against how it ends', () => {
    const none = learningCurves(stream(2), { n: 40, crossing: 0, noise: 0 })
    const half = learningCurves(stream(2), { n: 40, crossing: 0.5, noise: 0 })
    const full = learningCurves(stream(2), { n: 40, crossing: 1, noise: 0 })
    expect(none.crossings).toBeLessThan(0.15)
    expect(half.crossings).toBeGreaterThan(none.crossings)
    expect(full.crossings).toBeGreaterThan(half.crossings)
    expect(full.crossings).toBeGreaterThan(0.6)
    // With full crossing the best at the end is in the worst quarter at the start.
    const start = (toArray(full.clean) as number[][]).map((row) => row[0])
    expect(start.filter((v) => v > start[full.best]).length).toBeLessThan(10)
  })

  it('reports bad options', () => {
    expect(() => learningCurves(stream(0), { budgets: [1, 1, 2] })).toThrow(/increasing/)
    expect(() => learningCurves(stream(0), { crossing: 2 })).toThrow(/crossing/)
    expect(() => learningCurves(stream(0), { noise: -1 })).toThrow(/noise/)
    expect(() => learningCurves(stream(0), { n: 1.5 })).toThrow(/integer/)
  })
})
