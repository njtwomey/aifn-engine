import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { int, oneOf, real, sample, space, when } from 'aifn-compute/foundation/space'
import { learningCurveAt, learningCurveSurface, learningCurves } from 'aifn-methods/data/synthetic'

const sp = space({ lr: real(1e-4, 1, { scale: 'log' }), dropout: real(0, 0.8), layers: int(1, 4) })

/** Spearman's rank correlation of two lists without ties. */
function spearman(a: number[], b: number[]): number {
  const rank = (v: number[]) => {
    const order = v.map((_, i) => i).sort((i, j) => v[i] - v[j])
    const r = new Array<number>(v.length)
    order.forEach((i, k) => (r[i] = k))
    return r
  }
  const [ra, rb] = [rank(a), rank(b)]
  const n = a.length
  const d2 = ra.reduce((s, r, i) => s + (r - rb[i]) ** 2, 0)
  return 1 - (6 * d2) / (n * (n * n - 1))
}

describe('learningCurveSurface', () => {
  const points = Array.from({ length: 60 }, (_, i) => sample(child(stream(9), i), sp))

  it('the optimum has final loss 0.05 and no configuration does better', () => {
    const s = learningCurveSurface(stream(0), { space: sp, noise: 0 })
    expect(s.optimum.loss).toBeCloseTo(0.05, 12)
    expect(s.final(s.optimum.point)).toBeCloseTo(0.05, 12)
    for (const p of points) {
      expect(s.final(p)).toBeGreaterThanOrEqual(0.05)
      expect(s.final(p)).toBeLessThanOrEqual(0.3)
    }
  })

  it('every curve runs from its start in [0.6, 1] at r to its final loss at R, decreasing, in both families', () => {
    for (const shape of ['power', 'exponential'] as const) {
      const s = learningCurveSurface(stream(1), { space: sp, noise: 0, shape, minBudget: 2, maxBudget: 50 })
      for (const p of points.slice(0, 10)) {
        const start = s.clean(p, 2)
        expect(start).toBeGreaterThanOrEqual(0.6)
        expect(start).toBeLessThanOrEqual(1)
        expect(s.clean(p, 50)).toBeCloseTo(s.final(p), 12)
        expect(s.clean(p, 10)).toBeLessThan(start)
        expect(s.clean(p, 10)).toBeGreaterThan(s.final(p))
        expect(learningCurveAt(s.form(p), 7)).toBe(s.clean(p, 7))
      }
    }
  })

  it('crossing 0 orders starts as finals; crossing 1 reverses them', () => {
    const finals = (k: number) => {
      const s = learningCurveSurface(stream(2), { space: sp, noise: 0, crossing: k })
      return { start: points.map((p) => s.clean(p, 1)), end: points.map((p) => s.final(p)) }
    }
    const calm = finals(0)
    expect(spearman(calm.start, calm.end)).toBeCloseTo(1, 12)
    const wild = finals(1)
    expect(spearman(wild.start, wild.end)).toBeCloseTo(-1, 12)
    // The final losses do not depend on the crossing.
    expect(wild.end).toEqual(calm.end)
  })

  it('noise: drawn from the stream given, the same stream giving the same value', () => {
    const s = learningCurveSurface(stream(3), { space: sp, noise: 0.02 })
    const p = points[0]
    const a = s.at(p, 9, child(stream(4), 'e'))
    expect(s.at(p, 9, child(stream(4), 'e'))).toBe(a)
    expect(Math.abs(a - s.clean(p, 9))).toBeLessThan(0.02 * 6)
    expect(a).not.toBe(s.clean(p, 9))
    expect(() => s.at(p, 9)).toThrow(/stream/)
    // As curves for multiFidelitySteps: configuration i at budget t sees the noise of child(stream, i, t).
    const c = s.curves(points, stream(5))
    expect(c.count).toBe(60)
    expect(c.at(3, 27)).toBe(s.at(points[3], 27, child(stream(5), 3, 27)))
    expect(c.clean(3, 27)).toBe(s.clean(points[3], 27))
    // The clean curves are the same whatever the noise.
    expect(learningCurveSurface(stream(3), { space: sp, noise: 0 }).final(p)).toBe(s.final(p))
  })

  it('a choice dimension: the optimum is attained at a category', () => {
    const cs = space({ x: real(0, 1), kind: oneOf(['a', 'b', 'c']) })
    const s = learningCurveSurface(stream(6), { space: cs, noise: 0 })
    expect(['a', 'b', 'c']).toContain(s.optimum.point.kind)
    expect(s.optimum.loss).toBeCloseTo(0.05, 12)
  })

  it('reports bad options', () => {
    expect(() => learningCurveSurface(stream(0), { space: sp, crossing: 2 })).toThrow(/crossing/)
    expect(() => learningCurveSurface(stream(0), { space: sp, noise: -1 })).toThrow(/noise/)
    expect(() => learningCurveSurface(stream(0), { space: sp, minBudget: 5, maxBudget: 5 })).toThrow(/budgets/)
    const conditional = space({ a: oneOf(['x', 'y']), b: real(0, 1, { when: when('a', 'x') }) })
    expect(() => learningCurveSurface(stream(0), { space: conditional })).toThrow(/conditional/)
    // learningCurves is unchanged and still available beside it.
    expect(learningCurves(stream(0), { n: 2 }).forms.length).toBe(2)
  })
})
