import { describe, expect, it } from 'vitest'
import { functionalAnova } from 'aifn-compute/learning/explain'
import { tensor, toArray, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fanova, fanovaReconstruct, importanceFunctions } from 'aifn-methods/optimisation/importance'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type Term = { axes: number[]; values: number[]; marginal: number[]; variance: number; fraction: number }
type Case = { table: number[][][]; weights: number[][]; mean: number; variance: number; terms: Term[] }
const F = fixture<{ three: Case; two: Case }>('optimisation/importance')

const flat = (t: Tensor) => Array.from(toFlat(t) as ArrayLike<number>)

describe('fanova', () => {
  for (const name of ['three', 'two'] as const) {
    it(`matches numpy on the ${name}-axis table, every term`, () => {
      const c = F[name]
      const d = name === 'three' ? 3 : 2
      const r = fanova(c.table as never, { weights: c.weights, order: d })
      expect(r.mean).toBeCloseTo(c.mean, 12)
      expect(r.variance).toBeCloseTo(c.variance, 12)
      expect(r.terms.map((t) => t.axes)).toEqual(c.terms.map((t) => t.axes))
      r.terms.forEach((t, k) => {
        const ref = c.terms[k]
        flat(t.values).forEach((v, i) => expect(v).toBeCloseTo(ref.values[i], 12))
        flat(t.marginal).forEach((v, i) => expect(v).toBeCloseTo(ref.marginal[i], 12))
        expect(t.variance).toBeCloseTo(ref.variance, 12)
        expect(t.fraction).toBeCloseTo(ref.fraction, 12)
      })
      expect(r.explained).toBeCloseTo(1, 12)
      expect(r.higher).toBeCloseTo(0, 12)
    })
  }

  it('sums back to the table, and its terms are orthogonal with zero weighted means', () => {
    const c = F.three
    const r = fanova(c.table as never, { weights: c.weights, order: 3 })
    const back = flat(fanovaReconstruct(r))
    flat(tensor(c.table as never)).forEach((v, i) => expect(back[i]).toBeCloseTo(v, 12))
    const W = r.weights
    // Expand each term to the full grid and check the weighted inner products.
    const full = r.terms.map((t) => {
      const one = { ...r, terms: [t], mean: 0 }
      return flat(fanovaReconstruct(one))
    })
    const w: number[] = []
    for (let a = 0; a < 4; a++)
      for (let b = 0; b < 3; b++) for (let e = 0; e < 5; e++) w.push(W[0][a] * W[1][b] * W[2][e])
    for (let i = 0; i < full.length; i++) {
      expect(full[i].reduce((s, v, k) => s + w[k] * v, 0)).toBeCloseTo(0, 12)
      for (let j = i + 1; j < full.length; j++)
        expect(full[i].reduce((s, v, k) => s + w[k] * v * full[j][k], 0)).toBeCloseTo(0, 12)
    }
    // The variances add up.
    expect(r.terms.reduce((s, t) => s + t.variance, 0)).toBeCloseTo(r.variance, 12)
  })

  it('agrees with compute functionalAnova (main effects and pairs) on the grid', () => {
    const c = F.three
    const model = (X: Tensor) => (toArray(X) as number[][]).map(([a, b, e]) => c.table[a][b][e])
    const ref = functionalAnova(
      model as never,
      [
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1, 2, 3, 4],
      ],
      { weights: c.weights },
    )
    const r = fanova(c.table as never, { weights: c.weights })
    expect(r.mean).toBeCloseTo(ref.mean, 12)
    r.main.forEach((t, j) => flat(t.values).forEach((v, i) => expect(v).toBeCloseTo(ref.main[j][i], 12)))
    r.pairs.forEach((t, p) => flat(t.values).forEach((v, i) => expect(v).toBeCloseTo(ref.pairs[p].values[i], 12)))
    expect(r.higher * r.variance).toBeCloseTo(ref.higherVariance, 12)
  })

  it('an additive table has no interactions; a constant one has no variance', () => {
    const v = [0, 1, 2]
    const r = fanova(v.map((a) => v.map((b) => a * a + 3 * b)))
    expect(r.pairs[0].fraction).toBeCloseTo(0, 14)
    expect(r.main[0].fraction + r.main[1].fraction).toBeCloseTo(1, 14)
    const k = fanova([
      [2, 2],
      [2, 2],
    ])
    expect(k.variance).toBe(0)
    expect(k.terms.every((t) => t.fraction === 0)).toBe(true)
  })

  it('reports bad input', () => {
    expect(() => fanova([1, 2, 3], { order: 2 })).toThrow(/order/)
    expect(() => fanova([1, NaN])).toThrow(/finite/)
    expect(() => fanova([1, 2], { weights: [[1, -1]] })).toThrow(/non-negative/)
    expect(() => fanova([1, 2], { weights: [[1, 1, 1]] })).toThrow(/weights/)
  })

  it('registers its functions', () => {
    expectInfo(importanceFunctions, 'function')
    expect(Object.keys(importanceFunctions).sort()).toEqual(['fanova', 'fanovaReconstruct'])
  })
})
