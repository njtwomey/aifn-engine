import { describe, expect, it } from 'vitest'
import { bayesianSignedRankTest, expectedMaxOfDraws } from 'aifn-compute/probability/tests'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'

type Num = number
type Fx = {
  comparison: {
    bayesianSignedRankTest: {
      d: Num[]
      s: Num
      z0: Num
      rope: Num
      weights: Num[][]
      theta: Num[][]
      pLeft: Num
      pRope: Num
      pRight: Num
      mean: Num[]
    }[]
    expectedMaxOfDraws: { scores: Num[]; b: Num[]; expected: Num[] }[]
  }
}
const F = fixture<Fx>('probability/tests').comparison

describe('bayesianSignedRankTest', () => {
  it.each(F.bayesianSignedRankTest.map((c, i) => [i, c] as const))(
    'case %s: the simplex points of baycomp on the same Dirichlet weights',
    (_, c) => {
      const r = bayesianSignedRankTest(c.d, { rope: c.rope, z0: c.z0, weights: c.weights })
      const theta = toFlat(r.samples)
      const want = c.theta.flat()
      for (let k = 0; k < want.length; k++) expect(theta[k]).toBeCloseTo(want[k], 14)
      expect([r.pLeft, r.pRope, r.pRight]).toEqual([c.pLeft, c.pRope, c.pRight])
    },
  )

  it.each(F.bayesianSignedRankTest.map((c, i) => [i, c] as const))(
    'case %s: the mean of the posterior draws is the exact posterior mean',
    (_, c) => {
      const S = 20000
      const r = bayesianSignedRankTest(c.d, { rope: c.rope, s: c.s, z0: c.z0, samples: S, stream: stream(11) })
      const draws = toRows(r.samples) as number[][]
      expect(r.samples.shape).toEqual([S, 3])
      for (let j = 0; j < 3; j++) {
        const column = draws.map((p) => p[j])
        const mean = column.reduce((a, v) => a + v, 0) / S
        const sd = Math.sqrt(column.reduce((a, v) => a + (v - mean) ** 2, 0) / (S - 1))
        expect(Math.abs(mean - c.mean[j])).toBeLessThan(4 * (sd / Math.sqrt(S)) + 1e-12)
      }
      for (const p of draws) {
        expect(p[0] + p[1] + p[2]).toBeCloseTo(1, 14)
        expect(Math.min(...p)).toBeGreaterThanOrEqual(-1e-15)
      }
      expect(r.pLeft + r.pRope + r.pRight).toBeCloseTo(1, 15)
      // The probabilities are the fractions of draws each region wins (ties to the first).
      const wins = [0, 0, 0]
      for (const p of draws) wins[p[0] >= p[1] && p[0] >= p[2] ? 0 : p[1] >= p[2] ? 1 : 2]++
      expect([r.pLeft, r.pRope, r.pRight]).toEqual(wins.map((w) => w / S))
      expect(r.rope).toEqual([-c.rope, c.rope])
      expect(r.n).toBe(c.d.length)
    },
  )

  it('has no rope mass without a rope, takes an asymmetric rope, and checks its arguments', () => {
    const d = F.bayesianSignedRankTest[0].d
    const none = bayesianSignedRankTest(d, { stream: stream(1), samples: 500 })
    expect(none.pRope).toBe(0)
    expect(
      toFlat(none.samples)
        .filter((_, k) => k % 3 === 1)
        .every((v) => Math.abs(v) < 1e-14),
    ).toBe(true)
    const sym = bayesianSignedRankTest(d, { rope: 0.01, stream: stream(2), samples: 500 })
    const ends = bayesianSignedRankTest(d, { rope: [-0.01, 0.01], stream: stream(2), samples: 500 })
    expect(toFlat(ends.samples)).toEqual(toFlat(sym.samples))
    // s = 0 drops the pseudo-observation: a clear winner on all-positive differences.
    const clear = bayesianSignedRankTest([0.05, 0.04, 0.06, 0.05, 0.07], { rope: 0.01, s: 0, stream: stream(3) })
    expect(clear.pRight).toBe(1)
    expect(() => bayesianSignedRankTest(d, { stream: stream(0), rope: [0.1, -0.1] })).toThrow(DomainError)
    expect(() => bayesianSignedRankTest(d, { stream: stream(0), s: -1 })).toThrow(DomainError)
    expect(() => bayesianSignedRankTest([], { stream: stream(0) })).toThrow(DomainError)
    expect(() => bayesianSignedRankTest(d, {})).toThrow(/stream/)
    expect(() => bayesianSignedRankTest(d, { weights: [[0.5, 0.5]] })).toThrow(/columns/)
  })
})

describe('expectedMaxOfDraws', () => {
  it.each(F.expectedMaxOfDraws.map((c) => [c.scores.length, c] as const))(
    '%s scores: Eq. 8 of Dodge et al.',
    (_, c) => {
      const curve = toFlat(expectedMaxOfDraws(c.scores, c.b))
      c.expected.forEach((e, k) => expect(curve[k]).toBeCloseTo(e, 14))
      expect(expectedMaxOfDraws(c.scores, 1)).toBeCloseTo(c.scores.reduce((a, v) => a + v, 0) / c.scores.length, 14)
      for (let k = 1; k < curve.length; k++) expect(curve[k]).toBeGreaterThanOrEqual(curve[k - 1] - 1e-15)
      expect(curve[curve.length - 1]).toBeLessThanOrEqual(Math.max(...c.scores))
    },
  )

  it('checks its arguments', () => {
    expect(() => expectedMaxOfDraws([], 2)).toThrow(DomainError)
    expect(() => expectedMaxOfDraws([0.5], 0)).toThrow(DomainError)
    expect(() => expectedMaxOfDraws([0.5], [1, 2.5])).toThrow(DomainError)
  })
})
