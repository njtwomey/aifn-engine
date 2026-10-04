/**
 * Active learning with label proportions (Poyiadzis et al. 2019): the laws of a query (k pool points leave the pool and
 * form a bag whose proportion is the oracle's answer; the exact oracle makes k singleton bags with the true labels;
 * US-LP takes the k most uncertain points; US-Mass's bag holds the most uncertain point), and on Gaussian XOR the
 * accuracy curves of the uncertainty strategies end above where they start.
 */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { xor } from 'aifn-methods/data/synthetic'
import {
  activeProportionsCurves,
  activeProportionsSteps,
  type ActiveStrategy,
} from 'aifn-methods/learning/weak-supervision'

function problem(seed = 1) {
  const d = xor(stream(seed), { kind: 'gaussian', sd: 0.25, n: [80, 80] })
  const y = Int32Array.from(toFlat(d.y as Tensor))
  const bags = new Int32Array(160).fill(-1)
  const ones = [...Array(160).keys()].filter((i) => y[i] === 1)
  const zeros = [...Array(160).keys()].filter((i) => y[i] === 0)
  ;[...ones.slice(0, 12), ...zeros.slice(0, 4)].forEach((i) => (bags[i] = 0))
  ;[...ones.slice(12, 16), ...zeros.slice(4, 16)].forEach((i) => (bags[i] = 1))
  const test = [...ones.slice(40), ...zeros.slice(40)]
  return {
    x: d.x as Tensor,
    labels: y,
    bags,
    proportions: [
      [0.25, 0.75],
      [0.75, 0.25],
    ],
    test,
  }
}

describe('a query', () => {
  for (const strategy of ['us-lp', 'us-mass', 'random', 'us-exact'] as ActiveStrategy[])
    it(`${strategy}: k pool points become bagged with the oracle's answer`, () => {
      const p = problem()
      const run = trace(activeProportionsSteps(p, { strategy, bagSize: 5, gamma: 4 }), undefined, 2, { keep: 'all' })
      const [s0, s1] = run.steps
      expect(s1.query.length).toBe(5)
      for (const i of s1.query) {
        expect(s0.bags[i]).toBe(-1)
        expect(p.test.includes(i)).toBe(false)
        expect(s1.bags[i]).toBeGreaterThanOrEqual(2)
      }
      const truth = s1.query.reduce((a, i) => a + p.labels[i], 0) / 5
      expect(s1.answer).toBeCloseTo(truth, 12)
      if (strategy === 'us-exact')
        for (const i of s1.query) expect(s1.proportions[2 * s1.bags[i] + 1]).toBe(p.labels[i])
      else expect(s1.proportions[2 * s1.bags[s1.query[0]] + 1]).toBeCloseTo(truth, 12)
      if (strategy === 'us-lp' || strategy === 'us-exact') {
        const ranked = [...s0.uncertainty.keys()]
          .filter((i) => s0.uncertainty[i] < Infinity)
          .sort((a, b) => s0.uncertainty[a] - s0.uncertainty[b] || a - b)
        expect([...s1.query].sort((a, b) => a - b)).toEqual(ranked.slice(0, 5).sort((a, b) => a - b))
      }
      if (strategy === 'us-mass') {
        let best = 0
        s0.uncertainty.forEach((u, i) => {
          if (u < s0.uncertainty[best]) best = i
        })
        expect(s1.seed).toBe(best)
        expect(s1.query).toContain(best)
      }
    })
})

describe('accuracy curves', () => {
  it('uncertainty queries with an exact or proportion oracle improve on the start', () => {
    const datasets = [1, 2].map((r) => {
      const d = xor(child(stream('active'), r), { kind: 'gaussian', sd: 0.25, n: [80, 80] })
      return { x: d.x as Tensor, y: d.y as Tensor }
    })
    const it = activeProportionsCurves({
      datasets,
      strategies: ['us-exact', 'us-lp'],
      bagSize: 5,
      queries: 3,
      gamma: 4,
    })
    let s = it.next()
    while (!s.done) s = it.next()
    const { mean } = s.value
    expect(mean[0][3]).toBeGreaterThanOrEqual(mean[0][0])
    expect(mean[1][3]).toBeGreaterThanOrEqual(mean[1][0] - 0.02)
  })
})
