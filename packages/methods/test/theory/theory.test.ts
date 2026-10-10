import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { biasVariance, biasVarianceSweep, fitAndPredict } from 'aifn-methods/theory/bias-variance'
import { empiricalRademacher, realisable, shatteringTable } from 'aifn-methods/theory/capacity'
import { concentrationStudy, mcdiarmidStudy, runningMeans, standardisedSums } from 'aifn-methods/theory/concentration'
import { doubleDescent } from 'aifn-methods/theory/double-descent'
import { drawTrainingSet, targetFunction, unitGrid } from 'aifn-methods/theory'

describe('bias–variance by resampling', () => {
  it('adds up to the Monte Carlo test error', () => {
    const model = { kind: 'polynomial', degree: 3 } as const
    const r = biasVariance(stream(1), { model, repeats: 300, noise: 0.3 })
    // A fresh estimate: new training sets, new noisy targets at uniform x.
    const f = targetFunction('sine')
    let err = 0
    const sets = 300
    for (let k = 0; k < sets; k++) {
      const train = drawTrainingSet(child(stream(2), 'set', k), 'sine', 30, 0.3)
      const test = drawTrainingSet(child(stream(3), 'test', k), 'sine', 50, 0.3)
      const pred = fitAndPredict(model, train.x, train.y, test.x)
      for (let i = 0; i < 50; i++) err += (pred[i] - test.y[i]) ** 2 / (50 * sets)
    }
    expect(r.totals.error).toBeCloseTo(err, 1)
    void f
  })
  it('trades bias for variance as complexity grows', () => {
    const sweep = biasVarianceSweep(stream(4), { family: 'polynomial', values: [0, 3, 12], repeats: 150 })
    expect(sweep.bias2[0]).toBeGreaterThan(sweep.bias2[1])
    expect(sweep.variance[2]).toBeGreaterThan(sweep.variance[1])
    expect(sweep.trainError[2]).toBeLessThan(sweep.trainError[0])
    const knn = biasVarianceSweep(stream(5), { family: 'knn', values: [1, 15], repeats: 100 })
    expect(knn.trainError[0]).toBeCloseTo(0, 12)
    expect(knn.variance[0]).toBeGreaterThan(knn.variance[1])
  })
})

describe('double descent', () => {
  it('interpolates past p = n, with a test-error peak and a weight-norm peak at the threshold', () => {
    const n = 30
    const r = doubleDescent(stream(6), { n, features: [5, n, 8 * n], repeats: 6 })
    expect(r.trainError[2]).toBeLessThan(1e-12)
    expect(r.trainError[1]).toBeLessThan(1e-8)
    expect(r.testErrorMedian[1]).toBeGreaterThan(2 * r.testErrorMedian[2])
    expect(r.testErrorMedian[1]).toBeGreaterThan(r.testErrorMedian[0])
    expect(r.weightNorm[1]).toBeGreaterThan(r.weightNorm[2])
    const ridge = doubleDescent(stream(6), { n, features: [n], repeats: 6, ridge: 1e-2 })
    expect(ridge.testErrorMedian[0]).toBeLessThan(r.testErrorMedian[1])
  })
})

describe('concentration', () => {
  it('keeps the simulated tails under every bound, with Chernoff the tightest exponential bound', () => {
    for (const law of ['bernoulli', 'uniform', 'arcsine'] as const) {
      const c = concentrationStudy(stream(7), { law, n: 40, trials: 20_000 })
      for (let i = 0; i < c.t.length; i++) {
        const slack = 3 * Math.sqrt(Math.max(c.empirical[i], 1e-4) / 20_000)
        expect(c.empirical[i]).toBeLessThanOrEqual(c.hoeffding[i] + slack)
        expect(c.empirical[i]).toBeLessThanOrEqual(c.chernoff[i] + slack)
        expect(c.empirical[i]).toBeLessThanOrEqual(c.bernstein[i] + slack)
        expect(c.chernoff[i]).toBeLessThanOrEqual(c.hoeffding[i] + 1e-12)
      }
    }
  })
  it('keeps the empty-bins tail under McDiarmid', () => {
    const m = mcdiarmidStudy(stream(8), { balls: 100, bins: 50, trials: 3000 })
    const mean = m.values.reduce((a, b) => a + b, 0) / m.values.length
    expect(mean).toBeCloseTo(m.expected, 2)
    for (let i = 0; i < m.t.length; i++) expect(m.empirical[i]).toBeLessThanOrEqual(m.bound[i] + 0.02)
  })
  it('shows the LLN and the CLT', () => {
    const paths = runningMeans(stream(9), 'exponential', 5000, 4)
    for (let r = 0; r < 4; r++) expect(Math.abs(paths[r * 5000 + 4999] - 1)).toBeLessThan(0.1)
    const z = standardisedSums(stream(10), 'uniform', 30, 20_000)
    const mean = z.reduce((a, b) => a + b, 0) / z.length
    const v = z.reduce((a, b) => a + (b - mean) ** 2, 0) / z.length
    expect(mean).toBeCloseTo(0, 1)
    expect(v).toBeCloseTo(1, 1)
  })
})

describe('capacity', () => {
  it('half-planes shatter 3 points in general position but not the 4 corners of a square', () => {
    expect(
      shatteringTable(
        [
          [0, 0],
          [1, 0],
          [0, 1],
        ],
        'half-planes',
      ).shattered,
    ).toBe(true)
    const square = shatteringTable(
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      'half-planes',
    )
    expect(square.count).toBe(14)
    expect(
      shatteringTable(
        [
          [0, 1],
          [1, 0],
          [0, -1],
          [-1, 0],
        ],
        'rectangles',
      ).shattered,
    ).toBe(true)
    expect(
      shatteringTable(
        [
          [0, 0],
          [1, 0],
        ],
        'intervals',
      ).shattered,
    ).toBe(true)
    expect(
      shatteringTable(
        [
          [0, 0],
          [1, 0],
          [2, 0],
        ],
        'intervals',
      ).count,
    ).toBe(7)
    // Intervals with 1-D points (n x 1 matrix)
    expect(realisable([[0], [1], [2]], [1, -1, 1], 'intervals')).toBe(false)
    expect(realisable([[0], [1], [2]], [1, 1, -1], 'intervals')).toBe(true)
    expect(shatteringTable([[0], [1], [2]], 'intervals').count).toBe(7)
  })
  it('unitGrid handles m = 1 cleanly', () => {
    expect(Array.from(unitGrid(1))).toEqual([0])
  })
  it('estimates Rademacher complexity: 1 for all labellings, 0 for one hypothesis', () => {
    const all = shatteringTable(
      [
        [0, 0],
        [1, 0],
        [0, 1],
      ],
      'half-planes',
    )
    expect(empiricalRademacher(stream(11), all.labellings, 500).estimate).toBeCloseTo(1, 12)
    const one = empiricalRademacher(stream(12), [all.labellings[0]], 4000)
    expect(Math.abs(one.estimate)).toBeLessThan(4 * one.standardError + 0.02)
  })
})
