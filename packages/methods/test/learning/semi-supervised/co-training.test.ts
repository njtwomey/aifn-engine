/**
 * Co-training and tri-training, with the numbers of the site's self-training and co-training self-check: each round
 * adds $2 \times (1 + 3)$ points from a pool kept at 75, the tri-training rule on the note's worked example, and
 * tri-training converging on two moons.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { moons, semiSupervisedSplit, twoViewDataset } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import {
  coTrainingSteps,
  majority,
  productOfExperts,
  triTrainingRule,
  triTrainingSteps,
} from 'aifn-methods/learning/semi-supervised'

describe('coTrainingSteps', () => {
  it('adds 1 + 3 points per view per round and refills the pool to 75', () => {
    const d = twoViewDataset(stream(5), { n: 200 })
    const y = toFlat(d.y)
    const split = semiSupervisedSplit(stream(6), y, 2)
    const t = trace(
      coTrainingSteps(
        { views: [d.a, d.b], labels: split.labels, classes: 2, truth: y },
        { base: logisticRegression({ l2: 1 }), perRound: [1, 3], pool: 75 },
      ),
      undefined,
      5,
      { keep: 'all' },
    )
    expect(t.steps.map((s) => s.added)).toEqual([0, 8, 16, 24, 32, 40])
    expect(t.steps.map((s) => s.inPool.reduce((a, b) => a + b, 0))).toEqual([75, 75, 75, 75, 75, 75])
    const last = t.final
    for (let i = 0; i < 200; i++)
      if (last.source[i] >= 2) expect(last.labelledAt[i]).toBeGreaterThan(0)
      else if (last.source[i] === 1) expect(last.labelledAt[i]).toBe(0)
  })

  it('refuses views of the wrong length', () => {
    const d = twoViewDataset(stream(5), { n: 20 })
    expect(() =>
      coTrainingSteps({ views: [d.a, d.b], labels: [0, 1], classes: 2 }, { base: logisticRegression() }),
    ).toThrow(/rows/)
  })
})

describe('triTrainingRule', () => {
  it('follows the tri-training note’s worked example', () => {
    expect(triTrainingRule(0.125, 0.5, 0, 60)).toEqual({ update: true, keep: 3, last: 1 })
    const refused = triTrainingRule(0.1, 0.125, 3, 70)
    expect(refused.update).toBe(false)
    expect(refused.last).toBe(3)
    const all = triTrainingRule(0.05, 0.125, 3, 5)
    expect(all.update && all.keep === 5).toBe(true)
    const zero = triTrainingRule(0, 0.5, 0, 40)
    expect(zero.update && zero.keep === 40).toBe(true)
  })
})

describe('triTrainingSteps', () => {
  it('converges on two moons with four labels per class', () => {
    const d = moons(stream(2), { n: 200, noise: 0.15, spacing: 'random' })
    const y = toFlat(d.y)
    const split = semiSupervisedSplit(stream(3), y, 4)
    const t = trace(
      triTrainingSteps({ x: d.x, labels: split.labels, classes: 2, truth: y }, { base: logisticRegression({ l2: 1 }) }),
      undefined,
      20,
    )
    expect(t.final.converged).toBe(true)
  })
})

describe('combining learners', () => {
  it('majority takes two of three and productOfExperts renormalises the product', () => {
    const v = majority([Int32Array.from([0, 1, 2]), Int32Array.from([1, 1, 0]), Int32Array.from([1, 0, 1])])
    expect(Array.from(v)).toEqual([1, 1, 2])
    const p = productOfExperts(Float64Array.from([0.6, 0.4, 0, 1]), Float64Array.from([0.5, 0.5, 1, 0]), 2)
    expect(p[0]).toBeCloseTo(0.6, 12)
    expect(Array.from(p.slice(2))).toEqual([0.5, 0.5])
  })
})
