/**
 * Self-training on the worked example of the self-training note (seed 4): two labelled points at $-0.5$ and $2.5$ and
 * a thousand unlabelled draws from $\mathcal{N}(\mp 2, 1)$, with a logistic regression ($\ell_2 = 1$). The numbers
 * are those of the site's self-check.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { concat, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { blobs } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { accuracyOf, rowArgmax, selfTrainingSteps, type TrainingSet } from 'aifn-methods/learning/semi-supervised'

const d = blobs(stream(4), { n: 1000, centers: [[-2], [2]], sd: 1 })
const x = concat([tensor([[-0.5], [2.5]]), d.x], 0)
const truth = [0, 1, ...Array.from(toFlat(d.y))]
const labels = truth.map((v, i) => (i < 2 ? v : -1))
const problem = { x, labels, classes: 2, truth }

/** A logistic regression that records every model it fits. */
function recording() {
  const models: { weights: Tensor; intercept: Tensor }[] = []
  const base = {
    name: 'lr',
    fit(data: TrainingSet) {
      const m = logisticRegression({ l2: 1 }).fit(data)
      models.push(m as never)
      return m
    },
  }
  return { base, models }
}

describe('selfTrainingSteps', () => {
  it('starts at the midpoint of the two labels, proposes 254 / 6 and converges at round 19 with accuracy 0.989', () => {
    const { base, models } = recording()
    const t = trace(selfTrainingSteps(problem, { base, threshold: 0.9 }), undefined, 40, { keep: 'all' })
    const boundary = -toFlat(models[0].intercept)[0] / toFlat(models[0].weights)[0]
    expect(boundary).toBeCloseTo(1, 9)
    expect(t.steps[0].proposedPerClass).toEqual([254, 6])
    expect(t.final.t).toBe(19)
    expect(t.final.converged).toBe(true)
    expect(Math.abs(t.final.unlabelledAccuracy - 0.989)).toBeLessThanOrEqual(5e-4)
  })

  it('a threshold of 1 proposes nothing and stops after one refit', () => {
    const { base } = recording()
    const t = trace(selfTrainingSteps(problem, { base, threshold: 1 }), undefined, 10)
    expect(t.final.proposedCount).toBe(0)
    expect(t.final.t).toBe(1)
  })

  it('caps the pseudo-labels per class, and accumulating grows by the cap each round', () => {
    const { base } = recording()
    const capped = trace(selfTrainingSteps(problem, { base, threshold: 0.5, perClass: 5 }), undefined, 3)
    expect(capped.final.proposedPerClass.every((c) => c <= 5)).toBe(true)
    const acc = trace(
      selfTrainingSteps(problem, { base, threshold: 0.5, perClass: 5, keep: 'accumulate' }),
      undefined,
      3,
      { keep: 'all' },
    )
    expect(acc.steps.map((s) => s.proposedCount)).toEqual([10, 20, 30, 40])
  })

  it('refuses mismatched rows and a threshold outside [0, 1]', () => {
    const { base } = recording()
    expect(() => selfTrainingSteps({ ...problem, labels: [0, 1] }, { base })).toThrow(/rows/)
    expect(() => selfTrainingSteps(problem, { base, threshold: 1.5 })).toThrow(/threshold/)
  })
})

describe('row helpers', () => {
  it('rowArgmax breaks ties to the lower class and accuracyOf scores listed rows', () => {
    const { cls, conf } = rowArgmax([0.2, 0.8, 0.5, 0.5], 2)
    expect(Array.from(cls)).toEqual([1, 0])
    expect(Array.from(conf)).toEqual([0.8, 0.5])
    const p = Float64Array.from([0.9, 0.1, 0.3, 0.7, 0.6, 0.4])
    expect(accuracyOf(p, 2, [0, 1, 1])).toBeCloseTo(2 / 3, 12)
    expect(accuracyOf(p, 2, [0, 1, 1], [0, 1])).toBe(1)
    expect(accuracyOf(p, 2, [0, 1, 1], [])).toBeNaN()
  })
})
