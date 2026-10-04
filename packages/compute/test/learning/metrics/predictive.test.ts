/** Probabilistic regression metrics through `evaluate`: the predictive capability path (Normal predictives). */
import { describe, expect, it } from 'vitest'
import { tensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { evaluate } from 'aifn-compute/learning/estimators'
import { crpsGaussian, gaussianLogScore, logLoss, logScore, meanSquaredError } from 'aifn-compute/learning/metrics'
import { Categorical, Normal } from 'aifn-compute/probability/distributions'

const x = tensor([0, 1, 2, 3])
const y = tensor([0.1, 1.3, 1.8, 3.4])
const sd = [0.5, 0.5, 1, 2]
/** A probabilistic regressor: predictive N(x, sdᵢ²) per case, and the mean as its decision. */
const model = {
  predictive: (v: Tensor) => Normal(v, tensor(sd)),
  decide: (v: Tensor) => v,
}

describe('the predictive path for regression metrics', () => {
  it('gaussianLogScore, crpsGaussian and logScore read a Normal predictive through evaluate', () => {
    const r = evaluate(model, { x, y }, [gaussianLogScore, crpsGaussian, logScore, meanSquaredError])
    const direct = { mean: x, sd }
    expect(r.gaussianLogScore).toBeCloseTo(gaussianLogScore(y, direct), 12)
    expect(r.crpsGaussian).toBeCloseTo(crpsGaussian(y, direct), 12)
    // The log score of a Normal predictive is the Gaussian log score.
    expect(r.logScore).toBeCloseTo(r.gaussianLogScore, 12)
    expect(r.meanSquaredError).toBeCloseTo((0.1 ** 2 + 0.3 ** 2 + 0.2 ** 2 + 0.4 ** 2) / 4, 12)
  })
  it('logScore on a categorical predictive is log loss', () => {
    const p = tensor([
      [0.7, 0.2, 0.1],
      [0.1, 0.8, 0.1],
      [0.3, 0.3, 0.4],
    ])
    const labels = [0, 1, 2]
    expect(logScore(labels, Categorical(p))).toBeCloseTo(logLoss(labels, p), 12)
  })
  it('a Gaussian score refuses a non-Normal predictive and points to logScore', () => {
    expect(() => gaussianLogScore(y, Categorical(tensor([[0.5, 0.5]])))).toThrow(/logScore/)
  })
})

describe('DCG building blocks', () => {
  it('gainFunction and positionDiscount reproduce dcg', async () => {
    const { dcg, gainFunction, positionDiscount } = await import('aifn-compute/learning/metrics')
    expect(gainFunction('linear')(3)).toBe(3)
    expect(gainFunction(undefined)(3)).toBe(7)
    expect(gainFunction((g) => g * g)(3)).toBe(9)
    expect(positionDiscount(0)).toBe(1)
    const grades = [3, 2, 0, 1]
    const byHand = grades.reduce((s, g, i) => s + gainFunction('exponential')(g) * positionDiscount(i), 0)
    expect(dcg([grades], [[4, 3, 2, 1]])).toBeCloseTo(byHand, 12)
  })
})
