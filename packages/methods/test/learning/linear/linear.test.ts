import { describe, expect, it } from 'vitest'
import { perceptron, perceptronSteps } from 'aifn-methods/learning/linear'
import { tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { expectProtocol } from '../../protocol'
import { close, fx, X2, Y2 } from '../shared'

describe('perceptron', () => {
  it('matches scikit-learn without shuffling', () => {
    const m = perceptron().fit(dataset(X2, Y2))
    close(m.weights, fx.perceptron.coef)
    expect(m.bias).toBeCloseTo(fx.perceptron.intercept, 10)
    expect(m.converged).toBe(true)
  })
  it('the averaged perceptron predicts with the mean of every step’s weights', () => {
    // A hand-run example: x = (1), (−1), labels +1, −1, no intercept. Step 1 updates w to 1; step 2 makes no mistake.
    const x = tensor([[1], [-1]])
    const steps = perceptronSteps({ x, y: tensor([1, -1]), intercept: false })
    let s = steps.init({}, undefined as never)
    s = steps.step(s, { t: 0 } as never)
    expect(s.averageWeights.data[0]).toBe(1)
    s = steps.step(s, { t: 1 } as never)
    expect(s.averageWeights.data[0]).toBe(1)
    // On X2/Y2 the average is the plain mean of the traced weights.
    const m = perceptron({ average: true, epochs: 3 }).fit(dataset(X2, Y2))
    expect(m.averaged).toBe(true)
    const all = m.training.steps
    const plain = perceptron({ epochs: 3 }).fit(dataset(X2, Y2))
    expect(plain.averaged).toBe(false)
    const t = all[all.length - 1].t
    expect(all.length).toBe(t + 1)
    for (const j of [0, 1]) {
      const mean = all.slice(1).reduce((a, st) => a + st.weights.data[j], 0) / t
      expect(m.weights.data[j]).toBeCloseTo(mean, 10)
    }
    const acc = (mm: typeof m) =>
      Array.from(mm.decide(X2).data).filter((c, i) => c === Number(Y2.data[i])).length / X2.shape[0]
    expect(acc(m)).toBeGreaterThan(0.5)
  })
  it('follows the trace protocol', () => {
    const signs = tensor(fx.y3.slice(0, 40).map((c) => (c ? 1 : -1)))
    expectProtocol(
      perceptronSteps({ x: X2, y: signs, shuffle: true }),
      {},
      {
        n: 60,
        record: { k: (s) => s.mistakes },
        seed: 'p',
      },
    )
  })
})
