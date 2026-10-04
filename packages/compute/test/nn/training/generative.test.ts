import { describe, expect, it } from 'vitest'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import {
  fromData,
  mul,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { discriminatorLoss, generatorLoss } from 'aifn-compute/learning/losses'
import { adversarialTraining, contrastiveDivergence } from 'aifn-compute/nn/training'
import { adamRule, sgdRule, type RuleState, type UpdateRule } from 'aifn-compute/optim/first-order'

const first = (t: Value) => toFlat(t as Tensor)[0]

describe('adversarialTraining', () => {
  // A one-parameter generator G(z) = θ (a point mass) against a logistic critic D(x) = a·x + b, real data at 2.
  const realPoints = tensor([2, 2, 2, 2])
  const options = (criticSteps: number) =>
    adversarialTraining<Tensor[], Tensor[]>({
      criticSteps,
      criticOptimizer: sgdRule({ stepSize: 0.1 }) as UpdateRule<unknown>,
      generatorOptimizer: sgdRule({ stepSize: 0.05 }) as UpdateRule<unknown>,
      criticLoss: ([a, b], [theta]) => {
        const score = (x: Value) => sub(mul(a, x), mul(-1, b))
        return discriminatorLoss(score(realPoints), score(mul(theta, tensor([1, 1, 1, 1]))), 'non-saturating')
      },
      generatorLoss: ([theta], [a, b]) =>
        generatorLoss(sub(mul(a, mul(theta, tensor([1, 1, 1, 1]))), mul(-1, b)), 'non-saturating'),
    })

  it('makes k critic updates per generator update, deterministically', () => {
    const start = { generator: [tensor([0])], critic: [tensor([0]), tensor([0])] }
    const s = run(options(3), start, 2)
    expect((s.criticOptimizer as RuleState).t).toBe(6)
    expect((s.generatorOptimizer as RuleState).t).toBe(2)
    expect(first(run(options(3), start, 2).generator[0])).toBe(first(s.generator[0]))
  })

  it('moves the generator towards the data', () => {
    const s = run(options(1), { generator: [tensor([0])], critic: [tensor([0]), tensor([0])] }, 300)
    expect(Math.abs(first(s.generator[0]) - 2)).toBeLessThan(0.6)
    expect(s.diverged).toBe(false)
  })
})

describe('contrastiveDivergence', () => {
  // E_θ(x) = ½(x − θ)²: p_θ = N(θ, 1). Data from N(3, 1); the maximum-likelihood θ is the data mean.
  const N = 400
  const s0 = stream('data')
  const data = fromData(
    Float64Array.from({ length: N }, () => 3 + normal(s0)),
    [N, 1],
  )
  const mean = toFlat(data).reduce((a, b) => a + b, 0) / N
  const energy = ([theta]: Tensor[], x: Value) => mul(0.5, sum(square(sub(x, theta)), -1))
  const fresh = (s: ReturnType<typeof stream>, n: number) => uniform(s, -6, 6, { shape: [n, 1] }) as Tensor

  it('law: learns the mean of a Gaussian energy, with persistent Langevin negatives', () => {
    const alg = contrastiveDivergence<Tensor[], { x: Tensor }>({
      energy,
      data: { x: data },
      batchSize: 64,
      optimizer: adamRule({ stepSize: 0.05 }) as UpdateRule<unknown>,
      sampler: { steps: 10, stepSize: 0.1, fresh },
      bufferSize: 200,
    })
    const s = run(alg, { params: [tensor([0])] }, 300, { stream: stream(2) })
    expect(Math.abs(first(s.params[0]) - mean)).toBeLessThan(0.3)
    expect(Number.isFinite(s.generativeLoss)).toBe(true)
    expect(Number.isNaN(s.supervisedLoss)).toBe(true)
  })

  it('with generative weight 0 trains the supervised term alone and draws no negatives', () => {
    const alg = contrastiveDivergence<Tensor[], { x: Tensor }>({
      energy,
      data: { x: data },
      generativeWeight: 0,
      sampler: { fresh },
      supervised: ([theta], batch) => mul(0.5, sum(square(sub(batch.x, theta)))),
      optimizer: adamRule({ stepSize: 0.1 }) as UpdateRule<unknown>,
    })
    const s = run(alg, { params: [tensor([0])] }, 200, { stream: stream(2) })
    expect(Math.abs(first(s.params[0]) - mean)).toBeLessThan(0.3)
    expect(s.negatives.shape[0]).toBe(0)
    expect(Number.isNaN(s.generativeLoss)).toBe(true)
  })
})
