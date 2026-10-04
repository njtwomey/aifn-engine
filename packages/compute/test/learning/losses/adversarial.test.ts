import { describe, expect, it } from 'vitest'
import { grad, gradCheck, valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { stream, uniform, normals } from 'aifn-compute/foundation/random'
import {
  add,
  matmul,
  mul,
  reshape,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import {
  adversarialGames,
  binaryCrossEntropyWithLogits,
  contrastiveDivergenceLoss,
  discriminatorLoss,
  generatorLoss,
  getLoss,
  gradientPenalty,
} from 'aifn-compute/learning/losses'

const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}
const real = tensor([0.5, -1, 2])
const fake = tensor([-0.3, 0.8])
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length
const softplus = (z: number) => Math.log1p(Math.exp(z))

describe('adversarial losses', () => {
  it('are registered with the adversarial family', () => {
    for (const key of ['discriminatorLoss', 'generatorLoss', 'gradientPenalty'])
      expect(getLoss(key).info.family).toBe('adversarial')
    expect(getLoss('contrastiveDivergenceLoss').info.family).toBe('energy')
    expect(adversarialGames).toEqual(['minimax', 'non-saturating', 'wasserstein', 'hinge'])
  })

  it('the probability games score the discriminator by binary cross-entropy, real 1 and fake 0', () => {
    const bce = num(binaryCrossEntropyWithLogits(real, 1)) + num(binaryCrossEntropyWithLogits(fake, 0))
    expect(num(discriminatorLoss(real, fake, 'minimax'))).toBeCloseTo(bce, 12)
    expect(num(discriminatorLoss(real, fake, 'non-saturating'))).toBeCloseTo(bce, 12)
  })

  it('the Wasserstein and hinge critics match their formulas', () => {
    const r = [0.5, -1, 2]
    const f = [-0.3, 0.8]
    expect(num(discriminatorLoss(real, fake, 'wasserstein'))).toBeCloseTo(mean(f) - mean(r), 12)
    const hinge = mean(r.map((v) => Math.max(0, 1 - v))) + mean(f.map((v) => Math.max(0, 1 + v)))
    expect(num(discriminatorLoss(real, fake, 'hinge'))).toBeCloseTo(hinge, 12)
  })

  it('the generator losses: E log(1 − D), −E log D and −E f', () => {
    const f = [-0.3, 0.8]
    expect(num(generatorLoss(fake, 'minimax'))).toBeCloseTo(mean(f.map((z) => -softplus(z))), 12)
    expect(num(generatorLoss(fake, 'non-saturating'))).toBeCloseTo(mean(f.map((z) => softplus(-z))), 12)
    expect(num(generatorLoss(fake, 'wasserstein'))).toBeCloseTo(-mean(f), 12)
    // At D = 1/2 (logit 0) the minimax generator loss is log ½ and the non-saturating one log 2.
    expect(num(generatorLoss(tensor([0, 0]), 'minimax'))).toBeCloseTo(-Math.LN2, 12)
    expect(num(generatorLoss(tensor([0, 0]), 'non-saturating'))).toBeCloseTo(Math.LN2, 12)
  })

  it('the minimax gradient vanishes where the discriminator rejects fakes; the non-saturating one does not', () => {
    const confident = tensor([-8])
    const gMinimax = num(grad((z: Value) => generatorLoss(z, 'minimax'))(confident) as Value)
    const gNs = num(grad((z: Value) => generatorLoss(z, 'non-saturating'))(confident) as Value)
    expect(Math.abs(gMinimax)).toBeLessThan(1e-3)
    expect(Math.abs(gNs)).toBeGreaterThan(0.99)
  })

  it('law: at a point with densities p (data) and q (generator) the best logit is log(p/q), so D* = p/(p + q)', () => {
    for (const [p, q] of [
      [0.3, 0.1],
      [0.05, 0.4],
    ]) {
      // The expected discriminator loss at one point, p·softplus(−z) + q·softplus(z), from the losses' terms:
      // softplus(−z) is the non-saturating generator loss and softplus(z) minus the minimax one.
      const loss = (z: Value) => add(mul(p, generatorLoss(z, 'non-saturating')), mul(-q, generatorLoss(z, 'minimax')))
      const zStar = Math.log(p / q)
      expect(Math.abs(num(grad(loss)(tensor([zStar])) as Value))).toBeLessThan(1e-12)
      expect(1 / (1 + Math.exp(-zStar))).toBeCloseTo(p / (p + q), 12)
    }
  })

  it('the gradient penalty of a linear critic is (‖w‖ − 1)², with correct gradients in w', () => {
    const x = normals(stream(1), [5, 2])
    const y = normals(stream(2), [5, 2])
    const mix = uniform(stream(3), 0, 1, { shape: [5] }) as Tensor
    const w = tensor([[0.6], [1.2]])
    const critic = (wv: Value) => (pts: Value) => reshape(matmul(pts, wv), [5])
    const value = num(gradientPenalty(critic(w), x, y, mix))
    expect(value).toBeCloseTo((Math.hypot(0.6, 1.2) - 1) ** 2, 10)
    const report = gradCheck((wv: Tensor) => gradientPenalty(critic(wv), x, y, mix), w)
    expect(report.ok).toBe(true)
  })
})

describe('contrastiveDivergenceLoss', () => {
  it('is mean E(data) − mean E(samples), plus the energy penalty', () => {
    const pos = tensor([1, 2, 3])
    const neg = tensor([0.5, -0.5])
    expect(num(contrastiveDivergenceLoss(pos, neg))).toBeCloseTo(2, 12)
    expect(num(contrastiveDivergenceLoss(pos, neg, { regularisation: 0.5 }))).toBeCloseTo(2 + 0.5 * (14 / 3 + 0.25), 12)
  })

  it('law: for E_θ(x) = θx its gradient is mean(x_data) − mean(x_samples), the likelihood gradient', () => {
    const data = tensor([1, 2, 4])
    const samples = tensor([0, 3])
    const { grad: g } = valueAndGrad((theta: Value) =>
      contrastiveDivergenceLoss(
        matmul(reshape(data, [3, 1]), reshape(theta, [1, 1])),
        matmul(reshape(samples, [2, 1]), reshape(theta, [1, 1])),
      ),
    )(tensor([0.7]))
    expect(toFlat(g as Tensor)[0]).toBeCloseTo(7 / 3 - 1.5, 12)
  })
})
