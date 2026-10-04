import { describe, expect, it } from 'vitest'
import { valueAndGrad, grad } from 'aifn-compute/foundation/autodiff'
import { Normal } from 'aifn-compute/probability/distributions'
import {
  binaryCrossEntropy,
  binaryCrossEntropyWithLogits,
  crammerSingerHinge,
  distillation,
  focalLoss,
  gaussianNll,
  getLoss,
  hinge,
  huber,
  jensenShannonLoss,
  klLoss,
  listLosses,
  logCosh,
  lossRegistry,
  meanAbsoluteErrorLoss,
  meanSquaredErrorLoss,
  oneHot,
  pinball,
  poissonNll,
  softmaxCrossEntropy,
  softmaxFocalLoss,
  surrogates,
  westonWatkinsHinge,
  type Loss,
} from 'aifn-compute/learning/losses'
import { tensor, toFlat, unwrap, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Case = Record<string, unknown> & { value: number; grad: unknown }
const F = fixture<Record<string, Case & Record<string, Case>>>('learning/losses') as Record<
  string,
  Record<string, unknown>
>
const T = (x: unknown) => tensor(x as number[][])
const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number])
const num = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/** Check a loss of the predictions against a fixture's value and gradient. */
function check(f: (x: Value) => Value, x: unknown, fc: unknown, tol = 1e-10) {
  const c = fc as { value: number; grad: unknown }
  const r = valueAndGrad(f)(T(x))
  expect(Math.abs(num(r.value as Value) - c.value)).toBeLessThan(tol * Math.max(1, Math.abs(c.value)))
  const g = toFlat(r.grad as Tensor)
  flat(c.grad).forEach((e, i) => expect(Math.abs(g[i] - e)).toBeLessThan(tol * Math.max(1, Math.abs(e))))
}

describe('classification losses match torch', () => {
  it('binary cross-entropy', () => {
    const c = F.bceLogits
    check((z) => binaryCrossEntropyWithLogits(z, c.y as number[]), c.z, c)
    const w = F.bceLogitsWeighted
    check((z) => binaryCrossEntropyWithLogits(z, w.y as number[], { positiveWeight: 3 }), w.z, w)
    const p = F.bceProbs
    check((q) => binaryCrossEntropy(q, p.y as number[]), p.p, p)
  })

  it('binary cross-entropy has a finite gradient at p = y ∈ {0, 1} (review: was NaN)', () => {
    // d/dp −log(1 − p) = 1/(1 − p) = 1 at p = 0; d/dp −log p = −1/p = −1 at p = 1.
    const r = valueAndGrad((q: Value) => binaryCrossEntropy(q, [0, 1], { reduction: 'sum' }))(tensor([0, 1]))
    expect(num(r.value as Value)).toBe(0)
    expect(Array.from(toFlat(r.grad as Tensor))).toEqual([1, -1])
  })

  it('softmax cross-entropy with label smoothing and soft targets', () => {
    const c = F.softmaxCe
    check((z) => softmaxCrossEntropy(z, c.labels as number[], { labelSmoothing: 0.1 }), c.logits, c)
    const s = F.softmaxCeSoft
    check((z) => softmaxCrossEntropy(z, T(s.targets)), s.logits, s)
  })

  it('focal loss', () => {
    const c = F.focal
    check((z) => focalLoss(z, c.y as number[]), c.z, c)
  })

  it('multiclass hinges', () => {
    const w = F.westonWatkins
    check((s) => westonWatkinsHinge(s, w.labels as number[]), w.scores, w)
    const c = F.crammerSinger
    check((s) => crammerSingerHinge(s, c.labels as number[]), c.scores, c)
  })

  it('softmax focal loss with γ = 0 is cross-entropy', () => {
    const z = tensor([
      [1, 2, -1],
      [0.5, 0, 3],
    ])
    expect(num(softmaxFocalLoss(z, [1, 2], { gamma: 0 }))).toBeCloseTo(num(softmaxCrossEntropy(z, [1, 2])), 12)
  })

  it('margin surrogates pass through (0, 1) and bound the 0–1 loss', () => {
    for (const name of ['hinge', 'logistic', 'exponential', 'squaredHinge'] as const) {
      expect(num(surrogates[name](0))).toBeCloseTo(1, 12)
      for (const m of [-2, -0.5, 0.3, 2]) expect(num(surrogates[name](m))).toBeGreaterThanOrEqual(m <= 0 ? 1 : 0)
    }
    expect(num(surrogates.modifiedHuber(-2))).toBe(8)
    expect(num(hinge(tensor([2, -0.5]), [1, 1]))).toBeCloseTo(0.75, 12)
  })
})

describe('regression losses match torch', () => {
  const c = F.regression
  const target = c.target as unknown as number[]
  it('squared, absolute, Huber, log-cosh and pinball', () => {
    check((p) => meanSquaredErrorLoss(p, target), c.pred, c.mse)
    check((p) => meanAbsoluteErrorLoss(p, target), c.pred, c.mae)
    check((p) => huber(p, target, { delta: 1.5 }), c.pred, c.huber)
    check((p) => logCosh(p, target), c.pred, c.logCosh)
    check((p) => pinball(p, target, { quantile: 0.8 }), c.pred, c.pinball)
  })

  it('Poisson and Gaussian negative log-likelihoods', () => {
    const p = F.poisson
    check((eta) => poissonNll(eta, p.counts as number[]), p.eta, p)
    const g = F.gaussian
    check((m) => gaussianNll(m, g.target as number[], T(g.sd)), g.mean, g)
  })

  it('log-cosh does not overflow', () => {
    expect(num(logCosh(1000, 0))).toBeCloseTo(1000 - Math.LN2, 10)
  })
})

describe('divergences', () => {
  it('distillation matches torch', () => {
    const c = F.distillation
    check((s) => distillation(s, T(c.teacher)), c.student, c)
  })

  it('KL between normals and Jensen–Shannon between probability vectors', () => {
    expect(num(klLoss(Normal(0, 1), Normal(1, 2)))).toBeCloseTo(Math.log(2) + 2 / 8 - 0.5, 12)
    const js = num(jensenShannonLoss(tensor([1, 0]), tensor([0, 1])))
    expect(js).toBeCloseTo(Math.LN2, 12)
    const g = grad((mu: Value) => klLoss(Normal(mu, 1), Normal(0, 1)))(0.7)
    expect(g).toBeCloseTo(0.7, 12)
  })
})

describe('metadata', () => {
  it('every loss is registered with its note', () => {
    const all = Object.values(lossRegistry) as Loss[]
    expect(all.length).toBeGreaterThanOrEqual(20)
    for (const l of all) {
      expect(l.info.kind).toBe('loss')
      expect(l.info.notes?.length, l.info.key).toBeGreaterThan(0)
      for (const n of l.info.notes ?? []) expect(n).toMatch(/^[a-z0-9-]+$/)
    }
    expect(listLosses({ family: 'regression' }).map((l) => l.info.key)).toContain('huber')
    expect(getLoss('huber')).toBe(huber)
    expect(() => getLoss('nope')).toThrow()
  })

  it('oneHot rejects labels outside the classes', () => {
    expect(toFlat(oneHot([2, 0], 3))).toEqual([0, 0, 1, 1, 0, 0])
    expect(() => oneHot([3], 3)).toThrow()
  })
})
