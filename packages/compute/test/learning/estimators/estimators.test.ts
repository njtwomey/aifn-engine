/**
 * aifn-compute/learning/estimators: predictives, mixins, capability types and guards, `evaluate` over registered metrics,
 * `dataset()` and row selection. Models are small hand-built objects; fitted models of the applications package are tested with
 * their areas.
 */
import { describe, expect, expectTypeOf, it } from 'vitest'
import type { Univariate } from 'aifn-compute/foundation/contracts'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  asTensor,
  bernoulliPredictive,
  capabilities,
  categoricalPredictive,
  classProbabilities,
  dataset,
  evaluate,
  expectation,
  gaussianPredictive,
  hasDecide,
  hasExpect,
  hasPredictive,
  hasSample,
  hasScore,
  hasTraining,
  isClassDistribution,
  isUnivariate,
  metricInput,
  outputs,
  readout,
  rowCount,
  score,
  takeData,
  takeRows,
  withDecision,
  withExpectation,
  withSampling,
  type Decides,
  type Fitted,
  type InputOf,
  type Predicts,
  type Scores,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import {
  accuracy,
  auroc,
  logLoss,
  meanAbsoluteError,
  meanSquaredError,
  metricRegistry,
} from 'aifn-compute/learning/metrics'

const close = (actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number) => {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tol)
}
const sigmoid = (v: number) => 1 / (1 + Math.exp(-v))

describe('predictives', () => {
  it('Gaussian: logProb, cdf and quantile', () => {
    const d = gaussianPredictive(tensor([0, 1]), tensor([1, 2]))
    close(
      toFlat(asTensor(d.logProb(tensor([0, 3])))),
      [-0.5 * Math.log(2 * Math.PI), -0.5 - Math.log(2) - 0.5 * Math.log(2 * Math.PI)],
      1e-14,
    )
    close(toFlat(asTensor(d.cdf!(tensor([0, 1])))), [0.5, 0.5], 1e-15)
    close(toFlat(asTensor(d.quantile!(tensor([0.5, 0.5])))), [0, 1], 1e-15)
    expect(isUnivariate(d)).toBe(true)
    expect(isClassDistribution(d)).toBe(false)
    expect(() => classProbabilities(d)).toThrow(/class/)
  })

  it('Bernoulli and categorical: probabilities, modes and class matrices', () => {
    const b = bernoulliPredictive(tensor([0.2, 0.7]))
    close(toFlat(asTensor(b.logProb(tensor([1, 0])))), [Math.log(0.2), Math.log(0.3)], 1e-14)
    expect(toFlat(asTensor(b.mode!()))).toEqual([0, 1])
    const pb = classProbabilities(b)
    expect(pb.shape).toEqual([2, 2])
    close(toFlat(pb), [0.8, 0.2, 0.3, 0.7], 1e-15)
    const c = categoricalPredictive(tensor([[0.1, 0.6, 0.3]]))
    close(toFlat(asTensor(c.logProb(tensor([2])))), [Math.log(0.3)], 1e-14)
    expect(toFlat(asTensor(c.mode!()))).toEqual([1])
    close(toFlat(asTensor(c.mean())), [1.2], 1e-14)
    expect(c.sample(stream('c'), { shape: [5] }).shape).toEqual([5, 1])
    close(toFlat(classProbabilities(c)), [0.1, 0.6, 0.3], 1e-15)
  })

  it('expectation: sums for classes, Gauss–Hermite quadrature otherwise', () => {
    const c = categoricalPredictive(tensor([[0.1, 0.6, 0.3]]))
    close(toFlat(expectation(c, (k) => k * k)), [0.6 + 1.2], 1e-14)
    const g = gaussianPredictive(tensor([1]), tensor([0.5]))
    // E[exp Y] = exp(μ + σ²/2) for Y ~ N(μ, σ²).
    close(toFlat(expectation(g, Math.exp)), [Math.exp(1 + 0.125)], 1e-10)
    // E[Y⁴] = μ⁴ + 6μ²σ² + 3σ⁴: exact for a polynomial of degree 4.
    close(toFlat(expectation(g, (y) => y ** 4)), [1 + 6 * 0.25 + 3 * 0.0625], 1e-12)
    close(toFlat(expectation(g)), [1], 0)
  })

  it('asTensor wraps numbers and passes tensors through', () => {
    expect(asTensor(2).shape).toEqual([])
    const t = tensor([1, 2])
    expect(asTensor(t)).toBe(t)
  })
})

describe('mixins', () => {
  // A score-only model: logits [N, 3] = x W for a fixed W [2, 3].
  const W = [
    [1, 0, -1],
    [0, 1, 1],
  ]
  const scorer: Scores<Tensor> & { W: number[][] } = {
    W,
    score: (x) => {
      const rows = toRows(x)
      return tensor(rows.map((r) => [0, 1, 2].map((k) => r[0] * W[0][k] + r[1] * W[1][k])))
    },
  }
  const x = tensor([
    [2, 0],
    [0, 2],
    [-1, 1],
  ])

  it('withDecision argmax from scores keeps the model fields', () => {
    const m = withDecision(scorer, 'argmax')
    expect(toFlat(m.decide(x))).toEqual([0, 1, 2])
    expect(m.W).toBe(W)
    expect(hasDecide(scorer)).toBe(false)
    expect(capabilities(m)).toEqual(['decide', 'score'])
    // One score column is ambiguous for argmax.
    const one = withDecision({ score: (v: Tensor) => v }, 'argmax')
    expect(() => one.decide(tensor([1, 2]))).toThrow(/threshold/)
  })

  const probModel: Predicts<Tensor, Univariate<Tensor>> = {
    predictive: (x) => bernoulliPredictive(fromData(Float64Array.from(toFlat(x)), [x.shape[0]])),
  }
  const p = tensor([0.1, 0.3, 0.6, 0.9])

  it('withDecision threshold, mode and cost matrix', () => {
    expect(toFlat(withDecision(probModel, { threshold: 0.5 }).decide(p))).toEqual([0, 0, 1, 1])
    expect(toFlat(withDecision(probModel, { threshold: 0.2 }).decide(p))).toEqual([0, 1, 1, 1])
    expect(toFlat(withDecision(probModel, 'mode').decide(p))).toEqual([0, 0, 1, 1])
    // Missing a positive costs 4 and a false alarm 1: decide 1 when 4p > 1 − p, i.e. p > 0.2.
    const costs = [
      [0, 1],
      [4, 0],
    ]
    expect(toFlat(withDecision(probModel, { costs }).decide(p))).toEqual([0, 1, 1, 1])
    expect(toFlat(withDecision(probModel, { costs: tensor(costs) }).decide(p))).toEqual([0, 1, 1, 1])
    expect(() =>
      withDecision(probModel, {
        costs: [
          [0, 1, 1],
          [1, 0, 1],
          [1, 1, 0],
        ],
      }).decide(p),
    ).toThrow(/classes/)
    expect(() => withDecision(probModel, { costs: [[0, 1], [1]] }).decide(p)).toThrow(/square/)
  })

  it('withExpectation and withSampling', () => {
    const m = withSampling(withExpectation(probModel))
    close(toFlat(m.expect(p)), toFlat(p), 0)
    close(
      toFlat(m.expect(p, (y) => 3 * y + 1)),
      toFlat(p).map((q) => 3 * q + 1),
      1e-15,
    )
    expect(m.sample(stream('b'), p, 10).shape).toEqual([10, 4])
    expect(toFlat(m.sample(stream('b'), p, 3))).toEqual(toFlat(m.sample(stream('b'), p, 3)))
    expect(hasExpect(m) && hasSample(m)).toBe(true)
  })

  it('readout completes a forward pass; capabilities follow the completers', () => {
    const latent: Fitted<Tensor, Tensor> & { weight: number } = {
      weight: 2,
      forward: (x) =>
        fromData(
          Float64Array.from(toFlat(x), (v) => 2 * v),
          [x.shape[0]],
        ),
    }
    const bern = readout(latent, (h: Tensor) =>
      bernoulliPredictive(fromData(Float64Array.from(toFlat(h), sigmoid), [h.shape[0]])),
    )
    close(toFlat(asTensor(bern.predictive(tensor([0])).params.probs)), [0.5], 1e-15)
    expect(bern.weight).toBe(2)
    expect(hasPredictive(bern)).toBe(true)
    expect(hasDecide(bern)).toBe(false)
    const both = readout(latent, {
      decide: (h: Tensor) => fromData(Int32Array.from(toFlat(h), (e) => (e > 0 ? 1 : 0))),
      score: (h: Tensor) => h,
    })
    expect(toFlat(both.decide(tensor([-1, 1])))).toEqual([0, 1])
    expect(hasScore(both)).toBe(true)
    expect(hasPredictive(both)).toBe(false)

    expectTypeOf(bern.predictive).returns.toEqualTypeOf<Univariate<Tensor>>()
    expectTypeOf(both.decide).parameter(0).toEqualTypeOf<Tensor>()
    // @ts-expect-error: a decide-and-score readout has no predictive.
    void both.predictive
  })
})

describe('capability types', () => {
  const X = tensor([[1], [2]])
  const Y = tensor([1, 0])

  it('a model without a capability cannot be evaluated with a metric that needs it', () => {
    const decider: Decides<Tensor, Tensor> = { decide: (x) => x }
    // @ts-expect-error: Decides has no predictive.
    void decider.predictive
    // @ts-expect-error: logLoss reads a predictive, which a decide-only model lacks.
    expect(() => evaluate(decider, { x: X, y: Y }, [logLoss])).toThrow(/predictive/)
    // @ts-expect-error: auroc reads scores.
    expect(() => evaluate(decider, { x: X, y: Y }, [accuracy, auroc])).toThrow(/score/)
    // A decide-only model serves accuracy.
    expect(evaluate(decider, { x: tensor([1, 2]), y: tensor([1, 3]) }, [accuracy])).toEqual({ accuracy: 0.5 })
  })

  it('mixins add exactly the declared capabilities', () => {
    const scorer: Scores<Tensor> = { score: (x) => x }
    const decided = withDecision(scorer, 'argmax')
    expectTypeOf(decided).toHaveProperty('decide')
    expectTypeOf(decided).not.toHaveProperty('predictive')
    // @ts-expect-error: withExpectation needs a predictive.
    expect(() => withExpectation(scorer).expect(X)).toThrow()
    expectTypeOf<InputOf<typeof decided>>().toEqualTypeOf<Tensor>()
  })

  it('guards narrow unknown models', () => {
    const models: unknown[] = [
      { predictive: (x: Tensor) => gaussianPredictive(x, x) },
      { decide: (x: Tensor) => x },
      { training: { steps: [] } },
    ]
    expect(models.filter((m) => hasPredictive(m)).length).toBe(1)
    expect(models.filter((m) => hasTraining(m)).length).toBe(1)
    const m = models[0]
    if (hasPredictive(m)) expect(m.predictive(tensor([1, 2, 3])).batchShape).toEqual([3])
    expect(capabilities(null)).toEqual([])
  })
})

describe('evaluate', () => {
  // A logistic model on one feature: P(y = 1 | x) = σ(2x − 1), with scores, decisions and a predictive.
  const x = tensor([-2, -1, 0, 0.5, 1, 2, 3, -0.5])
  const y = tensor([0, 0, 1, 0, 1, 1, 1, 0])
  const probs = (v: Tensor) =>
    fromData(
      Float64Array.from(toFlat(v), (e) => sigmoid(2 * e - 1)),
      [v.shape[0]],
    )
  const model = withDecision(
    {
      score: (v: Tensor) =>
        fromData(
          Float64Array.from(toFlat(v), (e) => 2 * e - 1),
          [v.shape[0]],
        ),
      predictive: (v: Tensor) => bernoulliPredictive(probs(v)),
    },
    { threshold: 0.5 },
  )

  it('serves each registered metric the output its capability names, keyed by info.key', () => {
    const r = evaluate(model, dataset(x, y), [accuracy, logLoss, auroc])
    expect(Object.keys(r).sort()).toEqual(['accuracy', 'auroc', 'logLoss'])
    const p = toFlat(probs(x))
    const t = toFlat(y)
    expect(r.accuracy).toBeCloseTo(
      accuracy(
        t,
        Array.from(p, (q) => (q >= 0.5 ? 1 : 0)),
      ),
      12,
    )
    expect(r.logLoss).toBeCloseTo(logLoss(t, p), 12)
    expect(r.auroc).toBeCloseTo(auroc(t, p), 12)
    // By hand: −mean log P(yᵢ).
    const ll = -Array.from(t).reduce((s, yi, i) => s + Math.log(yi ? p[i] : 1 - p[i]), 0) / t.length
    expect(r.logLoss).toBeCloseTo(ll, 12)
  })

  it('computes each capability once', () => {
    let calls = 0
    const counted = {
      decide: (v: Tensor) => {
        calls++
        return v
      },
    }
    const r = evaluate(counted, dataset(tensor([1, 2, 4]), tensor([1, 2, 3])), [
      accuracy,
      meanSquaredError,
      meanAbsoluteError,
    ])
    expect(calls).toBe(1)
    expect(r.meanSquaredError).toBeCloseTo(1 / 3, 12)
    expect(r.meanAbsoluteError).toBeCloseTo(1 / 3, 12)
    expect(r.accuracy).toBeCloseTo(2 / 3, 12)
  })

  it('metricInput reduces a Bernoulli predictive to P(y = 1) and keeps other inputs', () => {
    const d = bernoulliPredictive(tensor([0.2, 0.9]))
    close(toFlat(metricInput(logLoss, d) as Tensor), [0.2, 0.9], 0)
    const c = categoricalPredictive(tensor([[0.1, 0.6, 0.3]]))
    expect((metricInput(logLoss, c) as Tensor).shape).toEqual([1, 3])
    const s = tensor([1, 2])
    expect(metricInput(auroc, s)).toBe(s)
    const out = outputs(model, [accuracy, auroc], x)
    expect(Object.keys(out).sort()).toEqual(['decide', 'score'])
    expect(score([accuracy], y, out).accuracy).toBeGreaterThan(0.5)
    expect(() => score([logLoss], y, out)).toThrow(/predictive/)
  })

  it('every registered metric with a capability can be served by some capability', () => {
    const caps = new Set(Object.values(metricRegistry).map((m) => m.info.capability))
    for (const c of caps) if (c !== undefined) expect(['decide', 'score', 'predictive']).toContain(c)
  })
})

describe('datasets and rows', () => {
  it('dataset() builds a contract dataset with optional targets and extras', () => {
    const x = tensor([
      [1, 2],
      [3, 4],
    ])
    const d = dataset(x, tensor([0, 1]), {
      groups: ['a', 'b'],
      meta: { name: 'toy', description: 'Two rows.', task: 'classification' },
    })
    expect(d.kind).toBe('dataset')
    expect(d.x).toBe(x)
    expect(toFlat(d.y)).toEqual([0, 1])
    expect(d.groups).toEqual(['a', 'b'])
    expectTypeOf(d).toMatchTypeOf<Supervised<Tensor, Tensor>>()
    const u = dataset(x)
    expect(u.kind).toBe('dataset')
    expect('y' in u).toBe(false)
  })

  it('rowCount and takeRows on tensors, lists and tables', () => {
    const t = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    expect(rowCount(t)).toBe(3)
    expect(toRows(takeRows(t, [2, 0]))).toEqual([
      [5, 6],
      [1, 2],
    ])
    const ints = fromData(Int32Array.of(4, 5, 6), [3])
    expect(takeRows(ints, [1]).dtype).toBe('int32')
    expect(takeRows(['a', 'b', 'c'], [1, 1])).toEqual(['b', 'b'])
    const table = { age: tensor([30, 40, 50]), city: ['Cork', 'Paris', 'Oslo'] }
    expect(rowCount(table)).toBe(3)
    const sub = takeRows(table, [2])
    expect(toFlat(sub.age as Tensor)).toEqual([50])
    expect(sub.city).toEqual(['Oslo'])
    expect(() => rowCount({ a: [1], b: [1, 2] })).toThrow(/rows/)
    expect(() => rowCount(tensor(1))).toThrow(/scalar/)
    const data = takeData(dataset(t, tensor([0, 1, 2]), { groups: ['a', 'b', 'c'] }), [1])
    expect(toFlat(data.y)).toEqual([1])
    expect(data.groups).toEqual(['b'])
    expect(data.kind).toBe('dataset')
    expect(() => takeRows(t, [3])).toThrow()
  })
})
