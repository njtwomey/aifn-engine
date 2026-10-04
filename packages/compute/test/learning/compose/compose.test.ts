/**
 * aifn-compute/learning/compose: pipelines (capabilities of the last step, lifted through the transforms), column transformers
 * and target transforms with their pushforward predictives. The estimators are tiny hand-written ones, so that the
 * tests need no application.
 */
import { describe, expect, expectTypeOf, it } from 'vitest'
import type { Univariate } from 'aifn-compute/foundation/contracts'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { add, exp, mean, mul, sub, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  affineTarget,
  columns,
  log1pTarget,
  logTarget,
  pipeline,
  powerTarget,
  pushForward,
  standardTarget,
  transformedPredictive,
  transformTarget,
} from 'aifn-compute/learning/compose'
import {
  asTensor,
  bernoulliPredictive,
  dataset,
  gaussianPredictive,
  withDecision,
  type Dataset,
  type Decides,
  type Estimator,
  type Predicts,
  type Scores,
  type Supervised,
  type Transforms,
} from 'aifn-compute/learning/estimators'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

/** Subtract the training column means. */
const centre = (): Estimator<Dataset<Tensor, unknown>, Transforms<Tensor, Tensor>> => ({
  name: 'centre',
  fit({ x }) {
    const m = mean(x, 0) as Tensor
    return { transform: (z: Tensor) => sub(z, m) as Tensor }
  },
})

/** Simple linear regression on the first column, with a Gaussian predictive at the residual standard deviation. */
type Line = Decides<Tensor, Tensor> & Predicts<Tensor, Univariate<Tensor>> & { slope: number; intercept: number }
const line = (): Estimator<Supervised<Tensor, Tensor>, Line> => ({
  name: 'line',
  fit({ x, y }) {
    const u = toRows(x).map((r) => r[0])
    const v = toFlat(y)
    const n = u.length
    const mu = u.reduce((a, b) => a + b, 0) / n
    const mv = v.reduce((a, b) => a + b, 0) / n
    let sxy = 0
    let sxx = 0
    for (let i = 0; i < n; i++) [sxy, sxx] = [sxy + (u[i] - mu) * (v[i] - mv), sxx + (u[i] - mu) ** 2]
    const slope = sxy / sxx
    const intercept = mv - slope * mu
    let rss = 0
    for (let i = 0; i < n; i++) rss += (v[i] - intercept - slope * u[i]) ** 2
    const sd = Math.sqrt(rss / (n - 2))
    const decide = (z: Tensor) => tensor(toRows(z).map((r) => intercept + slope * r[0]))
    return {
      slope,
      intercept,
      decide,
      predictive: (z: Tensor) => gaussianPredictive(decide(z), tensor(Array(z.shape[0]).fill(sd))),
    }
  },
})

/** A fixed logistic model on the sum of the features: P(y = 1 | x) = σ(Σⱼ xⱼ). */
type Logistic = Scores<Tensor> & Predicts<Tensor, Univariate<Tensor>>
const logistic = (): Estimator<Supervised<Tensor, Tensor>, Logistic> => ({
  name: 'logistic',
  fit: () => {
    const score = (z: Tensor) => tensor(toRows(z).map((r) => r.reduce((a, b) => a + b, 0)))
    return {
      score,
      predictive: (z: Tensor) => bernoulliPredictive(tensor(toFlat(score(z)).map((s) => 1 / (1 + Math.exp(-s))))),
    }
  },
})

const s = stream('compose')
const x = normals(child(s, 'x'), [60, 2])
const yClass = tensor(toRows(x).map((r) => (r[0] + r[1] > 0 ? 1 : 0)))

describe('pipeline', () => {
  it('fits each step on the previous output and has the capabilities of its last step', () => {
    const model = pipeline(centre(), logistic()).fit(dataset(x, yClass))
    const [c, last] = model.steps
    close(
      toFlat(asTensor(model.predictive(x).params.probs)),
      toFlat(asTensor(last.predictive(c.transform(x)).params.probs)),
      0,
    )
    expect(model.kind).toBe('model')
    expect(model.composition).toBe('pipeline')
    expect(model.names).toEqual(['centre', 'logistic'])
    expect(model.stages(x).length).toBe(2)
    close(toFlat(model.features(x) as Tensor), toFlat(c.transform(x)), 0)
    expectTypeOf(model.predictive).returns.toEqualTypeOf<Univariate<Tensor>>()
    expectTypeOf(model.score).returns.toEqualTypeOf<Tensor>()
    // @ts-expect-error: the last step does not decide.
    void model.decide
    const scorer: Estimator<Supervised<Tensor, Tensor>, Scores<Tensor>> = {
      name: 'scorer',
      fit: () => ({ score: (z) => z }),
    }
    const scored = pipeline(centre(), scorer).fit(dataset(x, yClass))
    // @ts-expect-error: the last step only scores, so the pipeline has no predictive.
    void scored.predictive
    expect(toFlat(withDecision(scored, 'argmax').decide(tensor([[1, 2]])))).toEqual([1])
    // @ts-expect-error: a pipeline needs at least one step.
    expect(() => pipeline()).toThrow()
  })
})

describe('columns', () => {
  it('transforms named columns side by side', () => {
    const table = { a: tensor([1, 2, 3]), b: tensor([10, 20, 60]), id: tensor([7, 8, 9]) }
    const model = columns({ a: centre(), b: centre(), id: 'drop' }).fit(dataset(table))
    expect(toRows(model.transform(table))).toEqual([
      [-1, -20],
      [0, -10],
      [1, 30],
    ])
    expect(model.slices).toEqual({ a: [0, 1], b: [1, 2] })
    expect(() => columns({ missing: centre() }).fit(dataset(table))).toThrow(/missing/)
  })
})

describe('transformTarget', () => {
  // log y = 0.5 + 0.8 x + N(0, 0.2²).
  const xr = normals(child(s, 'xr'), [200, 1])
  const logY = add(add(0.5, mul(0.8, xr)), mul(0.2, normals(child(s, 'e'), [200, 1])))
  const y = tensor(toFlat(exp(logY) as Tensor))

  it('pushes a Gaussian on log y forward to a log-normal and inverts point predictions', () => {
    const model = transformTarget(line(), logTarget()).fit(dataset(xr, y))
    expect(model.composition).toBe('transform-target')
    expect(model.map.name).toBe('log')
    expect(model.regressor.slope).toBeCloseTo(0.8, 1)
    const xt = tensor([[0], [1]])
    const d = model.predictive(xt)
    expect(d.name).toBe('LogNormal')
    const inner = model.regressor.predictive(xt)
    const mu = toFlat(asTensor(inner.mean()))
    const sd = toFlat(asTensor(inner.params.scale))
    close(toFlat(model.decide(xt)), mu.map(Math.exp), 1e-12)
    close(
      toFlat(d.mean()),
      mu.map((m, i) => Math.exp(m + sd[i] ** 2 / 2)),
      1e-12,
    )
    // Quadrature agrees with the closed form.
    close(toFlat(model.expect(xt, (v) => v)), toFlat(d.mean()), 1e-8)
    // log p(y) = log N(log y; μ, σ²) − log y.
    const lp = toFlat(d.logProb(tensor([2, 3])))
    const innerLp = toFlat(asTensor(inner.logProb(tensor([Math.log(2), Math.log(3)]))))
    close(
      lp,
      [0, 1].map((i) => innerLp[i] - Math.log([2, 3][i])),
      1e-12,
    )
    close(toFlat(d.cdf(d.quantile(tensor([0.3, 0.3])))), [0.3, 0.3], 1e-12)
    expect(model.sample(stream('ln'), xt, 5).shape).toEqual([5, 2])
    expect(() => transformTarget(line(), logTarget()).fit(dataset(tensor([[0], [1]]), tensor([1, -1])))).toThrow(
      /domain/,
    )
  })

  it('handles general monotone maps and fitted power transforms', () => {
    const base = gaussianPredictive(tensor([0]), tensor([1]))
    const flipped = transformedPredictive(base, affineTarget(1, -2)) // y = 1 − 2z
    close(toFlat(flipped.mean()), [1], 1e-12)
    close(toFlat(flipped.variance()), [4], 1e-10)
    close(toFlat(flipped.cdf(tensor(1))), [0.5], 1e-15)
    // Decreasing map: P(y ≤ 3) = P(z ≥ −1).
    close(toFlat(flipped.cdf(tensor(3))), [0.8413447460685429], 1e-12)
    expect(() => flipped.entropy()).toThrow()
    expect(pushForward(base, logTarget()).name).toBe('LogNormal')
    expect(pushForward(base, log1pTarget()).name).toBe('Transformed')
    const model = transformTarget(line(), powerTarget({ method: 'box-cox' })).fit(dataset(xr, y))
    const lambda = (model.map as unknown as { lambda: number }).lambda
    expect(Math.abs(lambda)).toBeLessThan(0.3)
    expect(model.predictive(tensor([[0]])).name).toBe('Transformed')
    const standard = standardTarget().fit(tensor([1, 2, 3]))
    expect(standard.apply(2)).toBeCloseTo(0, 12)
    expect(standard.invert(standard.apply(3))).toBeCloseTo(3, 12)
    expect(() => affineTarget(0, 0)).toThrow()
  })
})
