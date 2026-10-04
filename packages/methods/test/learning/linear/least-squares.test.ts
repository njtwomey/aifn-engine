/**
 * linearRegression against scikit-learn's LinearRegression and Ridge (fixture `learning/linear`, shared with the
 * logistic regression tests in learning/generalised/glm).
 */
import { describe, expect, expectTypeOf, it } from 'vitest'
import { linearRegression } from 'aifn-methods/learning/linear'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  asTensor,
  dataset,
  evaluate,
  hasPredictive,
  type AnyUnivariate,
  type InputOf,
} from 'aifn-compute/learning/estimators'
import { meanAbsoluteError, meanSquaredError, r2Score } from 'aifn-compute/learning/metrics'
import { fixture } from '../../fixtures'

export type LinearFixture = {
  x: number[][]
  y: number[]
  x_test: number[][]
  ols: { coef: number[]; intercept: number; predict: number[]; noise_sd: number }
  ridge: { alpha: number; coef: number[]; intercept: number; predict: number[] }
  deficient: { x: number[][]; coef: number[]; intercept: number }
  binary: { l2: number; y: number[]; coef: number[]; intercept: number; proba: number[]; predict: number[] }
  multinomial: { l2: number; y: number[]; coef: number[][]; intercept: number[]; proba: number[][]; predict: number[] }
}
const fx = fixture<LinearFixture>('learning/linear')
const X = tensor(fx.x)
const Y = tensor(fx.y)
const XT = tensor(fx.x_test)

const close = (actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number) => {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tol)
}

describe('linearRegression', () => {
  const model = linearRegression().fit(dataset(X, Y))

  it('matches scikit-learn LinearRegression', () => {
    close(toFlat(model.weights), fx.ols.coef, 1e-10)
    expect(model.intercept).toBeCloseTo(fx.ols.intercept, 10)
    close(toFlat(model.decide(XT)), fx.ols.predict, 1e-10)
    expect(model.noiseSd).toBeCloseTo(fx.ols.noise_sd, 10)
    expect(model.rank).toBe(3)
    expect(model.residualDof).toBe(40 - 4)
  })

  it('matches scikit-learn Ridge', () => {
    const ridge = linearRegression({ l2: fx.ridge.alpha }).fit(dataset(X, Y))
    close(toFlat(ridge.weights), fx.ridge.coef, 1e-10)
    expect(ridge.intercept).toBeCloseTo(fx.ridge.intercept, 10)
    close(toFlat(ridge.expect(XT)), fx.ridge.predict, 1e-10)
  })

  it('reports rank deficiency and returns the minimum-norm solution', () => {
    const m = linearRegression().fit(dataset(tensor(fx.deficient.x), Y))
    expect(m.rank).toBe(2)
    close(toFlat(m.weights), fx.deficient.coef, 1e-8)
  })

  it('has a Gaussian predictive with the plug-in noise', () => {
    const d = model.predictive(XT)
    expect(d.name).toBe('Normal')
    close(toFlat(asTensor(d.mean())), fx.ols.predict, 1e-10)
    close(toFlat(asTensor(d.params.scale)), Array(5).fill(fx.ols.noise_sd), 1e-12)
    // E[y²] = μ² + σ² by quadrature.
    close(
      toFlat(model.expect(XT, (v) => v * v)),
      fx.ols.predict.map((m) => m * m + fx.ols.noise_sd ** 2),
      1e-9,
    )
    // Mean Gaussian negative log-likelihood at the plug-in σ: ½ log 2πσ² + RSS / (2nσ²).
    const s2 = model.noiseSd ** 2
    const nll = -toFlat(asTensor(model.predictive(X).logProb(Y))).reduce((s, v) => s + v, 0) / 40
    expect(nll).toBeCloseTo(0.5 * Math.log(2 * Math.PI * s2) + model.rss / (2 * 40 * s2), 12)
  })

  it('samples from the predictive', () => {
    const draws = model.sample(stream('lr'), XT, 4000)
    expect(draws.shape).toEqual([4000, 5])
    const rows = toRows(draws)
    const mean0 = rows.reduce((s, r) => s + r[0], 0) / rows.length
    expect(Math.abs(mean0 - fx.ols.predict[0])).toBeLessThan(4 * (fx.ols.noise_sd / Math.sqrt(4000)))
    expect(toFlat(model.sample(stream('lr'), XT, 3))).toEqual(toFlat(model.sample(stream('lr'), XT, 3)))
  })

  it('is evaluated by the registered regression metrics', () => {
    const r = evaluate(model, dataset(X, Y), [meanSquaredError, meanAbsoluteError, r2Score])
    expect(r.meanSquaredError).toBeCloseTo(model.rss / 40, 12)
    expect(r.r2Score).toBeGreaterThan(0.95)
    expect(r.meanAbsoluteError).toBeGreaterThan(0)
  })

  it('has the declared capability types', () => {
    expectTypeOf(model.predictive).returns.toEqualTypeOf<AnyUnivariate>()
    expectTypeOf<InputOf<typeof model>>().toEqualTypeOf<Tensor>()
    const models: unknown[] = [model, { decide: (x: Tensor) => x }]
    expect(models.filter((m) => hasPredictive(m)).length).toBe(1)
    const m = models[0]
    if (hasPredictive(m)) expect(m.predictive(XT).batchShape).toEqual([5])
  })

  it('rejects mismatched shapes', () => {
    expect(() => linearRegression().fit(dataset(X, tensor([1, 2])))).toThrow(/targets/)
    expect(() => model.decide(tensor([[1, 2]]))).toThrow(/features/)
    expect(() => linearRegression({ l2: -1 })).toThrow()
  })
})
