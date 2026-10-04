// The compute combinators (aifn-compute/learning/compose, aifn-compute/learning/validate) and aifn-compute/numerics/interpolate's bases together
// with the learning area's transformers and models.
// These cases moved here from the compute tests so that compute's tests never import an application.
import { describe, expect, expectTypeOf, it } from 'vitest'
import { oneHotEncoder, polynomialFeatures, splineFeatures, standardScaler } from 'aifn-methods/learning/preprocessing'
import { columns, pipeline } from 'aifn-compute/learning/compose'
import type { AnyUnivariate } from 'aifn-compute/foundation/contracts'
import {
  classProbabilities,
  dataset,
  withDecision,
  type Estimator,
  type Scores,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import { accuracy, logLoss } from 'aifn-compute/learning/metrics'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { bsplineBasis } from 'aifn-compute/numerics/interpolate'
import {
  add,
  fromData,
  linspace,
  matmul,
  mul,
  tensor,
  toFlat,
  toRows,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { crossValidate, kFold } from 'aifn-compute/learning/validate'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

const s = stream('compose')
const x = normals(child(s, 'x'), [60, 2])
const logits = toFlat(matmul(x, tensor([2, -1])) as Tensor)
const u = toFlat(normals(child(s, 'u'), [60]))
const yClass = tensor(logits.map((l, i) => (l + 0.5 * u[i] > 0 ? 1 : 0)))

describe('pipeline', () => {
  const model = pipeline(
    standardScaler(),
    polynomialFeatures({ degree: 2, includeBias: false }),
    logisticRegression({ l2: 1 }),
  ).fit(dataset(x, yClass))

  it('fits each step on the previous output and exposes the fitted steps', () => {
    const [scaler, poly, logistic] = model.steps
    expect(scaler.kind).toBe('model')
    expect(scaler.name).toBe('standard-scaler')
    expect(poly.featureNames.length).toBe(5)
    expect(logistic.weights.shape).toEqual([5])
    expect(model.names).toEqual(['standard-scaler', 'polynomial-features', 'logistic-regression'])
    // The pipeline's predictive equals the final model's on the transformed inputs.
    const direct = logistic.predictive(poly.transform(scaler.transform(x)))
    close(toFlat(classProbabilities(model.predictive(x))), toFlat(classProbabilities(direct)), 0)
    expect(model.stages(x).length).toBe(3)
    expect(model.training.meta.stopped).toBe('done')
  })

  it('has exactly the capabilities of its last step', () => {
    expectTypeOf(model.predictive).returns.toEqualTypeOf<AnyUnivariate>()
    expectTypeOf(model.decide).parameter(0).toEqualTypeOf<Tensor>()
    const scorer: Estimator<Supervised<Tensor, Tensor>, Scores<Tensor>> = {
      name: 'scorer',
      fit: () => ({ score: (z) => z }),
    }
    const scored = pipeline(standardScaler(), scorer).fit(dataset(x, yClass))
    expectTypeOf(scored).toHaveProperty('score')
    // @ts-expect-error: the last step only scores, so the pipeline has no predictive.
    void scored.predictive
    expect('predictive' in scored).toBe(false)
    const transformsOnly = pipeline(standardScaler(), polynomialFeatures()).fit(dataset(x))
    expect(transformsOnly.transform(x).shape).toEqual([60, 6])
    expect(toFlat(withDecision(scored, 'argmax').decide(tensor([[1, 2]])))).toEqual([1])
  })
})

describe('columns', () => {
  it('transforms named columns side by side', () => {
    const table = { age: tensor([30, 40, 50, 60]), city: ['Cork', 'Oslo', 'Cork', 'Lima'], id: tensor([1, 2, 3, 4]) }
    const model = columns({ age: standardScaler(), city: oneHotEncoder(), id: 'drop' }).fit(dataset(table))
    const z = model.transform(table)
    expect(z.shape).toEqual([4, 4])
    expect(model.slices).toEqual({ age: [0, 1], city: [1, 4] })
    expect(model.featureNames).toEqual(['age', 'city:Cork', 'city:Lima', 'city:Oslo'])
    expect(toRows(z)[1].slice(1)).toEqual([0, 0, 1])
    expect(model.steps.city.categories).toEqual([['Cork', 'Lima', 'Oslo']])
    const withRest = columns({ city: oneHotEncoder() }, { remainder: 'passthrough' }).fit(dataset(table))
    expect(withRest.transform(table).shape).toEqual([4, 5])
  })
})

describe('crossValidate with a pipeline', () => {
  const s = stream('cv')
  const x = normals(child(s, 'x'), [80, 3])
  const y = tensor(toFlat(add(matmul(x, tensor([1, -2, 0])), mul(0.5, normals(child(s, 'e'), [80]))) as Tensor))
  const labels = tensor(toFlat(y).map((v) => (v > 0 ? 1 : 0)))

  it('keeps folds, models, predictions, metrics and traces', () => {
    const cv = crossValidate(pipeline(standardScaler(), logisticRegression()), dataset(x, labels), kFold({ k: 4 }), [
      accuracy,
      logLoss,
    ])
    expect(cv.folds.length).toBe(4)
    expect(cv.assignment.shape).toEqual([4, 80])
    expect(cv.scores.accuracy.shape).toEqual([4])
    expect(cv.mean.accuracy).toBeGreaterThan(0.8)
    expect(cv.folds[0].model.steps[0].name).toBe('standard-scaler')
    expect(cv.folds[0].training?.meta.stopped).toBe('done')
    expect(cv.outOfFold.decide?.shape).toEqual([80])
    expect(toFlat(cv.outOfFold.decide!).every((v) => v === 0 || v === 1)).toBe(true)
    expect(cv.directions.logLoss).toBe('lower')
  })
})

describe('splineFeatures', () => {
  it('splineFeatures in preprocess uses the same basis', () => {
    const xs = toFlat(linspace(0, 1, 40))
    const data = fromData(Float64Array.from(xs), [xs.length, 1])
    const f = splineFeatures({ knots: 5, degree: 3 }).fit(dataset(data))
    const knotsRow = toRows(f.knots)[0]
    const B = toRows(bsplineBasis(tensor(xs), tensor(knotsRow), 3))
    close(toRows(f.transform(data)).flat(), B.flat(), 1e-14)
  })
})
