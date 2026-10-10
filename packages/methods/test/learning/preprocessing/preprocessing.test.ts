// Smoke tests (one per export family); reference values from scikit-learn 1.9 / scipy 1.18, inlined.
import { describe, expect, it } from 'vitest'
import {
  boxCox,
  boxCoxInverse,
  boxCoxLambda,
  fitTransform,
  maxAbsScaler,
  minMaxScaler,
  oneHotEncoder,
  ordinalEncoder,
  polynomialFeatures,
  powerTransform,
  randomFourierFeatures,
  robustScaler,
  simpleImputer,
  splineFeatures,
  standardScaler,
  targetEncodeCrossFit,
  targetEncoder,
  whitening,
  yeoJohnson,
  yeoJohnsonInverse,
  yeoJohnsonLambda,
} from 'aifn-methods/learning/preprocessing'
import { normals, stream } from 'aifn-compute/foundation/random'
import { matmul, tensor, toFlat, toRows, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

const X = tensor([
  [1, 10],
  [2, 20],
  [3, 30],
  [10, 0],
])

describe('scalers', () => {
  it('standard, min–max, robust and max-abs scale and invert', () => {
    const s = standardScaler().fit({ x: X })
    close(toFlat(s.mean), [4, 15], 1e-15)
    const z = s.transform(X)
    for (const col of toRows(transpose(z) as Tensor)) close([col.reduce((a, b) => a + b, 0)], [0], 1e-12)
    close(toFlat(s.inverseTransform(z)), toFlat(X), 1e-12)
    close(toFlat(minMaxScaler().fit({ x: X }).transform(X)), [0, 1 / 3, 1 / 9, 2 / 3, 2 / 9, 1, 1, 0], 1e-15)
    const r = robustScaler().fit({ x: X })
    close(toFlat(r.median), [2.5, 15], 1e-15)
    close(toFlat(maxAbsScaler().fit({ x: X }).transform(X)).slice(0, 2), [0.1, 1 / 3], 1e-15)
    const constant = standardScaler().fit({ x: tensor([[1], [1]]) })
    expect(constant.constant).toEqual([true])
  })
})

describe('encoders', () => {
  const cats = ['a', 'b', 'a', 'c', 'b', 'a', 'c', 'c', 'a']
  const y = tensor([1, 2, 1.5, 4, 2.5, 0.5, 5, 3.5, 1])

  it('one-hot and ordinal encode and invert', () => {
    const oh = oneHotEncoder().fit({ x: cats })
    expect(oh.categories).toEqual([['a', 'b', 'c']])
    expect(toRows(oh.transform(['c', 'a']))).toEqual([
      [0, 0, 1],
      [1, 0, 0],
    ])
    expect(oh.inverseTransform(oh.transform(cats))).toEqual(cats)
    const dropped = oneHotEncoder({ drop: 'first' }).fit({ x: cats })
    expect(dropped.transform(['a', 'c']).shape).toEqual([2, 2])
    expect(dropped.inverseTransform(dropped.transform(['a', 'c']))).toEqual(['a', 'c'])
    expect(() => oh.transform(['z'])).toThrow(/unknown/)
    expect(toRows(oneHotEncoder({ handleUnknown: 'ignore' }).fit({ x: cats }).transform(['z']))).toEqual([[0, 0, 0]])
    const ord = ordinalEncoder().fit({
      x: tensor([
        [3, 1],
        [1, 1],
        [2, 5],
      ]),
    })
    expect(toRows(ord.transform(tensor([[1, 5]])))).toEqual([[0, 1]])
    expect(toRows(ord.inverseTransform(tensor([[2, 0]])) as Tensor)).toEqual([[3, 1]])
  })

  it('target encoding matches scikit-learn (auto, m = 2, cross-fitted)', () => {
    close(
      toFlat(targetEncoder().fit({ x: cats, y }).encodings[0]),
      [1.019448946515397, 2.2512155591572123, 4.060606060606061],
      1e-12,
    )
    const m2 = targetEncoder({ smooth: 2 }).fit({ x: cats, y })
    close(toFlat(m2.encodings[0]), [1.4444444444444446, 2.291666666666667, 3.4333333333333336], 1e-12)
    close(toFlat(m2.transform(['zzz'])), [m2.targetMean], 0)
    const cf = targetEncodeCrossFit({ x: cats, y }, { smooth: 2, folds: 3 })
    close(
      toFlat(cf.encoded),
      [
        1.75, 2.6666666666666665, 1.75, 3.291666666666667, 2.2222222222222223, 1.6333333333333335, 2.611111111111111,
        2.611111111111111, 1.3666666666666667,
      ],
      1e-12,
    )
  })
})

describe('imputation', () => {
  it('fills NaN by column statistics and reports empty columns', () => {
    const x = tensor([
      [1, NaN, NaN],
      [NaN, 2, NaN],
      [3, 2, NaN],
      [3, 5, NaN],
    ])
    const mean = simpleImputer().fit({ x })
    expect(toRows(mean.transform(x))[1][0]).toBeCloseTo(7 / 3, 14)
    expect(mean.empty).toEqual([false, false, true])
    expect(mean.missing).toEqual([1, 1, 4])
    expect(toFlat(simpleImputer({ strategy: 'median' }).fit({ x }).statistics).slice(0, 2)).toEqual([3, 2])
    expect(toFlat(simpleImputer({ strategy: 'most-frequent' }).fit({ x }).statistics).slice(0, 2)).toEqual([3, 2])
  })
})

describe('feature expansions', () => {
  it('polynomial features in scikit-learn order', () => {
    const p = polynomialFeatures({ degree: 3 }).fit({ x: tensor([[2, 3]]) })
    expect(toRows(p.powers)).toEqual([
      [0, 0],
      [1, 0],
      [0, 1],
      [2, 0],
      [1, 1],
      [0, 2],
      [3, 0],
      [2, 1],
      [1, 2],
      [0, 3],
    ])
    expect(toFlat(p.transform(tensor([[2, 3]])))).toEqual([1, 2, 3, 4, 6, 9, 8, 12, 18, 27])
    expect(
      polynomialFeatures({ degree: 2, interactionOnly: true, includeBias: false }).fit({ x: tensor([[1, 2, 3]]) })
        .featureNames,
    ).toEqual(['x0', 'x1', 'x2', 'x0 x1', 'x0 x2', 'x1 x2'])
  })

  it('spline features match scikit-learn SplineTransformer', () => {
    const train = tensor([[0], [1], [2], [3]])
    const s = splineFeatures({ knots: 4, degree: 3 }).fit({ x: train })
    close(
      toFlat(s.transform(tensor([[0.5], [2.25]]))),
      [
        0.020833333333333332, 0.47916666666666663, 0.4791666666666667, 0.020833333333333332, 0, 0, 0, 0, 0.0703125,
        0.6119791666666666, 0.31510416666666663, 0.0026041666666666665,
      ],
      1e-14,
    )
    close(toFlat(s.transform(tensor([[-1]]))), [1 / 6, 2 / 3, 1 / 6, 0, 0, 0], 1e-14)
    const cont = splineFeatures({ knots: 4, degree: 3, extrapolation: 'continue' }).fit({ x: train })
    close(toFlat(cont.transform(tensor([[4]]))), [0, 0, -1 / 6, 2 / 3, -5 / 6, 4 / 3], 1e-13)
  })

  it('random Fourier features approximate the squared-exponential kernel', () => {
    const x = tensor([
      [0, 0],
      [0.5, -0.3],
    ])
    const rff = randomFourierFeatures({ components: 20000, lengthscale: 0.8 }).fit({ x }, { stream: stream('rff') })
    const z = toRows(rff.transform(x))
    const k = z[0].reduce((a, v, i) => a + v * z[1][i], 0)
    expect(Math.abs(k - Math.exp(-(0.25 + 0.09) / (2 * 0.64)))).toBeLessThan(0.02)
    expect(() => randomFourierFeatures().fit({ x })).toThrow(/stream/)
  })
})

describe('whitening', () => {
  it('PCA and ZCA give identity covariance and invert', () => {
    const raw = normals(stream('w'), [200, 3])
    const x = matmul(
      raw,
      tensor([
        [2, 0, 0],
        [1, 1, 0],
        [0.5, 0.2, 0.3],
      ]),
    ) as Tensor
    for (const method of ['pca', 'zca'] as const) {
      const w = whitening({ method }).fit({ x })
      const z = w.transform(x)
      const cov = toRows(matmul(transpose(z), z) as Tensor).map((r) => r.map((v) => v / 199))
      close(cov.flat(), [1, 0, 0, 0, 1, 0, 0, 0, 1], 1e-10)
      close(toFlat(w.inverseTransform(z)), toFlat(x), 1e-10)
      expect(w.singular).toBe(false)
    }
    expect(() => whitening({ components: 0 }).fit({ x })).toThrow(DomainError)
    expect(() => whitening({ components: 4 }).fit({ x })).toThrow(DomainError)
    expect(() => whitening({ components: 1.5 }).fit({ x })).toThrow(DomainError)
  })
})

describe('power transforms', () => {
  const x = [0.5, 1.2, 2.0, 3.3, 4.1, 7.9, 12.5, 0.9, 2.2, 5.0]

  it('λ by maximum likelihood matches scipy', () => {
    expect(boxCoxLambda(x).lambda).toBeCloseTo(0.053913509000458715, 6)
    expect(yeoJohnsonLambda(x.map((v) => v - 3)).lambda).toBeCloseTo(0.4134266217442406, 6)
  })

  it('transforms invert', () => {
    const t = tensor([0.3, 2, 5])
    close(toFlat(boxCoxInverse(boxCox(t, 0.4), 0.4)), toFlat(t), 1e-13)
    close(toFlat(yeoJohnsonInverse(yeoJohnson(tensor([-2, 0, 3]), 1.3), 1.3)), [-2, 0, 3], 1e-13)
    expect(boxCox(Math.E, 0)).toBeCloseTo(1, 15)
  })

  it('powerTransform matches scikit-learn PowerTransformer', () => {
    const { model, z } = fitTransform(powerTransform(), { x: tensor(x.map((v) => [v - 3])) })
    expect(model.lambdas[0]).toBeCloseTo(0.4134266217442406, 6)
    close(toFlat(model.transform(tensor([[0], [2]]))), [0.1160269345560965, 0.6852372100891834], 1e-6)
    close(
      toFlat(model.inverseTransform(z)),
      x.map((v) => v - 3),
      1e-10,
    )
    expect(() => powerTransform({ method: 'box-cox' }).fit({ x: tensor([[-1], [2]]) })).toThrow(/positive/)
  })
})
