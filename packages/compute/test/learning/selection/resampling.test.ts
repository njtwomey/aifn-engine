/**
 * aifn-compute/learning/selection, bootstrap prediction error: the leave-one-out bootstrap, .632 and .632+ estimates of
 * Efron and Tibshirani (1997) on fixed bootstrap samples, against a numpy reference for least squares (squared error)
 * and scikit-learn's one-nearest-neighbour classifier (0-1 loss); the no-information rate in its closed form; drawn
 * samples reproducible from the stream and consistent with the out-of-bag counts.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { lstsq } from 'aifn-compute/numerics/linalg'
import { bootstrapPredictionError } from 'aifn-compute/learning/selection'
import { fixture } from '../../fixtures'

interface Case {
  x: number[][]
  y: number[]
  indices: number[][]
  apparent: number
  looBootstrap: number
  noInformation: number
  relativeOverfitting: number
  err632: number
  err632plus: number
  outOfBagCount: number[]
  perPoint: number[]
}

const F = fixture<{ bootstrap: { regression: Case; classification: Case } }>('learning/selection')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(b[i])) expect(a[i]).toBeNaN()
    else expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
  }
}

/** Least squares with an intercept. */
const leastSquares = {
  name: 'least squares',
  fit({ x, y }: { x: Tensor; y: Tensor }) {
    const rows = toRows(x).map((r) => [1, ...r])
    const w = toFlat(lstsq(tensor(rows), y).x as Tensor)
    return { decide: (z: Tensor) => tensor(toRows(z).map((r) => r.reduce((s, v, j) => s + v * w[j + 1], w[0]))) }
  },
}

/** One-nearest-neighbour classification. */
const nearest = {
  name: '1-nn',
  fit({ x, y }: { x: Tensor; y: Tensor }) {
    const rows = toRows(x)
    const labels = toFlat(y)
    const d2 = (a: number[], b: number[]) => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0)
    return {
      decide: (z: Tensor) =>
        tensor(toRows(z).map((r) => labels[rows.reduce((best, q, j) => (d2(r, q) < d2(r, rows[best]) ? j : best), 0)])),
    }
  },
}

describe('bootstrapPredictionError', () => {
  it.each([
    ['least squares, squared error', 'regression', leastSquares, 'squared'],
    ['one-nearest-neighbour, 0-1 loss', 'classification', nearest, 'zeroOne'],
  ] as const)('%s matches the reference on fixed samples', (_, key, estimator, loss) => {
    const c = F.bootstrap[key]
    const e = bootstrapPredictionError(estimator, { kind: 'dataset', x: tensor(c.x), y: tensor(c.y) } as const, {
      indices: c.indices,
      loss,
    })
    expect(e.B).toBe(c.indices.length)
    expect(e.apparent).toBeCloseTo(c.apparent, 10)
    expect(e.looBootstrap).toBeCloseTo(c.looBootstrap, 10)
    expect(e.noInformation).toBeCloseTo(c.noInformation, 10)
    expect(e.relativeOverfitting).toBeCloseTo(c.relativeOverfitting, 10)
    expect(e.err632).toBeCloseTo(c.err632, 10)
    expect(e.err632plus).toBeCloseTo(c.err632plus, 10)
    expect(Array.from(toFlat(e.outOfBagCount))).toEqual(c.outOfBagCount)
    close(toFlat(e.perPoint), c.perPoint, 1e-9)
    // The .632+ weight form.
    expect((1 - e.weight) * e.apparent + e.weight * Math.min(e.looBootstrap, e.noInformation)).toBeCloseTo(
      e.err632plus,
      12,
    )
  })

  it('a custom pointwise loss equals the named one', () => {
    const c = F.bootstrap.regression
    const data = { kind: 'dataset', x: tensor(c.x), y: tensor(c.y) } as const
    const squared = (y: Tensor, p: Tensor) => fromData(Float64Array.from(toFlat(y), (v, i) => (v - toFlat(p)[i]) ** 2))
    const e = bootstrapPredictionError(leastSquares, data, { indices: c.indices.slice(0, 10), loss: squared })
    const f = bootstrapPredictionError(leastSquares, data, { indices: c.indices.slice(0, 10) })
    expect(e.err632plus).toBeCloseTo(f.err632plus, 12)
  })

  it('draws reproducibly from the stream; the in-bag counts and out-of-bag counts agree', () => {
    const c = F.bootstrap.classification
    const data = { kind: 'dataset', x: tensor(c.x), y: tensor(c.y) } as const
    const a = bootstrapPredictionError(nearest, data, { B: 20, stream: stream(7), loss: 'zeroOne' })
    const b = bootstrapPredictionError(nearest, data, { B: 20, stream: stream(7), loss: 'zeroOne' })
    expect(a.err632plus).toBe(b.err632plus)
    const inBag = toRows(a.inBag)
    expect(inBag.every((row) => row.reduce((s, v) => s + v, 0) === c.y.length)).toBe(true)
    const oob = c.y.map((_, i) => inBag.filter((row) => row[i] === 0).length)
    expect(Array.from(toFlat(a.outOfBagCount))).toEqual(oob)
    // Replaying the drawn samples as indices gives the same estimates.
    const samples = inBag.map((row) => row.flatMap((count, i) => Array<number>(count).fill(i)))
    const replay = bootstrapPredictionError(nearest, data, { indices: samples, loss: 'zeroOne' })
    expect(replay.looBootstrap).toBe(a.looBootstrap)
    // 1-NN on its own training rows: no training error, and the no-information rate is p1 (1 - q1) + q1 (1 - p1).
    expect(a.apparent).toBe(0)
    const p1 = c.y.reduce((s, v) => s + v, 0) / c.y.length
    expect(a.noInformation).toBeCloseTo(2 * p1 * (1 - p1), 12)
  })

  it('throws without a stream or indices, and for indices of the wrong width', () => {
    const data = { kind: 'dataset', x: tensor([[0], [1], [2]]), y: tensor([0, 1, 2]) } as const
    expect(() => bootstrapPredictionError(leastSquares, data)).toThrow(/stream/)
    expect(() => bootstrapPredictionError(leastSquares, data, { indices: [[0, 1]] })).toThrow(/rows/)
  })
})
