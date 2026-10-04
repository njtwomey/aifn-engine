import { describe, expect, it } from 'vitest'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { isotonicRegressor } from 'aifn-methods/learning'

describe('isotonicRegressor estimator', () => {
  it('fits the canonical worked example sequence', () => {
    const x = tensor([[0], [1], [2], [3], [4], [5]])
    const y = tensor([2, 5, 3, 1, 6, 4])

    const estimator = isotonicRegressor()
    const model = estimator.fit(dataset(x, y))

    // Fitted values on training design points: (2, 3, 3, 3, 5, 5)
    const pred = model.predict(x)
    expect(Array.from(toFlat(pred))).toEqual([2, 3, 3, 3, 5, 5])
  })

  it('interpolates linearly and handles out-of-bounds clipping', () => {
    const x = tensor([[0], [1], [2], [3], [4], [5]])
    const y = tensor([2, 5, 3, 1, 6, 4])

    const model = isotonicRegressor({ interpolation: 'linear', outOfBounds: 'clip' }).fit(dataset(x, y))

    // At x = -1 (below min 0): clips to 2
    // At x = 2.5 (between 2 and 3, both have y=3): 3
    // At x = 3.5 (midpoint of x=3 (y=3) and x=4 (y=5)): 4
    // At x = 10 (above max 5): clips to 5
    const q = tensor([[-1], [2.5], [3.5], [10]])
    const pred = model.predict(q)
    expect(Array.from(toFlat(pred))).toEqual([2, 3, 4, 5])
  })

  it('supports step-function interpolation and NaN outOfBounds', () => {
    const x = tensor([[0], [1], [2], [3], [4], [5]])
    const y = tensor([2, 5, 3, 1, 6, 4])

    const model = isotonicRegressor({ interpolation: 'step', outOfBounds: 'nan' }).fit(dataset(x, y))

    const q = tensor([[-1], [3.5], [6]])
    const pred = Array.from(toFlat(model.predict(q)))
    expect(Number.isNaN(pred[0])).toBe(true)
    expect(pred[1]).toBe(3) // step function holds y(3)=3 until x=4
    expect(Number.isNaN(pred[2])).toBe(true)
  })

  it('supports antitonic regression (increasing: false)', () => {
    const x = tensor([[0], [1], [2], [3]])
    const y = tensor([1, 4, 2, 5])

    const model = isotonicRegressor({ increasing: false }).fit(dataset(x, y))
    const pred = Array.from(toFlat(model.predict(x)))

    // Monotonically non-increasing
    for (let i = 1; i < pred.length; i++) {
      expect(pred[i]).toBeLessThanOrEqual(pred[i - 1])
    }
  })

  it('rejects multi-dimensional inputs', () => {
    const x = tensor([
      [0, 1],
      [1, 2],
    ])
    const y = tensor([1, 2])
    expect(() => isotonicRegressor().fit(dataset(x, y))).toThrow(/univariate/)
  })
})
