import { describe, expect, it } from 'vitest'
import { kNearestNeighbours, kNearestNeighboursRegression } from 'aifn-methods/learning/neighbours'
import { toRows } from 'aifn-compute/foundation/tensor'
import { dataset, hasPredictive } from 'aifn-compute/learning/estimators'
import { DomainError } from 'aifn-compute/foundation/errors'
import { close, fx, X3, XQ, Y3 } from '../shared'

describe('k-nearest neighbours', () => {
  it('matches scikit-learn (uniform, distance, Manhattan)', () => {
    close(kNearestNeighbours({ k: 5 }).fit(dataset(X3, Y3)).predictive(XQ), fx.knn.uniform)
    close(kNearestNeighbours({ k: 5, weights: 'distance' }).fit(dataset(X3, Y3)).predictive(XQ), fx.knn.distance)
    close(kNearestNeighbours({ k: 4, metric: 'manhattan' }).fit(dataset(X3, Y3)).predictive(XQ), fx.knn_manhattan)
  })
  it('exposes the neighbours, nearest first', () => {
    const m = kNearestNeighbours({ k: 3 }).fit(dataset(X3, Y3))
    const nb = m.neighbours(X3)
    expect(toRows(nb.index).map((r) => r[0])).toEqual(fx.x3.map((_, i) => i))
    const d = toRows(nb.distance)
    for (const r of d) expect(r[0] <= r[1] && r[1] <= r[2]).toBe(true)
    expect(hasPredictive(m)).toBe(true)
  })
  it('validates k for classification and regression', () => {
    expect(() => kNearestNeighbours({ k: 0 })).toThrow(DomainError)
    expect(() => kNearestNeighboursRegression({ k: 0 })).toThrow(DomainError)
    expect(() => kNearestNeighboursRegression({ k: -1 })).toThrow(DomainError)
    expect(() => kNearestNeighboursRegression({ k: 1.5 })).toThrow(DomainError)
  })
})
