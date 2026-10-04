/** The scalar row kernel behind pairwiseDistances. */
import { describe, expect, it } from 'vitest'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { pairwiseDistances, rowDistance, squaredRowDistance, type PairwiseMetric } from 'aifn-compute/numerics/linalg'

const a = [
  [0, 1, 2],
  [3, -1, 0.5],
]
const b = [
  [1, 1, 1],
  [0, 0, 0],
  [-2, 4, 1],
]
const flat = (m: number[][]) => Float64Array.from(m.flat())

describe('rowDistance', () => {
  it('is the entry of pairwiseDistances for every metric', () => {
    const metrics: PairwiseMetric[] = ['euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine']
    for (const metric of metrics) {
      const D = toFlat(pairwiseDistances(a, b, { metric, p: 3 }))
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 3; j++) {
          const r = rowDistance(flat(a), i, flat(b), j, 3, metric, 3)
          if (Number.isNaN(D[i * 3 + j])) expect(r).toBeNaN()
          else expect(r).toBe(D[i * 3 + j])
        }
    }
  })
  it('squaredRowDistance is the squared Euclidean distance', () => {
    expect(squaredRowDistance(flat(a), 0, flat(b), 2, 3)).toBe(4 + 9 + 1)
  })
})
