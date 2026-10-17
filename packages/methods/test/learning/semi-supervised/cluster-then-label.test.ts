/** Cluster-then-label on the site's self-check: cluster classes [0, 1], and $P(1 \mid$ point 2$) = 0.8$. */
import { describe, expect, it } from 'vitest'
import { tensor } from 'aifn-compute/foundation/tensor'
import { clusterClasses, clusterThenLabelProbabilities } from 'aifn-methods/learning/semi-supervised'

const R = [
  [0.99, 0.01],
  [0.9, 0.1],
  [0.2, 0.8],
  [0.98, 0.02],
]

describe('cluster-then-label', () => {
  it('names each cluster by the labelled responsibility it gets', () => {
    expect(Array.from(clusterClasses(R, [0, -1, -1, 1], 2))).toEqual([0, 1])
    expect(Array.from(clusterClasses(tensor(R), [0, -1, -1, 1], 2))).toEqual([0, 1])
    expect(Array.from(clusterClasses(R, [-1, -1, -1, 1], 2))).toEqual([1, 1])
    expect(Array.from(clusterClasses(R, [-1, -1, -1, -1], 2))).toEqual([-1, -1])
  })

  it('sums responsibilities by cluster class, uniform where no assigned cluster has mass', () => {
    const p = clusterThenLabelProbabilities(R, [0, 1], 2)
    expect(p[2 * 2 + 1]).toBeCloseTo(0.8, 12)
    expect(p[0]).toBeCloseTo(0.99, 12)
    const q = clusterThenLabelProbabilities([[1, 0]], [-1, 0], 2)
    expect(Array.from(q)).toEqual([0.5, 0.5])
  })

  it('refuses mismatched rows and labels out of range', () => {
    expect(() => clusterClasses(R, [0, 1], 2)).toThrow(/rows/)
    expect(() => clusterClasses(R, [0, 2, -1, -1], 2)).toThrow(/labels/)
  })
})
