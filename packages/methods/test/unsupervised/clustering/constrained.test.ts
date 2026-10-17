/** aifn-methods/unsupervised/clustering: COP-k-means, PCK-means, seeded centroids and RCA on the note's examples. */
import { describe, expect, it } from 'vitest'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import {
  clusteringAlgorithms,
  clusteringFunctions,
  constrainedKMeansSteps,
  constraintClosure,
  relevantComponentMetric,
  seededCentroids,
  type ConstraintPair,
} from 'aifn-methods/unsupervised/clustering'

// The semi-supervised clustering note's six points.
const x = [[0], [1], [4], [6], [9], [10]]
const start = { centroids: [[0], [10]] }
const lab = (st: { labels: Int32Array }) => Array.from(st.labels).join('')

describe('constrainedKMeansSteps on 0, 1, 4, 6, 9, 10', () => {
  it('without constraints is k-means: {0,1,4}, {6,9,10}, inertia 52/3', () => {
    const plain = run(constrainedKMeansSteps(x, { k: 2, rule: 'pck', weight: 0 }), start, 20)
    expect(lab(plain)).toBe('000111')
    expect(plain.inertia).toBeCloseTo(52 / 3, 12)
    expect(Array.from(toFlat(plain.centroids))).toEqual([5 / 3, 25 / 3])
  })

  it('COP with must-link 4–6: {0,1,4,6}, {9,10}, inertia 23.25', () => {
    const cop = run(constrainedKMeansSteps(x, { k: 2, rule: 'cop', mustLink: [[2, 3]] }), start, 20)
    expect(lab(cop)).toBe('000011')
    expect(cop.inertia).toBeCloseTo(23.25, 12)
    expect(cop.violations).toBe(0)
  })

  it('COP with the closure of must-link 4–6 and cannot-link 1–6: {0,1}, {4,6,9,10}', () => {
    const closed = run(
      constrainedKMeansSteps(x, { k: 2, rule: 'cop', mustLink: [[2, 3]], cannotLink: [[1, 3]] }),
      start,
      20,
    )
    expect(lab(closed)).toBe('001111')
    expect(closed.failed).toBe(false)
  })

  it('PCK at μ = (5/3, 25/3): 6 joins 4 only when w > 20/3', () => {
    const mid = { centroids: [[5 / 3], [25 / 3]] }
    const at = (weight: number) =>
      constrainedKMeansSteps(x, { k: 2, rule: 'pck', weight, mustLink: [[2, 3]] }).init(mid, stream(0))
    expect(at(6.6).labels[3]).toBe(1)
    expect(at(6.7).labels[3]).toBe(0)
  })

  it('COP fails, keeping the partial pass, when a point has no legal cluster; seeds go to their cluster', () => {
    // 0 cannot share with 1 or 2, and 1 cannot share with 2: three points, two clusters.
    const cannot: ConstraintPair[] = [
      [0, 1],
      [0, 2],
      [1, 2],
    ]
    const st = constrainedKMeansSteps([[0], [1], [2]], { k: 2, rule: 'cop', cannotLink: cannot }).init(
      { centroids: [[0], [2]] },
      stream(0),
    )
    expect(st.failed).toBe(true)
    expect(st.converged).toBe(true)
    expect(Array.from(st.labels)).toEqual([0, 1, -1])
    const seeded = constrainedKMeansSteps(x, { k: 2, rule: 'cop', fixed: [-1, -1, -1, -1, -1, 0] }).init(
      start,
      stream(0),
    )
    expect(seeded.labels[5]).toBe(0)
    expect(() => constrainedKMeansSteps(x, { k: 2, rule: 'cop' }).init({ centroids: [[0]] }, stream(0))).toThrow(
      /starting centroids/,
    )
  })
})

describe('constraintClosure', () => {
  it('adds cannot-link 1–4 from must-link 4–6 and cannot-link 1–6, and flags a contradiction', () => {
    const cl = constraintClosure(6, [[2, 3]], [[1, 3]])
    expect(cl.members).toEqual([[0], [1], [2, 3], [4], [5]])
    expect(cl.cannotLink[2]).toContain(1)
    expect(cl.cannotLink[1]).toEqual([2, 3])
    expect(cl.mustLink[3]).toEqual([2])
    expect(cl.consistent).toBe(true)
    expect(constraintClosure(3, [[0, 1]], [[0, 1]]).consistent).toBe(false)
    expect(() => constraintClosure(3, [[0, 3]], [])).toThrow(/out of range/)
  })
})

describe('PCK-means and RCA on two long bars', () => {
  const s = stream(3)
  const zx = toFlat(normals(child(s, 'a'), 120))
  const zy = toFlat(normals(child(s, 'b'), 120))
  const X = Array.from(zx, (v, i) => [2.5 * v, (i < 60 ? 0 : 1.6) + 0.25 * zy[i]])
  const pairs: [number, number][] = []
  for (let i = 0; i < 40; i++) pairs.push([(i * 7) % 120, (i * 13 + 5) % 120])
  const ml = pairs.filter(([a, b]) => a !== b && a < 60 === b < 60)
  const cl = pairs.filter(([a, b]) => a < 60 !== b < 60)

  it('PCK-means never raises its objective', () => {
    const alg = constrainedKMeansSteps(X, { k: 2, rule: 'pck', weight: 2, mustLink: ml, cannotLink: cl })
    let st = alg.init(
      {
        centroids: [
          [-2, 0.8],
          [2, 0.8],
        ],
      },
      stream(0),
    )
    for (let t = 0; t < 40 && !st.converged; t++) {
      const next = alg.step(st, { t, stream: stream(0) })
      expect(next.objective).toBeLessThanOrEqual(st.objective + 1e-9)
      st = next
    }
    expect(st.converged).toBe(true)
    expect(st.objective).toBeCloseTo(218.4007840307389, 9)
  })

  it('RCA applied twice gives the identity chunklet covariance', () => {
    const groups = constraintClosure(120, ml, cl).members.filter((g) => g.length > 1)
    const m = relevantComponentMetric(X, groups)
    const again = relevantComponentMetric(m.transformed, groups)
    expect(again.covariance[0][0]).toBeCloseTo(1, 6)
    expect(again.covariance[1][1]).toBeCloseTo(1, 6)
    expect(again.covariance[0][1]).toBeCloseTo(0, 6)
    // The whitening is symmetric and stretches the narrow direction.
    expect(m.whitening[0][1]).toBeCloseTo(m.whitening[1][0], 14)
    expect(m.whitening[1][1]).toBeCloseTo(4.776985529376478, 9)
    expect(() => relevantComponentMetric(X, [[0], [1]])).toThrow(/no chunklet/)
  })
})

describe('seededCentroids', () => {
  it('averages the seeds of each cluster and falls back for a cluster without any', () => {
    expect(seededCentroids(x, [0, 0, -1, -1, 1, 1], 3, [[0], [0], [5]])).toEqual([[0.5], [9.5], [5]])
  })

  it('is registered', () => {
    expect(clusteringAlgorithms.constrainedKMeansSteps.info.notes).toContain('semi-supervised-clustering')
    expect(clusteringFunctions.relevantComponentMetric.info.cite).toEqual(['barhillel2005rca'])
  })
})
