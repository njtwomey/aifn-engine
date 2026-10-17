/**
 * `harmonicExtension` on the site's self-check: on the path 0 - 1 - 2 - 3 with a full Gaussian graph ($\gamma = 0.5$)
 * and labels at the ends, the extension at node 1 is its harmonic score 0.379485; and the note's five-node refit.
 */
import { describe, expect, it } from 'vitest'
import { harmonicExtension, harmonicLabels, pointAffinity } from 'aifn-compute/graph/propagation'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'

describe('harmonicExtension', () => {
  const X = [[0], [1], [2], [3]]
  const gamma = 0.5
  const W = X.map((a) => X.map((b) => (a === b ? 0 : Math.exp(-gamma * (a[0] - b[0]) ** 2))))
  const h = harmonicLabels(W, [0, -1, -1, 1], { classes: 2 })
  const F = toRows(h.scores) as number[][]

  it('reproduces the harmonic score at an unlabelled node, 0.379485', () => {
    const ext = harmonicExtension(X, F, [[1]], { gamma })
    const e = toRows(ext.scores) as number[][]
    expect(Math.abs(e[0][1] - F[1][1])).toBeLessThan(1e-9)
    expect(F[1][1]).toBeCloseTo(0.379485, 6)
    expect(toFlat(ext.labels)).toEqual([0])
    expect(ext.labels.dtype).toBe('int32')
  })

  it('averages only the k nearest nodes, and keeps far queries finite', () => {
    const near = toRows(harmonicExtension(X, F, [[2.9]], { gamma, neighbours: 1 }).scores) as number[][]
    expect(near[0]).toEqual(F[3])
    const far = toRows(harmonicExtension(X, F, [[1e4]], { gamma }).scores) as number[][]
    expect(far[0][1]).toBeCloseTo(1, 12)
  })

  it('agrees with pointAffinity: the same kernel and neighbours', () => {
    const scores = harmonicLabels(pointAffinity(X, { gamma }), [0, -1, -1, 1]).scores
    const e = toRows(harmonicExtension(X, scores, [[2]], { gamma }).scores) as number[][]
    expect(e[0][1]).toBeCloseTo((toRows(scores) as number[][])[2][1], 9)
  })

  it('refuses a non-positive gamma and mismatched rows', () => {
    expect(() => harmonicExtension(X, F, [[0]], { gamma: 0 })).toThrow(/gamma/)
    expect(() => harmonicExtension(X, F.slice(1), [[0]], { gamma })).toThrow(/rows/)
  })
})

describe('harmonicLabels on the note’s five-node refit', () => {
  it('gives f2 = 6/13, f3 = 3/13 and f5 = 2/13', () => {
    const W5 = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 1],
      [0, 1, 0, 1, 0],
      [0, 0, 1, 0, 2],
      [0, 1, 0, 2, 0],
    ]
    const h5 = toRows(harmonicLabels(W5, [1, -1, -1, 0, -1], { classes: 2 }).scores) as number[][]
    expect(h5[1][1]).toBeCloseTo(6 / 13, 9)
    expect(h5[2][1]).toBeCloseTo(3 / 13, 9)
    expect(h5[4][1]).toBeCloseTo(2 / 13, 9)
  })
})
