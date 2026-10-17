/** Link-prediction heuristics (aifn-compute/graph/measures), against the site's self-check. */
import { describe, expect, it } from 'vitest'
import { pairHeuristics } from 'aifn-compute/graph/measures'
import { fromEdges } from 'aifn-compute/graph'
import { toFlat } from 'aifn-compute/foundation/tensor'

describe('pairHeuristics', () => {
  it('triangle 0-1-2 with tail 2-3: one common neighbour of degree 3 for (0, 3) and (1, 3)', () => {
    const g = fromEdges(
      4,
      [
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 3],
      ],
      { directed: false },
    )
    const h = pairHeuristics(g, [
      [0, 3],
      [1, 3],
    ])
    expect(Array.from(toFlat(h.commonNeighbours))).toEqual([1, 1])
    for (const v of toFlat(h.adamicAdar)) expect(v).toBeCloseTo(1 / Math.log(3), 12)
  })
  it('ignores self-loops and repeated edges; pairs without common neighbours score 0', () => {
    const g = fromEdges(
      3,
      [
        [0, 1],
        [0, 1],
        [1, 1],
        [1, 2],
      ],
      { directed: false },
    )
    const h = pairHeuristics(g, [
      [0, 2],
      [0, 1],
    ])
    expect(Array.from(toFlat(h.commonNeighbours))).toEqual([1, 0])
    expect(toFlat(h.adamicAdar)[0]).toBeCloseTo(1 / Math.log(2), 12)
  })
  it('throws for a pair outside the graph', () => {
    expect(() => pairHeuristics(fromEdges(2, [[0, 1]], { directed: false }), [[0, 5]])).toThrow()
  })
})
