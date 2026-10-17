/** Hop distances and k-hop neighbourhoods (aifn-compute/graph/traversal), against the site's self-checks. */
import { describe, expect, it } from 'vitest'
import { hopDistances, kHopNeighbourhood } from 'aifn-compute/graph/traversal'
import { floydWarshall } from 'aifn-compute/graph/shortest-paths'
import { erdosRenyiGraph } from 'aifn-compute/graph/structures'
import { fromEdges } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'

/** The triangle 0-1-2 with a tail 2-3 (the message-passing note's graph). */
const tail = fromEdges(
  4,
  [
    [0, 1],
    [0, 2],
    [1, 2],
    [2, 3],
  ],
  { directed: false },
)

describe('hopDistances', () => {
  it('row 0 of the triangle with a tail is [0, 1, 1, 2] (graphormer check)', () => {
    expect((toRows(hopDistances(tail)) as number[][])[0]).toEqual([0, 1, 1, 2])
  })
  it('equals Floyd-Warshall on a unit-weight copy, Infinity where unreachable', () => {
    const g = erdosRenyiGraph(stream(3), 12, 0.15, { directed: true })
    const unit = { ...g, edges: g.edges.map((e) => ({ from: e.from, to: e.to })) }
    expect(Array.from(toFlat(hopDistances(g)))).toEqual(Array.from(toFlat(floydWarshall(unit).distance)))
  })
})

describe('kHopNeighbourhood', () => {
  it('from node 3 of the triangle with a tail: sizes [1, 2, 4] (computation-graph check)', () => {
    const r = kHopNeighbourhood(tail, [3], 2)
    expect(r.sizes).toEqual([1, 2, 4])
    expect(Array.from(toFlat(r.nodes))).toEqual([3, 2, 0, 1])
    expect(Array.from(toFlat(r.depth))).toEqual([2, 2, 1, 0])
  })
  it('distances are to the nearest source, and repeated sources count once', () => {
    const path = fromEdges(
      5,
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
      ],
      { directed: false },
    )
    const r = kHopNeighbourhood(path, [0, 4, 0], 1)
    expect(Array.from(toFlat(r.depth))).toEqual([0, 1, -1, 1, 0])
    expect(r.sizes).toEqual([2, 4])
  })
  it('follows edges forwards or backwards on a directed graph', () => {
    const g = fromEdges(4, [
      [0, 1],
      [1, 2],
      [2, 3],
    ])
    expect(Array.from(toFlat(kHopNeighbourhood(g, 1, 2).nodes))).toEqual([1, 2, 3])
    expect(Array.from(toFlat(kHopNeighbourhood(g, 1, 2, { direction: 'in' }).nodes))).toEqual([1, 0])
  })
  it('throws for a source outside the graph or a negative k', () => {
    expect(() => kHopNeighbourhood(tail, [4], 1)).toThrow()
    expect(() => kHopNeighbourhood(tail, [0], -1)).toThrow()
  })
})
