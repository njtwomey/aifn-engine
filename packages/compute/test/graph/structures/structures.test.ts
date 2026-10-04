/** Standard graphs (aifn-compute/graph/structures), checked by node and edge counts, degrees and the generators' laws. */
import { describe, expect, it } from 'vitest'
import {
  balancedTreeGraph,
  barabasiAlbertGraph,
  chainGraph,
  completeBipartiteGraph,
  completeGraph,
  cycleGraph,
  epsilonBallGraph,
  erdosRenyiGraph,
  gridGraph,
  kNearestNeighbourGraph,
  randomDag,
  starGraph,
  wattsStrogatzGraph,
} from 'aifn-compute/graph/structures'
import { degrees } from 'aifn-compute/graph/matrices'
import { isDag } from 'aifn-compute/graph/traversal'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

describe('deterministic structures', () => {
  it('node and edge counts', () => {
    expect([chainGraph(5).nodes, chainGraph(5).edges.length]).toEqual([5, 4])
    expect([cycleGraph(5).nodes, cycleGraph(5).edges.length]).toEqual([5, 5])
    expect([starGraph(4).nodes, starGraph(4).edges.length]).toEqual([5, 4])
    expect(completeGraph(5).edges.length).toBe(10)
    expect(completeGraph(5, { directed: true }).edges.length).toBe(20)
    expect(completeBipartiteGraph(2, 3).edges.length).toBe(6)
    // A balanced binary tree of depth 3 has 15 nodes and 14 edges.
    expect([balancedTreeGraph(2, 3).nodes, balancedTreeGraph(2, 3).edges.length]).toEqual([15, 14])
    // A 3 × 4 lattice: 3·3 + 2·4 = 17 edges with 4 neighbours.
    expect([gridGraph(3, 4).nodes, gridGraph(3, 4).edges.length]).toEqual([12, 17])
    expect(() => cycleGraph(2)).toThrow()
  })
  it('degrees: every node of a cycle has degree 2, the star centre n', () => {
    expect(toFlat(degrees(cycleGraph(6)))).toEqual([2, 2, 2, 2, 2, 2])
    expect(toFlat(degrees(starGraph(4)))[0]).toBe(4)
  })
})

describe('random structures', () => {
  it('are reproducible from their stream and obey their laws', () => {
    const a = erdosRenyiGraph(stream(1), 30, 0.2)
    expect(erdosRenyiGraph(stream(1), 30, 0.2).edges).toEqual(a.edges)
    // About p·n(n − 1)/2 = 87 edges.
    expect(Math.abs(a.edges.length - 87)).toBeLessThan(30)
    expect(isDag(randomDag(stream(2), 12, 0.4))).toBe(true)
    // Watts–Strogatz keeps nk/2 edges; Barabási–Albert starts from a star of m edges and adds m per new node.
    expect(wattsStrogatzGraph(stream(3), 20, 4, 0.3).edges.length).toBe(40)
    expect(barabasiAlbertGraph(stream(4), 20, 2).edges.length).toBe(2 + 2 * (20 - 2 - 1))
  })
})

describe('graphs from points', () => {
  const pts = [
    [0, 0],
    [1, 0],
    [0, 1],
    [5, 5],
  ]
  it('k nearest neighbours and ε-balls', () => {
    const g = kNearestNeighbourGraph(pts, 1, { mode: 'directed' })
    expect(g.edges.length).toBe(4)
    // The far point's nearest neighbour is one of the three near it.
    expect(g.edges.find((e) => e.from === 3)!.to).toBeLessThan(3)
    const e = epsilonBallGraph(pts, 1.01)
    expect(e.edges.map((x) => [x.from, x.to])).toEqual([
      [0, 1],
      [0, 2],
    ])
  })
})
