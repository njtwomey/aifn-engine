/** Rewiring (aifn-compute/graph/rewiring), against the site's self-check (depth-long-range). */
import { describe, expect, it } from 'vitest'
import { fosrRewire, withVirtualNode } from 'aifn-compute/graph/rewiring'
import { barbellGraph, chainGraph, completeGraph } from 'aifn-compute/graph/structures'
import { adjacencyMatrix } from 'aifn-compute/graph/matrices'
import { eigh } from 'aifn-compute/numerics/linalg'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { fromEdges, type Graph } from 'aifn-compute/graph'

/** The spectral gap 1 - mu_2 of D^{-1/2} A D^{-1/2}, as the site's candidate computed it. */
function gap(g: Graph): number {
  const n = g.nodes
  const a = toFlat(adjacencyMatrix(g))
  const d = Array.from({ length: n }, (_, i) => a.slice(i * n, i * n + n).reduce((s, v) => s + v, 0))
  const m = new Float64Array(n * n).map((_, k) => a[k] / Math.sqrt(d[Math.floor(k / n)] * d[k % n]))
  return 1 - toFlat(eigh(fromData(m, [n, n])).values)[1]
}

describe('withVirtualNode', () => {
  it('adds node V joined to every node', () => {
    const g = withVirtualNode(barbellGraph(5, 3))
    expect(g.nodes).toBe(14)
    expect(g.edges.length).toBe(24 + 13)
    expect(g.edges.slice(24).every((e, v) => e.from === v && e.to === 13)).toBe(true)
  })
  it('both directions for a directed graph, and a label for the new node', () => {
    const g = withVirtualNode(fromEdges(2, [[0, 1]], { labels: ['a', 'b'] }))
    expect(g.edges.map((e) => [e.from, e.to])).toEqual([
      [0, 1],
      [0, 2],
      [2, 0],
      [1, 2],
      [2, 1],
    ])
    expect(g.labels).toEqual(['a', 'b', 'virtual'])
  })
})

describe('fosrRewire', () => {
  it('barbell(5, 3): adds (0, 9), (1, 10), (2, 11); the gap goes from 0.0217 to 0.234', () => {
    const g = barbellGraph(5, 3)
    expect(gap(g)).toBeCloseTo(0.0217, 4)
    const r = fosrRewire(g, 3)
    expect(r.added).toEqual([
      [0, 9],
      [1, 10],
      [2, 11],
    ])
    expect(gap(r.graph)).toBeCloseTo(0.234, 3)
  })
  it('stops when the graph is complete, and rejects directed graphs', () => {
    expect(fosrRewire(completeGraph(4), 2).added).toEqual([])
    expect(fosrRewire(chainGraph(3), 5).added.length).toBe(1)
    expect(() => fosrRewire(fromEdges(2, [[0, 1]]), 1)).toThrow()
  })
})
