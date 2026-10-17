/** Subgraph policies and DS-WL (aifn-compute/graph/isomorphism), against the site's self-check (subgraph-policies). */
import { describe, expect, it } from 'vitest'
import { subgraphBag, subgraphWeisfeilerLeman, type SubgraphPolicy } from 'aifn-compute/graph/isomorphism'
import type { Graph } from 'aifn-compute/graph'
import { bicyclopentyl, decalin, hexagon, rook, shrikhande, twoTriangles } from './pairs'

const POLICIES: SubgraphPolicy[] = ['node-deleted', 'edge-deleted', 'ego']
const separated = (a: Graph, b: Graph) => POLICIES.map((p) => subgraphWeisfeilerLeman([a, b], p).distinguished)

describe('subgraphBag', () => {
  it('the 6-cycle has 6 subgraphs under every policy, each on all 6 nodes', () => {
    expect(POLICIES.map((p) => subgraphBag(hexagon, p).length)).toEqual([6, 6, 6])
    expect(subgraphBag(hexagon, 'ego').every((s) => s.graph.nodes === 6)).toBe(true)
  })
  it('a depth-1 ego network of the 6-cycle keeps the two edges at its root, which is marked', () => {
    const s = subgraphBag(hexagon, 'ego')[0]
    expect(s.graph.edges.map((e) => [e.from, e.to])).toEqual([
      [0, 1],
      [5, 0],
    ])
    expect(s.initial).toEqual([1, 0, 0, 0, 0, 0])
  })
  it('throws for a directed graph', () => {
    expect(() => subgraphBag({ kind: 'graph', nodes: 2, edges: [{ from: 0, to: 1 }] }, 'ego')).toThrow()
  })
})

describe('subgraphWeisfeilerLeman', () => {
  it('6-cycle vs two triangles: separated by ND, ED and EGO+', () => {
    expect(separated(hexagon, twoTriangles)).toEqual([true, true, true])
  })
  it('decalin vs bicyclopentyl: deletion separates, depth-1 ego networks do not', () => {
    expect(separated(decalin, bicyclopentyl)).toEqual([true, true, false])
  })
  it("rook's graph vs Shrikhande: only edge deletion separates (Bevilacqua et al. 2022, proposition 3)", () => {
    expect(separated(rook, shrikhande)).toEqual([false, true, false])
  })
})
