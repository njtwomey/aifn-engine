/**
 * The generators added in v0.2.10 (aifn-compute/graph/structures): greedy b-matching, the stochastic block model,
 * random recursive trees, the barbell and the hypergraph expansions, checked against the site candidates' self-checks
 * and their laws.
 */
import { describe, expect, it } from 'vitest'
import {
  barbellGraph,
  blockMembership,
  bMatchingGraph,
  cliqueExpansion,
  erdosRenyiGraph,
  randomRecursiveTree,
  starExpansion,
  stochasticBlockModel,
} from 'aifn-compute/graph/structures'
import { edgeHomophily } from 'aifn-compute/graph/measures'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'

const pairs = (g: Graph) => g.edges.map((e) => `${e.from}-${e.to}`)
const degrees = (g: Graph) => {
  const d = new Array<number>(g.nodes).fill(0)
  for (const e of g.edges) {
    d[e.from]++
    d[e.to]++
  }
  return d
}

/** The site's hypergraph-neural-networks fixture: six authors and two hypergraphs with one clique expansion. */
const HYPEREDGES = {
  papers: [
    [0, 1, 2],
    [2, 3],
    [3, 4, 5],
  ],
  split: [
    [0, 1],
    [1, 2],
    [0, 2],
    [2, 3],
    [3, 4, 5],
  ],
}

describe('bMatchingGraph', () => {
  it('joins the closest pairs while both ends are below b (site check: points 0, 1, 3)', () => {
    const x = [[0], [1], [3]]
    expect(pairs(bMatchingGraph(x, 1))).toEqual(['0-1'])
    expect(pairs(bMatchingGraph(x, 2))).toEqual(['0-1', '0-2', '1-2'])
    expect(bMatchingGraph(x, 1).edges[0].weight).toBe(1)
    expect(bMatchingGraph(x, 1).directed).toBe(false)
  })
  it('keeps every degree at most b', () => {
    const x = toRows(uniform(stream(0), 0, 1, { shape: [60, 2] })) as number[][]
    const g = bMatchingGraph(x, 4)
    expect(Math.max(...degrees(g))).toBeLessThanOrEqual(4)
    // Greedy b-matching leaves few nodes short of b.
    expect(g.edges.length).toBeGreaterThanOrEqual(90)
  })
  it('rejects b that is not a positive integer', () => {
    expect(() => bMatchingGraph([[0], [1]], 0)).toThrow()
  })
})

describe('stochasticBlockModel', () => {
  it('numbers nodes block by block', () => {
    expect(Array.from(toFlat(blockMembership([2, 3, 1])))).toEqual([0, 0, 1, 1, 1, 2])
  })
  it("reproduces the site's block-model graph (sizes [40, 40], pIn 0.12, pOut 0.02, stream 3)", () => {
    const g = stochasticBlockModel(stream(3), [40, 40], { pIn: 0.12, pOut: 0.02 })
    expect(g.edges.length).toBe(227)
    expect(edgeHomophily(g, blockMembership([40, 40]))).toBeCloseTo(0.8678414096916299, 12)
  })
  it('equal probabilities give the Erdős–Rényi graph of the same stream', () => {
    const a = stochasticBlockModel(stream(0), [10, 10], { pIn: 0.3, pOut: 0.3 })
    const b = erdosRenyiGraph(stream(0), 20, 0.3)
    expect(pairs(a)).toEqual(pairs(b))
  })
  it('a probability matrix: zero between blocks keeps every edge inside a block', () => {
    const g = stochasticBlockModel(
      stream(1),
      [5, 5],
      [
        [0.9, 0],
        [0, 0.2],
      ],
    )
    expect(g.edges.every((e) => e.from < 5 === e.to < 5)).toBe(true)
  })
  it('mean homophily over five seeds matches the catalogue probes (0.854 and 0.494 expected)', () => {
    const y = blockMembership([40, 40])
    const mean = (pIn: number, pOut: number) =>
      [0, 1, 2, 3, 4].reduce(
        (a, seed) => a + edgeHomophily(stochasticBlockModel(stream(seed), [40, 40], { pIn, pOut }), y) / 5,
        0,
      )
    expect(Math.abs(mean(0.12, 0.02) - 0.854)).toBeLessThan(0.03)
    expect(Math.abs(mean(0.06, 0.06) - 0.494)).toBeLessThan(0.04)
  })
  it('rejects bad probabilities, shapes and asymmetric undirected matrices', () => {
    expect(() => stochasticBlockModel(stream(0), [3, 3], { pIn: 1.2, pOut: 0 })).toThrow()
    expect(() => stochasticBlockModel(stream(0), [3, 3], [[0.5]])).toThrow()
    expect(() =>
      stochasticBlockModel(
        stream(0),
        [3, 3],
        [
          [0.5, 0.1],
          [0.2, 0.5],
        ],
      ),
    ).toThrow()
  })
})

describe('randomRecursiveTree', () => {
  it('has n - 1 edges, each from an earlier node, and successive calls on one stream differ', () => {
    const s = stream(0)
    const a = randomRecursiveTree(s, 12)
    const b = randomRecursiveTree(s, 12)
    expect(a.edges.length).toBe(11)
    expect(a.edges.every((e) => e.from < e.to)).toBe(true)
    expect(pairs(a)).not.toEqual(pairs(b))
    expect(pairs(randomRecursiveTree(stream(0), 12))).toEqual(pairs(a))
  })
  it('draws parent of v as floor(u v) from the stream itself', () => {
    const u = toFlat(uniform(stream(5), 0, 1, { shape: [4] }))
    const t = randomRecursiveTree(stream(5), 5)
    expect(t.edges.map((e) => e.from)).toEqual([0, 1, 2, 3].map((k) => Math.floor(u[k] * (k + 1))))
  })
})

describe('barbellGraph', () => {
  it('barbell(5, 3): 13 nodes and 24 edges (site check), networkx numbering', () => {
    const g = barbellGraph(5, 3)
    expect([g.nodes, g.edges.length]).toEqual([13, 24])
    expect(pairs(g).slice(20)).toEqual(['4-5', '5-6', '6-7', '7-8'])
  })
  it('path length 0 joins the cliques by one edge', () => {
    expect(pairs(barbellGraph(3, 0)).slice(6)).toEqual(['2-3'])
    expect(() => barbellGraph(1, 2)).toThrow()
  })
})

describe('hypergraph expansions', () => {
  it('both site hypergraphs have the same 7-edge clique expansion, all weights 1', () => {
    const key = (g: Graph) =>
      g.edges
        .map((e) => `${e.from}-${e.to}:${e.weight}`)
        .sort()
        .join(' ')
    expect(key(cliqueExpansion(6, HYPEREDGES.papers))).toBe(key(cliqueExpansion(6, HYPEREDGES.split)))
    expect(cliqueExpansion(6, HYPEREDGES.papers).edges.length).toBe(7)
  })
  it('a pair in two hyperedges gets weight 2', () => {
    const g = cliqueExpansion(3, [
      [0, 1],
      [0, 1, 2],
    ])
    expect(g.edges.map((e) => [e.from, e.to, e.weight])).toEqual([
      [0, 1, 2],
      [0, 2, 1],
      [1, 2, 1],
    ])
  })
  it('star expansion: one directed edge per membership each way', () => {
    const { toEdges, toNodes } = starExpansion(6, HYPEREDGES.papers)
    expect(toEdges.nodes).toBe(9)
    expect(pairs(toEdges).slice(0, 4)).toEqual(['0-6', '1-6', '2-6', '2-7'])
    expect(pairs(toNodes).slice(0, 2)).toEqual(['6-0', '6-1'])
  })
  it('rejects empty hyperedges, repeated members and members outside the nodes', () => {
    expect(() => cliqueExpansion(3, [[]])).toThrow()
    expect(() => cliqueExpansion(3, [[0, 0]])).toThrow()
    expect(() => starExpansion(3, [[0, 3]])).toThrow()
  })
})
