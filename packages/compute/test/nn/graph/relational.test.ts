/** R-GCN (aifn-compute/nn/graph), against the site's self-check (rgcn-conv) on its bibliography graph. */
import { describe, expect, it } from 'vitest'
import { basisWeights, rgcnConv, rgcnTerms } from 'aifn-compute/nn/graph'
import type { Graph } from 'aifn-compute/graph'
import { tensor, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'

// The site's heterogeneous-graph-networks/bibliography.ts: 8 authors, 8 papers and 4 venues, three relations into
// papers.
const [AUTHORS, PAPERS, VENUES] = [8, 8, 4]
const N = AUTHORS + PAPERS + VENUES
const author = (a: number) => a - 1
const paper = (p: number) => AUTHORS + p - 1
const venue = (v: number) => AUTHORS + PAPERS + v - 1
const FEATURES = [
  [0, 1],
  [1, 1],
  [2, 1],
  [0, 2],
  [1, 0],
  [2, 2],
  [0.5, 1.5],
  [1.5, 0.5],
  [1, 0],
  [2, 0],
  [0, 2],
  [1, 1],
  [2, 1],
  [0, 1],
  [1, 2],
  [2, 2],
  [1, -1],
  [-1, 1],
  [2, 0],
  [0, -2],
]
const WRITTEN_BY: Record<number, number[]> = {
  1: [1, 2],
  2: [2, 3],
  3: [4],
  4: [1, 5, 6],
  5: [3, 7],
  6: [8],
  7: [6, 7, 8],
  8: [4, 5],
}
const CITES: Record<number, number[]> = {
  1: [2, 3],
  2: [],
  3: [],
  4: [1, 2],
  5: [1, 3, 4],
  6: [5],
  7: [4, 6],
  8: [1, 5, 6, 7],
}
const PUBLISHED_IN: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 4, 7: 2, 8: 3 }
const directed = (edges: [number, number][]): Graph => ({
  kind: 'graph',
  nodes: N,
  edges: edges.map(([from, to]) => ({ from, to })),
  directed: true,
})
const GRAPHS = [
  directed(Object.entries(WRITTEN_BY).flatMap(([p, as]) => as.map((a): [number, number] => [author(a), paper(+p)]))),
  directed(Object.entries(CITES).flatMap(([p, cs]) => cs.map((c): [number, number] => [paper(c), paper(+p)]))),
  directed(Object.entries(PUBLISHED_IN).map(([p, v]): [number, number] => [venue(v), paper(+p)])),
]
const WEIGHTS = [
  [
    [0, 1],
    [1, 0],
  ],
  [
    [0.5, 0],
    [0, 0.5],
  ],
  [
    [1, 0],
    [0, 0],
  ],
]
const h = tensor(FEATURES)
const relations = GRAPHS.map((graph, r) => ({ graph, weight: tensor(WEIGHTS[r]) }))
const W0 = tensor([
  [1, 0],
  [0, 1],
])
const row = (v: Value, i: number) => (toRows(v as Tensor) as number[][])[i]
const close = (got: number[], want: number[]) => want.forEach((w, i) => expect(got[i]).toBeCloseTo(w, 12))

describe('rgcnTerms and rgcnConv', () => {
  it("P1's terms: written by (1, 0.5), cites (0.5, 0.5), published in (1, 0)", () => {
    const terms = rgcnTerms(h, relations)
    close(row(terms[0], paper(1)), [1, 0.5])
    close(row(terms[1], paper(1)), [0.5, 0.5])
    close(row(terms[2], paper(1)), [1, 0])
    // P2 cites nothing: a zero row.
    close(row(terms[1], paper(2)), [0, 0])
  })
  it('P1 per relation (3.5, 1), per node (1.8, 0.4)', () => {
    close(row(rgcnConv(h, relations, W0), paper(1)), [3.5, 1])
    close(row(rgcnConv(h, relations, W0, 'node'), paper(1)), [1.8, 0.4])
  })
  it('edge weights do not scale the messages', () => {
    const weighted = relations.map((r) => ({
      ...r,
      graph: { ...r.graph, edges: r.graph.edges.map((e) => ({ ...e, weight: 7 })) },
    }))
    close(row(rgcnConv(h, weighted, W0), paper(1)), [3.5, 1])
  })
})

describe('basisWeights', () => {
  it('W_2 = V1 / 2 and W_3 = V1 - V2', () => {
    const V1 = tensor([
      [1, 0],
      [0, 1],
    ])
    const V2 = tensor([
      [0, 1],
      [1, 0],
    ])
    const coefficients = [
      [0, 1],
      [0.5, 0],
      [1, -1],
    ]
    const W = basisWeights(coefficients, [V1, V2])
    expect(toRows(W[1] as Tensor)).toEqual([
      [0.5, 0],
      [0, 0.5],
    ])
    expect(toRows(W[2] as Tensor)).toEqual([
      [1, -1],
      [-1, 1],
    ])
    expect(() => basisWeights([[1]], [V1, V2])).toThrow()
  })
})
