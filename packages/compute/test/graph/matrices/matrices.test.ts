/** Graph matrices (aifn-compute/graph/matrices): adjacency, degree, Laplacians and incidence. */
import { describe, expect, it } from 'vitest'
import {
  adjacencyMatrix,
  degreeMatrix,
  degrees,
  hypergraphIncidence,
  incidenceMatrix,
  laplacian,
  rescaledLaplacian,
} from 'aifn-compute/graph/matrices'
import { fromEdges } from 'aifn-compute/graph'
import { cycleGraph } from 'aifn-compute/graph/structures'
import { eigh } from 'aifn-compute/numerics/linalg'
import { diag, matmul, sub, sum, tensor, toFlat, toRows, transpose } from 'aifn-compute/foundation/tensor'
import { unweightedShortestPaths } from 'aifn-compute/graph/traversal'

const g = fromEdges(
  3,
  [
    [0, 1, 2],
    [1, 2, 3],
  ],
  { directed: false },
)

describe('graph matrices', () => {
  it('adjacency (weighted or counted), degrees and the Laplacian L = D − A', () => {
    expect(toRows(adjacencyMatrix(g))).toEqual([
      [0, 2, 0],
      [2, 0, 3],
      [0, 3, 0],
    ])
    expect(toRows(adjacencyMatrix(g, { weighted: false }))).toEqual([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ])
    expect(toFlat(degrees(g))).toEqual([2, 5, 3])
    expect(toRows(laplacian(g))).toEqual(toRows(sub(degreeMatrix(g), adjacencyMatrix(g))))
  })
  it('the Laplacian is BBᵀ for the oriented incidence matrix, with rows summing to 0 and a zero eigenvalue', () => {
    const B = incidenceMatrix(g, { oriented: true, weighted: false })
    expect(toRows(matmul(B, transpose(B)))).toEqual(toRows(laplacian(g, { weighted: false })))
    for (const row of toRows(laplacian(g))) expect(row.reduce((a, b) => a + b, 0)).toBe(0)
    const values = toFlat(eigh(laplacian(cycleGraph(6))).values)
    expect(Math.min(...values.map(Math.abs))).toBeLessThan(1e-12)
    // The normalised Laplacians have spectrum in [0, 2].
    for (const normalisation of ['symmetric', 'random-walk'] as const)
      expect(sum(laplacian(g, { normalisation })) as number).not.toBeNaN()
    const sym = toFlat(eigh(laplacian(cycleGraph(6), { normalisation: 'symmetric' })).values)
    for (const v of sym) expect(v > -1e-12 && v < 2 + 1e-12).toBe(true)
  })
})

describe('laplacian of an affinity matrix (replaces the site candidate affinityLaplacian)', () => {
  it('the weighted path [[1, -1, 0], [-1, 3, -2], [0, -2, 2]] (semi-supervised check)', () => {
    const W = [
      [0, 1, 0],
      [1, 0, 2],
      [0, 2, 0],
    ]
    expect(toRows(laplacian(W))).toEqual([
      [1, -1, 0],
      [-1, 3, -2],
      [0, -2, 2],
    ])
    expect(toRows(laplacian(tensor(W)))).toEqual(toRows(laplacian(W)))
  })
  it('the diagonal (self-loops) cancels in the unnormalised form', () => {
    const W = [
      [5, 1],
      [1, 7],
    ]
    expect(toRows(laplacian(W))).toEqual([
      [1, -1],
      [-1, 1],
    ])
    expect(() => laplacian([[0, 1]])).toThrow()
  })
})

/** The 6-node graph of the spectral-graph-convolution note: a triangle 0-1-2, then 2-3, 3-4 and 3-5. */
const g6 = fromEdges(
  6,
  [
    [0, 1],
    [0, 2],
    [1, 2],
    [2, 3],
    [3, 4],
    [3, 5],
  ],
  { directed: false },
)

describe('rescaledLaplacian', () => {
  it('symmetric by default with lambda max 2; unnormalised with 2 d_max = 6', () => {
    const sym = rescaledLaplacian(g6)
    expect(sym.lambdaMax).toBe(2)
    const L = laplacian(g6, { weighted: false, normalisation: 'symmetric' })
    expect(toFlat(sym.lTilde)).toEqual(toFlat(sub(L, diag(tensor([1, 1, 1, 1, 1, 1])))))
    expect(rescaledLaplacian(g6, { normalisation: 'unnormalised' }).lambdaMax).toBe(6)
  })
  it('the spectrum lies in [-1, 1]', () => {
    const values = toFlat(eigh(rescaledLaplacian(g6).lTilde).values)
    for (const v of values) expect(Math.abs(v)).toBeLessThanOrEqual(1 + 1e-12)
  })
  it('powers are local: (L tilde)^k is zero beyond k hops', () => {
    const { lTilde } = rescaledLaplacian(g6)
    const hops = toFlat(unweightedShortestPaths(g6, 0).distance)
    const row = toRows(matmul(lTilde, lTilde))[0] as number[]
    row.forEach((v, i) => {
      if (hops[i] > 2) expect(v === 0).toBe(true)
    })
  })
})

describe('hypergraphIncidence', () => {
  it("the site's coauthor hypergraph: author 3's row is [1, 1, 0]", () => {
    const papers = [
      [0, 1, 2],
      [2, 3],
      [3, 4, 5],
    ]
    const H = toRows(hypergraphIncidence(6, papers)) as number[][]
    expect(H[2]).toEqual([1, 1, 0])
    expect(H.map((r) => r.reduce((a, b) => a + b, 0))).toEqual([1, 1, 2, 2, 1, 1])
  })
  it('throws for an empty hyperedge', () => {
    expect(() => hypergraphIncidence(3, [[]])).toThrow()
  })
})
