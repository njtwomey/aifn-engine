/** Graph matrices (aifn-compute/graph/matrices): adjacency, degree, Laplacians and incidence. */
import { describe, expect, it } from 'vitest'
import { adjacencyMatrix, degreeMatrix, degrees, incidenceMatrix, laplacian } from 'aifn-compute/graph/matrices'
import { fromEdges } from 'aifn-compute/graph'
import { cycleGraph } from 'aifn-compute/graph/structures'
import { eigh } from 'aifn-compute/numerics/linalg'
import { matmul, sub, sum, toFlat, toRows, transpose } from 'aifn-compute/foundation/tensor'

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
