/** Chebyshev graph filters and ChebNet (aifn-compute/nn/graph), against the site's chebyshev-graph-filter check. */
import { describe, expect, it } from 'vitest'
import { ChebConv, chebConv, chebyshevFilter, chebyshevTerms } from 'aifn-compute/nn/graph'
import { laplacian, rescaledLaplacian } from 'aifn-compute/graph/matrices'
import { unweightedShortestPaths } from 'aifn-compute/graph/traversal'
import { chebyshevSeries } from 'aifn-compute/numerics/interpolate'
import { eigh } from 'aifn-compute/numerics/linalg'
import { fromEdges } from 'aifn-compute/graph'
import { stream } from 'aifn-compute/foundation/random'
import { diag, matmul, shapeOf, tensor, toFlat, transpose, type Tensor } from 'aifn-compute/foundation/tensor'

/** The note's 6-node graph: a triangle 0-1-2, then 2-3, 3-4 and 3-5. */
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
const flat = (t: unknown) => Array.from(toFlat(t as Tensor))

describe('chebyshevFilter', () => {
  it('the recurrence equals U g(Lambda) U^T x to 1e-8 for theta = (0.4, -0.3, 0.2, 0.1)', () => {
    const { lTilde, lambdaMax } = rescaledLaplacian(g6)
    const theta = [0.4, -0.3, 0.2, 0.1]
    const x = tensor([1, -2, 0.5, 0, 3, 1])
    const y = flat(chebyshevFilter(lTilde, x, theta))
    const { values, vectors } = eigh(laplacian(g6, { weighted: false, normalisation: 'symmetric' }))
    const gains = flat(values).map((l) => chebyshevSeries(theta, l, [0, lambdaMax]))
    const exact = flat(matmul(matmul(matmul(vectors, diag(tensor(gains))), transpose(vectors)), x))
    y.forEach((v, i) => expect(v).toBeCloseTo(exact[i], 8))
  })
  it('a degree-K response is exactly zero beyond K hops', () => {
    const { lTilde } = rescaledLaplacian(g6)
    const hops = flat(unweightedShortestPaths(g6, 0).distance)
    for (let K = 0; K <= 3; K++) {
      const y = flat(chebyshevFilter(lTilde, tensor([1, 0, 0, 0, 0, 0]), Array(K + 1).fill(1)))
      y.forEach((v, i) => {
        if (hops[i] > K) expect(v === 0).toBe(true)
      })
    }
  })
})

describe('chebyshevTerms', () => {
  it('worked example, unnormalised with lambda max 6: T1 x and T2 x', () => {
    const { lTilde, lambdaMax } = rescaledLaplacian(g6, { normalisation: 'unnormalised' })
    expect(lambdaMax).toBe(6)
    const terms = chebyshevTerms(lTilde, tensor([1, 0, 0, 0, 0, 0]), 2).map(flat)
    ;[-1 / 3, -1 / 3, -1 / 3, 0, 0, 0].forEach((v, i) => expect(terms[1][i]).toBeCloseTo(v, 12))
    ;[-1 / 3, 2 / 3, 4 / 9, 2 / 9, 0, 0].forEach((v, i) => expect(terms[2][i]).toBeCloseTo(v, 12))
  })
  it('throws for a signal without one row per node', () => {
    expect(() => chebyshevTerms(rescaledLaplacian(g6).lTilde, tensor([1, 2]), 1)).toThrow()
  })
})

describe('chebConv and ChebConv', () => {
  it('one weight per order on one channel equals chebyshevFilter', () => {
    const { lTilde } = rescaledLaplacian(g6)
    const h = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 0],
      [0, 2],
      [1, 0],
    ])
    const w = [tensor([[0.5], [0]]), tensor([[-1], [0]]), tensor([[0.25], [0]])]
    const out = flat(chebConv(lTilde, h, w))
    const ref = flat(chebyshevFilter(lTilde, tensor([1, 0, 1, 2, 0, 1]), [0.5, -1, 0.25]))
    out.forEach((v, i) => expect(v).toBeCloseTo(ref[i], 12))
  })
  it('the layer: K + 1 weights, output V x G', () => {
    const layer = ChebConv(g6, 2, 3, 2)
    const p = layer.init(stream(0))
    expect(p.weights.length).toBe(3)
    const h = tensor([[1, 0], ...Array.from({ length: 5 }, () => [0, 1])])
    expect(shapeOf(layer.apply(p, h) as Tensor)).toEqual([6, 3])
  })
})
