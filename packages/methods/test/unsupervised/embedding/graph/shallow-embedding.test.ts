/** aifn-methods/unsupervised/embedding/graph: dot-product embeddings and the similarities they reconstruct. */
import { describe, expect, it } from 'vitest'
import { matmul, tensor, toFlat, transpose } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import { eigh } from 'aifn-compute/numerics/linalg'
import {
  dotProductEmbedding,
  graphEmbeddingFunctions,
  netmfMatrix,
  nodeSimilarity,
  reconstructionError,
} from 'aifn-methods/unsupervised/embedding/graph'

const flat = (t: Parameters<typeof toFlat>[0]) => Array.from(toFlat(t))
const path = fromEdges(
  4,
  [
    [0, 1],
    [1, 2],
    [2, 3],
  ],
  { directed: false },
)

describe('nodeSimilarity', () => {
  it('gives A, A² (degrees on the diagonal) and the NetMF matrix', () => {
    expect(flat(nodeSimilarity(path, 'adjacency'))).toEqual([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    expect(flat(nodeSimilarity(path, 'two-hop'))).toEqual([1, 0, 1, 0, 0, 2, 0, 1, 1, 0, 2, 0, 0, 1, 0, 1])
    expect(flat(nodeSimilarity(path, 'walk', 4))).toEqual(flat(netmfMatrix(path, { window: 4 })))
  })
})

describe('dotProductEmbedding', () => {
  it('reconstructs a PSD matrix of rank d exactly', () => {
    const z0 = tensor([
      [1, 0],
      [1, 1],
      [0, 2],
      [-1, 1],
    ])
    const S = matmul(z0, transpose(z0))
    const fit = dotProductEmbedding(S, 2)
    expect(fit.relativeError).toBeLessThan(1e-20)
    expect(fit.z.shape).toEqual([4, 2])
    expect(fit.eigenvalues[2]).toBeCloseTo(0, 12)
  })

  it('leaves the error of the Eckart–Young theorem: the squared dropped eigenvalues over the squared norm', () => {
    const z0 = tensor([
      [2, 0, 0.5],
      [1, 1, 0],
      [0, 2, 1],
      [1, 0, 1],
    ])
    const S = matmul(z0, transpose(z0))
    const lambda = flat(eigh(S).values)
    const norm = lambda.reduce((a, v) => a + v * v, 0)
    for (const d of [1, 2, 3]) {
      const want = lambda.slice(d).reduce((a, v) => a + v * v, 0) / norm
      expect(dotProductEmbedding(S, d).relativeError).toBeCloseTo(want, 12)
    }
  })

  it('leaves the columns of non-positive eigenvalues at zero', () => {
    // The adjacency of a path has eigenvalues of both signs.
    const fit = dotProductEmbedding(nodeSimilarity(path, 'adjacency'), 4)
    const z = flat(fit.z)
    fit.eigenvalues.forEach((l, k) => {
      if (l <= 0) for (let i = 0; i < 4; i++) expect(z[i * 4 + k]).toBe(0)
    })
    expect(() => dotProductEmbedding(nodeSimilarity(path, 'adjacency'), 5)).toThrow(/dim/)
  })
})

describe('reconstructionError', () => {
  it('is 0 for the exact rows, 0.1 for the top eigenvector of [[2, 1], [1, 2]] alone, and 0 for a zero S', () => {
    const S = [
      [2, 1],
      [1, 2],
    ]
    const a = Math.sqrt(1.5)
    const b = Math.sqrt(0.5)
    expect(
      reconstructionError(S, [
        [a, b],
        [a, -b],
      ]),
    ).toBeCloseTo(0, 14)
    expect(reconstructionError(S, [[a], [a]])).toBeCloseTo(0.1, 14)
    expect(
      reconstructionError(
        [
          [0, 0],
          [0, 0],
        ],
        [[1], [1]],
      ),
    ).toBe(0)
  })

  it('every function is registered', () => {
    for (const key of ['netmfMatrix', 'nodeSimilarity', 'dotProductEmbedding', 'reconstructionError'])
      expect(graphEmbeddingFunctions[key].info.key).toBe(key)
  })
})
