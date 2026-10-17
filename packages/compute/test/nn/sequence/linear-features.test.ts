/**
 * aifn-compute/nn/sequence `positiveRandomFeatures` and `linearAttentionWeights` (the linear-attention note's checks):
 * 200,000 features estimate $\exp(\qvec^\top\kvec) = 0.9704$ as 0.9692, and the weights times $\Vmat$ equal
 * `linearAttention` with the same feature map, causal and decayed.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { matmul, mul, sum, tensor, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import {
  eluFeatureMap,
  linearAttention,
  linearAttentionWeights,
  positiveRandomFeatures,
} from 'aifn-compute/nn/sequence'

const rows = (v: Value) => toRows(v as Tensor)
const maxDiff = (a: number[][], b: number[][]) =>
  Math.max(...a.flatMap((r, i) => r.map((v, j) => Math.abs(v - b[i][j]))))

describe('linear attention features and weights', () => {
  it('estimates the softmax kernel without bias: 0.9692 for exp(qTk) = 0.9704', () => {
    const q = tensor([[0.3, -0.2, 0.5, 0.1]])
    const k = tensor([[0.4, 0.1, -0.3, 0.2]])
    const { featureMap, omega } = positiveRandomFeatures(stream(3), 4, 200000)
    expect(omega.shape).toEqual([200000, 4])
    const est = sum(mul(featureMap(q), featureMap(k))) as number
    const exact = Math.exp(0.3 * 0.4 - 0.2 * 0.1 - 0.5 * 0.3 + 0.1 * 0.2)
    expect(exact).toBeCloseTo(0.9704, 4)
    expect(Math.abs(est / exact - 1)).toBeLessThan(0.01)
  })

  it('gives weights whose product with V is linearAttention, causal or decayed', () => {
    const s = stream(4)
    const Q = normals(s, [6, 4], 0, 0.5)
    const K = normals(s, [6, 4], 0, 0.5)
    const V = normals(s, [6, 2])
    const pf = positiveRandomFeatures(stream(5), 4, 32).featureMap
    const W = linearAttentionWeights(Q, K, pf)
    rows(W).forEach((row, i) => {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      row.forEach((w, j) => expect(j > i ? w : 0).toBe(0))
    })
    const ref = rows(linearAttention(Q, K, V, { featureMap: pf }))
    expect(maxDiff(rows(matmul(W, V)), ref)).toBeLessThan(1e-12)
    const decayed = linearAttentionWeights(Q, K, eluFeatureMap, true, 0.5)
    expect(maxDiff(rows(matmul(decayed, V)), rows(linearAttention(Q, K, V, { decay: 0.5 })))).toBeLessThan(1e-12)
    const full = rows(linearAttentionWeights(Q, K, pf, false))
    expect(full.every((row) => row.every((w) => w > 0))).toBe(true)
  })
})
