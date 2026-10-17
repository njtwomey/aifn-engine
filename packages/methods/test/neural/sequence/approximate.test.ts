/**
 * aifn-methods/neural/sequence low-rank and hashing attention (the low-rank note's checks): each approximation is exact
 * at its limit (full-rank SVD, Linformer with one segment per token, Nyström with a landmark per token, LSH with many
 * rounds of two buckets), and a constant attention matrix has singular values (1, 0).
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { scaledDotProductAttention } from 'aifn-compute/nn/attention'
import {
  angularBuckets,
  angularRotations,
  linformerMatrix,
  lowRankAt,
  lowRankSeries,
  lshAttentionMatrix,
  nystromMatrix,
  rowTotalVariation,
  segmentMeans,
  topicTokens,
} from 'aifn-methods/neural/sequence'

describe('low-rank and hashing attention', () => {
  const n = 32
  const { q, k, topic } = topicTokens(stream(6), n, 8, 4, 0.6)
  const P = toRows(scaledDotProductAttention(tensor(q), tensor(k), tensor(k), { scale: 6 }).weights as Tensor)

  it('draws unit-length tokens in runs of topics', () => {
    expect(topic.slice(0, 9)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 1])
    q.forEach((row) => expect(Math.hypot(...row)).toBeCloseTo(1, 12))
  })

  it('is exact at the limit of each method', () => {
    const series = lowRankSeries(P)
    expect(rowTotalVariation(lowRankAt(series, n), P)).toBeLessThan(1e-10)
    expect(rowTotalVariation(linformerMatrix(q, k, 6, segmentMeans(n, n)), P)).toBeLessThan(1e-12)
    expect(rowTotalVariation(nystromMatrix(q, k, 6, n, 1e-12), P)).toBeLessThan(1e-8)
    const lsh = lshAttentionMatrix(q, k, 6, angularRotations(stream(7), 8, 2, 40))
    expect(lsh.keysPerQuery).toBe(n)
    expect(rowTotalVariation(lsh.weights, P)).toBeLessThan(1e-12)
    expect(rowTotalVariation(lowRankAt(series, 2), P)).toBeGreaterThan(rowTotalVariation(lowRankAt(series, 8), P))
  })

  it('finds rank one in a constant attention matrix, and averages segments', () => {
    const t = lowRankSeries([
      [0.5, 0.5],
      [0.5, 0.5],
    ]).singularValues
    expect(t[0]).toBeCloseTo(1, 12)
    expect(Math.abs(t[1])).toBeLessThan(1e-12)
    expect(segmentMeans(4, 2)).toEqual([
      [0.5, 0.5, 0, 0],
      [0, 0, 0.5, 0.5],
    ])
  })

  it('hashes opposite vectors to opposite buckets', () => {
    const R = angularRotations(stream(7), 2, 4, 3)
    const opposite = [
      [1, 0],
      [-1, 0],
    ]
    angularBuckets(opposite, R).forEach(([a, b]) => expect(Math.abs(a - b)).toBe(2))
  })
})
