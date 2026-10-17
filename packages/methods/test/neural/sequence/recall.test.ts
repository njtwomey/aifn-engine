/**
 * aifn-methods/neural/sequence recall: MQAR's invariants (distinct keys, queries a permutation, answers follow keys),
 * the long-range note's worked example (8 pairs, Hadamard keys of width 4: attention reads 4.05, 3.01, 5.98, 1.06,
 * 2.04, 6.97, 4.98, 8.91 at scale 10) and the probe's checks at 16 pairs.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { scaledDotProductAttention } from 'aifn-compute/nn/attention'
import { associativeRecallByDimension, multiQueryAssociativeRecall } from 'aifn-methods/neural/sequence'

describe('multi-query associative recall', () => {
  it('draws distinct keys, queries them all once, and answers with their values', () => {
    const m = multiQueryAssociativeRecall(stream(1), { pairs: 8, keyVocab: 26, valueVocab: 10 })
    expect(new Set(m.keys).size).toBe(8)
    expect([...m.queries].sort((a, b) => a - b)).toEqual([...m.keys].sort((a, b) => a - b))
    m.queries.forEach((q, i) => expect(m.values[m.keys.indexOf(q)]).toBe(m.answers[i]))
    expect(() => multiQueryAssociativeRecall(stream(1), { pairs: 8, keyVocab: 4, valueVocab: 10 })).toThrow(/keyVocab/)
  })

  it('reproduces the worked example: attention reads within 0.1 of every value at scale 10', () => {
    const h = 0.5
    const K = tensor([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [h, h, h, h],
      [h, -h, h, -h],
      [h, h, -h, -h],
      [h, -h, -h, h],
    ])
    const V = tensor([[4], [3], [6], [1], [2], [7], [5], [9]])
    const reads = toFlat(scaledDotProductAttention(K, K, V, { scale: 10 }).output as Tensor)
    const want = [4.05, 3.01, 5.98, 1.06, 2.04, 6.97, 4.98, 8.91]
    expect(reads.map((r) => +r.toFixed(2))).toEqual(want)
  })
})

describe('recall by state size', () => {
  it('attention recalls from d = 16; linear and delta memories need d well above N = 16', () => {
    const r = associativeRecallByDimension(stream(7), {
      pairs: 16,
      dims: [4, 8, 16, 32, 64, 128],
      trials: 12,
      valueVocab: 32,
      scale: 30,
      beta: 1,
    })
    expect(r.dims).toEqual([4, 8, 16, 32, 64, 128])
    expect(r.attention.slice(2).every((v) => v >= 0.95)).toBe(true)
    expect(r.delta[5]).toBeGreaterThanOrEqual(0.95)
    expect(r.linear[5]).toBeGreaterThanOrEqual(0.9)
    expect(r.delta[0]).toBeLessThan(0.6)
    expect(r.delta[2]).toBeLessThan(0.9)
    expect(r.linear[2]).toBeLessThan(0.9)
    const half = associativeRecallByDimension(stream(7), {
      pairs: 16,
      dims: [16],
      trials: 12,
      valueVocab: 32,
      scale: 30,
      beta: 0.5,
    })
    expect(half.delta[0]).toBeGreaterThan(r.delta[2])
  })
})
