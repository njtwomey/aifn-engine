/**
 * aifn-compute/nn/attention block selection (the trainable-sparse-attention note's checks): future blocks score
 * $-\infty$, keeping every visible block captures all the weight, `topK` 0 keeps the own block, and the worked example
 * (16 keys in blocks of 4, $\qvec = (2, 1)$, scale 1) scores (2, 1, -2, -1) and keeps 87.63% then 99.26% of the weight.
 */
import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  blockScores,
  blockSparseMask,
  capturedWeight,
  causalMask,
  scaledDotProductAttention,
  selectBlocks,
} from 'aifn-compute/nn/attention'

describe('block selection', () => {
  const s = stream(2)
  const q = normals(s, [8, 4])
  const k = normals(s, [8, 4])

  it('hides future blocks and captures all weight when every visible block is kept', () => {
    const sc = blockScores(q, k, 2)
    expect(sc.shape).toEqual([8, 4])
    const rows = toRows(sc)
    expect(rows[1][1]).toBe(Number.NEGATIVE_INFINITY)
    expect(Number.isFinite(rows[2][1])).toBe(true)
    const sel = selectBlocks(sc, 2, { topK: 4 })
    expect(sel[7]).toEqual([0, 1, 2, 3])
    const W = scaledDotProductAttention(q, k, k, { mask: causalMask(8), scale: 0.5 }).weights as Tensor
    const cap = toFlat(capturedWeight(W, blockSparseMask(8, 2, sel)))
    cap.forEach((c) => expect(Math.abs(c - 1)).toBeLessThan(1e-12))
    expect(selectBlocks(sc, 2, { topK: 0 })[5]).toEqual([2])
  })

  it('reproduces the worked example: scores (2, 1, -2, -1), top-1 keeps {0, 3} and 87.63%, top-2 99.26%', () => {
    const kk = [
      [2, 0],
      [0, 0],
      [1, 1],
      [1, -1],
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 1],
      [-1, 0],
      [-1, 1],
      [-1, -1],
      [-1, 0],
      [0, -1],
      [1, -1],
      [-1, -1],
      [0, -1],
    ]
    const qq = kk.map(() => [2, 1])
    const ws = blockScores(qq, kk, 4, { scale: 1 })
    expect(toRows(ws)[15]).toEqual([2, 1, -2, -1])
    const e = kk.map((x) => Math.exp(2 * x[0] + x[1]))
    const Z = e.reduce((a, b) => a + b, 0)
    const row = [e.map((x) => x / Z)]
    const top1 = selectBlocks(ws, 4, { topK: 1 })
    expect(top1[15]).toEqual([0, 3])
    const c1 = toFlat(capturedWeight(row, [toRows(blockSparseMask(16, 4, top1))[15]]))[0]
    const c2 = toFlat(capturedWeight(row, [toRows(blockSparseMask(16, 4, selectBlocks(ws, 4, { topK: 2 })))[15]]))[0]
    expect(c1).toBeCloseTo(0.8763, 4)
    expect(c2).toBeCloseTo(0.9926, 4)
  })

  it('normalises scores into probabilities over the visible blocks', () => {
    const p = toRows(blockScores(q, k, 2, { normalise: true }))
    p.forEach((row) => expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12))
    expect(p[1][1]).toBe(0)
    expect(toFlat(blockScores(tensor([[1, 0]]), tensor([[1, 0]]), 1))[0]).toBeCloseTo(Math.SQRT1_2, 14)
  })
})
