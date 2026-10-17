/**
 * aifn-compute/nn/attention sparse patterns (the sliding-window note's checks): a half-width-2 window over 16 tokens
 * scores 74 pairs and equals `positionMask` with window 3, a causal window reaches positions 4 to 10 from position 10
 * after three layers, one global token gives full reach in two layers, and causal BigBird sees no future key.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import {
  attentionReach,
  maskDensity,
  positionMask,
  positionRange,
  sparseAttentionMask,
} from 'aifn-compute/nn/attention'

describe('sparse attention masks', () => {
  it('scores 16 x 5 - 6 = 74 pairs with a window of half-width 2, the band of positionMask window 3', () => {
    const m = sparseAttentionMask(16, 'sliding', { window: 2 })
    expect(maskDensity(m).pairs).toBe(74)
    expect(maskDensity(toRows(m)).keysPerQuery).toBe(74 / 16)
    const p = positionRange(16)
    expect(toFlat(m)).toEqual(toFlat(positionMask(p, p, { window: 3 })))
  })

  it('reaches positions 4 to 10 from position 10 after three causal layers of the current token and two before', () => {
    const c = sparseAttentionMask(16, 'sliding', { window: 2, causal: true })
    const row = toRows(attentionReach(c, 3))[10]
    expect(row.flatMap((v, j) => (v ? [j] : []))).toEqual([4, 5, 6, 7, 8, 9, 10])
  })

  it('gives full reach in two layers with one global token, and causal BigBird has no future key', () => {
    const g = sparseAttentionMask(32, 'global', { window: 1, globals: 1 })
    expect(toFlat(attentionReach(g, 2)).every((v) => v === 1)).toBe(true)
    const b = toRows(
      sparseAttentionMask(32, 'bigbird', { window: 1, globals: 1, random: 2, stream: stream(1), causal: true }),
    )
    expect(b.every((row, i) => row.every((v, j) => j <= i || v === 0))).toBe(true)
    expect(b.every((row, i) => row[i] === 1)).toBe(true)
    expect(() => sparseAttentionMask(8, 'bigbird', { random: 1 })).toThrow(/stream/)
  })

  it('builds the dilated and strided patterns by their offsets', () => {
    const dilated = toRows(sparseAttentionMask(9, 'dilated', { window: 2, dilation: 2 }))[4]
    expect(dilated).toEqual([1, 0, 1, 0, 1, 0, 1, 0, 1])
    const strided = toRows(sparseAttentionMask(9, 'strided', { stride: 3, causal: true }))[8]
    expect(strided).toEqual([0, 0, 1, 0, 0, 1, 1, 1, 1])
  })
})
