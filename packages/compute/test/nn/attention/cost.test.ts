/**
 * aifn-compute/nn/attention `attentionCost` and `layerLinearCost`: attention over all heads equals the projections and
 * feed-forward network exactly at $n = 6 d_{\text{model}}$ (the efficient-attention note's crossover), and the
 * per-method formulas at small sizes.
 */
import { describe, expect, it } from 'vitest'
import { attentionCost, layerLinearCost } from 'aifn-compute/nn/attention'

describe('attention cost', () => {
  it('crosses over at n = 6 d_model (d_model 1024, head width 64)', () => {
    const dModel = 1024
    const n = 6 * dModel
    expect((dModel / 64) * attentionCost('full', n, { headWidth: 64 }).multiplyAdds).toBe(layerLinearCost(n, dModel))
  })

  it('counts each method by its formula', () => {
    const o = { headWidth: 2, window: 4, blockSize: 4, topK: 1, features: 3, rank: 5, buckets: 4, rounds: 2 }
    const n = 16
    expect(attentionCost('full', n, o)).toEqual({ multiplyAdds: 2 * 256 * 2, scoreMemory: 256 })
    expect(attentionCost('flash', n, o)).toEqual({ multiplyAdds: 2 * 256 * 2, scoreMemory: 32 })
    expect(attentionCost('window', n, o)).toEqual({ multiplyAdds: 2 * 16 * 4 * 2, scoreMemory: 64 })
    expect(attentionCost('block', n, o)).toEqual({
      multiplyAdds: 32 + 16 * 4 * 2 + 2 * 16 * 8 * 2,
      scoreMemory: 64 + 128,
    })
    expect(attentionCost('linear', n, o)).toEqual({ multiplyAdds: 2 * 16 * 3 * 2 + 2 * 16 * 3, scoreMemory: 9 })
    expect(attentionCost('linformer', n, o)).toEqual({ multiplyAdds: 4 * 16 * 5 * 2, scoreMemory: 80 })
    expect(attentionCost('lsh', n, o)).toEqual({ multiplyAdds: 2 * (16 * 2 * 2 + 2 * 16 * 8 * 2), scoreMemory: 256 })
  })
})
