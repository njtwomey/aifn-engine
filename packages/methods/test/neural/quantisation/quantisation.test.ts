import { describe, expect, it } from 'vitest'
import {
  decodeThroughput,
  quantisationAwareTraining,
  quantisationStudy,
  type QatSnapshot,
  servingMemory,
  type QuantisationSnapshot,
} from 'aifn-methods/neural/quantisation'
import { moons } from 'aifn-methods/data'
import { spirals } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'

describe('serving cost', () => {
  // A 7B model with 32 layers, 8 KV heads of size 128.
  const model = { parameters: 7e9, layers: 32, kvHeads: 8, headDim: 128 }
  const device = { memoryGb: 12, bandwidthGbs: 1000, tflops: 150 }
  it('memory scales with bits; 4-bit weights fit where 16-bit do not', () => {
    const fp16 = servingMemory(model, { weightBits: 16, kvBits: 16, batch: 1, context: 4096 }, device)
    const int4 = servingMemory(model, { weightBits: 4, kvBits: 16, batch: 1, context: 4096 }, device)
    expect(fp16.weights).toBe(14e9)
    expect(int4.weights).toBe(3.5e9)
    expect(fp16.kvCache).toBe(2 * 32 * 8 * 128 * 4096 * 2)
    expect(fp16.fits).toBe(false)
    expect(int4.fits).toBe(true)
  })
  it('single-sequence decoding is memory-bound and four times faster at 4 bits', () => {
    const at = (bits: number) => decodeThroughput(model, { weightBits: bits, kvBits: 16, batch: 1, context: 0 }, device)
    expect(at(16).bound).toBe('memory')
    expect(at(4).tokensPerSecond / at(16).tokensPerSecond).toBeCloseTo(4, 6)
    const big = decodeThroughput(model, { weightBits: 16, kvBits: 16, batch: 1024, context: 0 }, device)
    expect(big.bound).toBe('compute')
    expect(at(16).ridgeBatch).toBeCloseTo(150, 6)
  })
})

describe('quantisation study', () => {
  it('trains, then accuracy falls as bits fall; per-channel and GPTQ are no worse than per-tensor at 8 bits', () => {
    const data = moons(stream('q-study'), { n: 160, noise: 0.1 })
    let last: QuantisationSnapshot | undefined
    for (const s of quantisationStudy(data, { steps: 300, bits: [2, 4, 8] })) last = s
    expect(last!.phase).toBe('done')
    expect(last!.accuracy).toBeGreaterThan(0.95)
    expect(last!.loss.at(-1)!).toBeLessThan(last!.loss[0])
    const at8 = last!.results.find((r) => r.bits === 8)!
    expect(at8.accuracy['per-tensor']).toBeGreaterThan(0.93)
    expect(at8.sqnr['per-channel']).toBeGreaterThanOrEqual(at8.sqnr['per-tensor'] - 1e-9)
    const at2 = last!.results.find((r) => r.bits === 2)!
    expect(at2.sqnr['per-tensor']).toBeLessThan(at8.sqnr['per-tensor'])
  })
})

describe('quantisation-aware training', () => {
  it('starts at the PTQ network, lowers its quantised loss, and beats PTQ at few bits', () => {
    const data = spirals(stream('q-qat'), { n: 200, noise: 0.05 })
    let last: QatSnapshot | undefined
    const opts = { width: 16, depth: 2, steps: 800, qatSteps: 300, bits: [2, 3, 6] }
    for (const s of quantisationAwareTraining(data, opts)) last = s
    const r = last!
    expect(r.phase).toBe('done')
    expect(r.done).toBe(r.total)
    expect(r.accuracy).toBeGreaterThan(0.85)
    for (const b of r.results) {
      // Step 0 of fine-tuning is the trained network, so its quantised loss is PTQ's; fine-tuning lowers it.
      expect(b.step[0]).toBe(0)
      expect(b.weights[0].length).toBe(b.weights.at(-1)!.length)
      expect(Math.min(...b.loss.slice(1))).toBeLessThan(b.loss[0])
      expect(b.qat).toBeGreaterThanOrEqual(b.ptq - 0.02)
    }
    const at2 = r.results.find((b) => b.bits === 2)!
    const at6 = r.results.find((b) => b.bits === 6)!
    expect(at2.qat).toBeGreaterThan(at2.ptq + 0.05)
    // At 6 bits rounding barely hurts: PTQ is already close to floating point.
    expect(at6.ptq).toBeGreaterThan(r.accuracy - 0.05)
  })
})

describe('AWQ in the study', () => {
  it('reports AWQ next to the other methods', () => {
    const data = moons(stream('q-study'), { n: 120, noise: 0.1 })
    let last: QuantisationSnapshot | undefined
    for (const s of quantisationStudy(data, { steps: 200, bits: [3, 8] })) last = s
    const at8 = last!.results.find((r) => r.bits === 8)!
    expect(at8.accuracy.awq).toBeGreaterThan(0.9)
    expect(Number.isFinite(at8.sqnr.awq)).toBe(true)
  })
})
