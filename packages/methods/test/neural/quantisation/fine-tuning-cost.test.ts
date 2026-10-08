/**
 * The fine-tuning budget against the worked numbers of the AI Field Notes on QLoRA ("Worked example: memory for 8B and
 * 70B models") and on bootstrapping a language model for a new task (step 3), and the orderings the formulas imply.
 */
import { describe, expect, it } from 'vitest'
import {
  adapterParameters,
  fineTuningCompute,
  fineTuningMemory,
  linearParameters,
  LLAMA_3_1_70B,
  LLAMA_3_1_8B,
  quantisationFunctions,
  type FineTuningSetup,
  type TransformerTrainShape,
} from 'aifn-methods/neural/quantisation'

const GB = 1e9
/** Model state only: one token, no activations or logits to speak of. */
const state = (method: FineTuningSetup['method']): FineTuningSetup => ({
  method,
  microBatch: 1,
  sequence: 1,
  checkpointing: true,
  logits: 'chunked',
})
const modelState = (shape: TransformerTrainShape, method: FineTuningSetup['method']) => {
  const m = fineTuningMemory(shape, state(method))
  return m.weights + m.adapterState + m.optimiserState + m.gradients
}

describe('parameter counts', () => {
  it('reproduces the Llama 3.1 counts of the notes', () => {
    expect(linearParameters(LLAMA_3_1_8B) / 1e9).toBeCloseTo(6.98, 2)
    expect((LLAMA_3_1_8B.parameters - linearParameters(LLAMA_3_1_8B)) / 1e9).toBeCloseTo(1.05, 2)
    expect(linearParameters(LLAMA_3_1_70B) / 1e9).toBeCloseTo(68.45, 2)
    expect((LLAMA_3_1_70B.parameters - linearParameters(LLAMA_3_1_70B)) / 1e9).toBeCloseTo(2.1, 2)
    // Rank 16 on all seven layers: 1,310,720 per block of the 8B model, 41.9 million and 207 million in all.
    expect(adapterParameters(LLAMA_3_1_8B, 16) / 32).toBe(1_310_720)
    expect(adapterParameters(LLAMA_3_1_8B, 16) / 1e6).toBeCloseTo(41.94, 2)
    expect(adapterParameters(LLAMA_3_1_70B, 16) / 1e6).toBeCloseTo(207.1, 1)
  })

  it('counts adapters by hand for a small grouped-query shape', () => {
    // h = 8, 4 query heads and 2 key-value heads of size 2, f = 12: q 8→8, k and v 8→4, o 8→8, gate and up 8→12,
    // down 12→8.
    const tiny: TransformerTrainShape = {
      parameters: 0,
      layers: 3,
      width: 8,
      ffnWidth: 12,
      vocab: 10,
      queryHeads: 4,
      kvHeads: 2,
      headDim: 2,
      tiedEmbeddings: true,
    }
    expect(adapterParameters(tiny, 2, ['q'])).toBe(3 * 2 * (8 + 8))
    expect(adapterParameters(tiny, 2, ['k', 'v'])).toBe(3 * 2 * 2 * (8 + 4))
    expect(adapterParameters(tiny, 2)).toBe(3 * 2 * (16 + 12 + 12 + 16 + 20 + 20 + 20))
    expect(linearParameters(tiny)).toBe(3 * (64 + 32 + 32 + 64 + 96 + 96 + 96))
  })
})

describe('fineTuningMemory', () => {
  it("reproduces the QLoRA note's model state for 8B and 70B", () => {
    expect(modelState(LLAMA_3_1_8B, 'full') / GB).toBeCloseTo(128.5, 1)
    expect(modelState(LLAMA_3_1_70B, 'full') / GB).toBeCloseTo(1128.8, 1)
    expect(modelState(LLAMA_3_1_8B, 'lora') / GB).toBeCloseTo(16.7, 1)
    expect(modelState(LLAMA_3_1_70B, 'lora') / GB).toBeCloseTo(144.4, 1)
    expect(modelState(LLAMA_3_1_8B, 'qlora') / GB).toBeCloseTo(6.4, 1)
    expect(modelState(LLAMA_3_1_70B, 'qlora') / GB).toBeCloseTo(42.8, 1)
  })

  it("reproduces the bootstrapping note's LoRA run: 20.3 GB, 24.5 GB with the logits", () => {
    const setup: FineTuningSetup = {
      method: 'lora',
      microBatch: 8,
      sequence: 1024,
      checkpointing: true,
      logits: 'chunked',
    }
    const m = fineTuningMemory(LLAMA_3_1_8B, setup)
    expect(m.weights / GB).toBeCloseTo(16.06, 2)
    expect((m.adapterState + m.optimiserState + m.gradients) / GB).toBeCloseTo(0.67, 2)
    // Layer inputs 2.15 GB and the recomputed layer 1.44 GB.
    expect(m.activations / GB).toBeCloseTo(2.15 + 1.44, 2)
    expect(m.total / GB).toBeCloseTo(20.3, 1)
    const full = fineTuningMemory(LLAMA_3_1_8B, { ...setup, logits: 'full' })
    expect(full.logits / GB).toBeCloseTo(4.2, 1)
    expect(full.total / GB).toBeCloseTo(24.5, 1)
  })

  it("reproduces the QLoRA note's activations: 9.1 GB stored with fused attention, 0.54 GB of checkpoints", () => {
    const run = { method: 'lora' as const, microBatch: 1, sequence: 2048, logits: 'full' as const }
    expect(fineTuningMemory(LLAMA_3_1_8B, { ...run, checkpointing: false }).activations / GB).toBeCloseTo(9.1, 1)
    const stored = fineTuningMemory(LLAMA_3_1_8B, { ...run, checkpointing: false, attention: 'stored' })
    expect(stored.activations).toBeCloseTo(2048 * 4096 * (34 + (5 * 32 * 2048) / 4096) * 32, 0)
    const inputs = 2 * 2048 * 4096 * 32
    expect(inputs / GB).toBeCloseTo(0.54, 2)
    expect(fineTuningMemory(LLAMA_3_1_8B, { ...run, checkpointing: true }).activations).toBeGreaterThan(inputs)
    // Logits of one 2048-token sequence in 32 bits: 1.05 GB.
    expect(fineTuningMemory(LLAMA_3_1_8B, { ...run, checkpointing: true }).logits / GB).toBeCloseTo(1.05, 2)
  })

  it('orders the methods, the optimisers and checkpointing as the formulas say', () => {
    const run = { microBatch: 4, sequence: 2048, checkpointing: true, logits: 'chunked' as const }
    const full = fineTuningMemory(LLAMA_3_1_8B, { method: 'full', ...run }).total
    const lora = fineTuningMemory(LLAMA_3_1_8B, { method: 'lora', ...run }).total
    const qlora = fineTuningMemory(LLAMA_3_1_8B, { method: 'qlora', ...run }).total
    expect(lora).toBeLessThan(full)
    expect(qlora).toBeLessThan(lora)
    // QLoRA without double quantisation: 0.373 bits more per linear weight.
    const single = fineTuningMemory(LLAMA_3_1_8B, { method: 'qlora', ...run, doubleQuant: false }).weights
    const double = fineTuningMemory(LLAMA_3_1_8B, { method: 'qlora', ...run }).weights
    expect(((single - double) * 8) / linearParameters(LLAMA_3_1_8B)).toBeCloseTo(0.373, 3)
    // 8-bit Adam: 10 bytes per parameter for full fine-tuning instead of 16.
    expect(modelState(LLAMA_3_1_8B, 'full') / LLAMA_3_1_8B.parameters).toBe(16)
    const adam8 = fineTuningMemory(LLAMA_3_1_8B, { ...state('full'), optimiser: 'adam8' })
    expect((adam8.weights + adam8.optimiserState + adam8.gradients) / LLAMA_3_1_8B.parameters).toBe(10)
    const stored = fineTuningMemory(LLAMA_3_1_8B, { method: 'lora', ...run, checkpointing: false }).activations
    const checkpointed = fineTuningMemory(LLAMA_3_1_8B, { method: 'lora', ...run }).activations
    expect(checkpointed).toBeLessThan(stored)
  })

  it('says whether a run fits, paging the optimiser state when asked', () => {
    const run: FineTuningSetup = {
      method: 'qlora',
      microBatch: 8,
      sequence: 1024,
      checkpointing: true,
      logits: 'chunked',
    }
    expect(fineTuningMemory(LLAMA_3_1_8B, run, { memoryGb: 16 }).fits).toBe(true)
    expect(fineTuningMemory(LLAMA_3_1_8B, { ...run, method: 'lora' }, { memoryGb: 16 }).fits).toBe(false)
    expect(fineTuningMemory(LLAMA_3_1_8B, { ...run, method: 'lora' }, { memoryGb: 24 }).fits).toBe(true)
    expect(fineTuningMemory(LLAMA_3_1_70B, run, { memoryGb: 80 }).fits).toBe(true)
    expect(fineTuningMemory(LLAMA_3_1_8B, { ...state('full') }, { memoryGb: 80 }).fits).toBe(false)
    // Paged: the moments and master copy may spill to the host; 4 bytes per parameter stay resident.
    expect(fineTuningMemory(LLAMA_3_1_8B, { ...state('full'), optimiser: 'paged' }, { memoryGb: 80 }).fits).toBe(true)
    expect(fineTuningMemory(LLAMA_3_1_8B, run).fits).toBe(true)
  })
})

describe('fineTuningCompute', () => {
  const run = { tokens: 4000 * 630, epochs: 3, mfu: 0.3, peakFlops: 990e12, pricePerHour: 3 }

  it("reproduces the bootstrapping note's run: 3.6e17 FLOPs, about 20 minutes, about $1", () => {
    const c = fineTuningCompute(LLAMA_3_1_8B, { method: 'lora', checkpointing: true }, run)
    expect(c.flops / 1e17).toBeCloseTo(3.6, 1)
    expect(c.seconds).toBeGreaterThan(1150)
    expect(c.seconds).toBeLessThan(1300)
    expect(c.cost).toBeCloseTo(1.02, 1)
  })

  it('counts 6Ψ per token for full fine-tuning, a third more with checkpointing, and less for LoRA', () => {
    const psi = LLAMA_3_1_8B.parameters
    expect(fineTuningCompute(LLAMA_3_1_8B, { method: 'full', checkpointing: false }, run).flopsPerToken).toBe(6 * psi)
    expect(fineTuningCompute(LLAMA_3_1_8B, { method: 'full', checkpointing: true }, run).flopsPerToken).toBe(8 * psi)
    const lora = fineTuningCompute(LLAMA_3_1_8B, { method: 'lora', checkpointing: false }, run).flopsPerToken
    expect(lora).toBe(4 * psi + 2 * adapterParameters(LLAMA_3_1_8B, 16))
    expect(fineTuningCompute(LLAMA_3_1_8B, { method: 'qlora', checkpointing: false }, run).flopsPerToken).toBe(lora)
  })
})

it('registers the fine-tuning functions', () => {
  for (const key of ['adapterParameters', 'linearParameters', 'fineTuningMemory', 'fineTuningCompute'])
    expect(quantisationFunctions[key]?.info.module).toBe('neural/quantisation')
})
