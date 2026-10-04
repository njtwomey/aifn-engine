import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { modularArithmetic } from 'aifn-methods/data/synthetic'
import { grokkingRun, ModularMlp, pairsOf } from 'aifn-methods/neural/grokking'

describe('grokking', () => {
  it('maps pairs to logits over the residues', () => {
    const model = ModularMlp({ p: 7, embed: 4, width: 8 })
    const params = model.init(stream('mlp'))
    const data = modularArithmetic(stream('table'), { p: 7 })
    const pairs = pairsOf(data.table)
    expect(toFlat(pairs.a).slice(0, 8)).toEqual([0, 0, 0, 0, 0, 0, 0, 1])
    expect((unwrap(model.apply(params, pairs)) as Tensor).shape).toEqual([49, 7])
  })

  it('fits the training pairs and records curves and checkpoints from step 0', () => {
    const data = modularArithmetic(stream('fit'), { p: 11, fraction: 0.5 })
    const snaps = [...grokkingRun(data, { embed: 8, width: 32, steps: 200, recordEvery: 20, checkpointEvery: 100 })]
    expect(snaps.map((s) => s.step)).toEqual([0, 100, 200])
    const last = snaps.at(-1)!
    expect(last.curves.steps).toEqual(Array.from({ length: 11 }, (_, i) => 20 * i))
    expect(last.checkpoints.map((c) => c.step)).toEqual([0, 100, 200])
    expect(last.curves.trainAccuracy.at(-1)).toBe(1)
    expect(last.curves.trainAccuracy[0]).toBeLessThan(0.5)
    // The embedding's spectrum is a distribution over frequencies 0 … 5.
    const spectrum = data.truth.spectrum(last.checkpoints.at(-1)!.params.embedding)
    expect(spectrum.length).toBe(6)
    expect(spectrum.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('generalises late on modular addition with weight decay (the page default, shortened)', () => {
    const data = modularArithmetic(stream('grokking-data-0'), { p: 31, fraction: 0.5 })
    const snaps = [...grokkingRun(data, { steps: 1500, recordEvery: 50, checkpointEvery: 1500 })]
    const c = snaps.at(-1)!.curves
    const at = (s: number) => c.steps.indexOf(s)
    expect(c.trainAccuracy[at(200)]).toBe(1)
    expect(c.testAccuracy[at(200)]).toBeLessThan(0.1)
    expect(c.testAccuracy.at(-1)).toBeGreaterThan(0.6)
  }, 180_000)
})

describe('grokkingRun by full-batch L-BFGS', () => {
  it('fits the training pairs and may stop early', () => {
    const data = modularArithmetic(stream('lbfgs'), { p: 7, fraction: 0.6 })
    const snaps = [
      ...grokkingRun(data, {
        embed: 4,
        width: 16,
        steps: 150,
        recordEvery: 10,
        checkpointEvery: 50,
        method: 'lbfgs',
        weightDecay: 0.01,
      }),
    ]
    const last = snaps.at(-1)!
    expect(last.step).toBe(last.steps)
    expect(last.curves.trainAccuracy.at(-1)).toBe(1)
  })
})
