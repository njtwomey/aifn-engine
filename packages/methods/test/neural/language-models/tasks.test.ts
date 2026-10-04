import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { recordActivations } from 'aifn-compute/nn/training'
import { sequenceTasks } from 'aifn-methods/data/synthetic'
import { Gpt, taskAccuracy, taskLoss, taskTrainingRun } from 'aifn-methods/neural/language-models'

describe('prompt–answer training', () => {
  it('taskLoss averages the cross-entropy over the weighted positions only', () => {
    // Two positions, two tokens: logits (0, 0) and (log 3, 0); only the second is weighted.
    const logits = fromData(Float64Array.from([0, 0, Math.log(3), 0]), [1, 2, 2])
    const targets = fromData(Int32Array.from([1, 0]), [1, 2])
    const loss = unwrap(taskLoss(logits, targets, fromData(Float64Array.from([0, 1]), [1, 2]))) as number | Tensor
    expect(typeof loss === 'number' ? loss : loss.data[0]).toBeCloseTo(-Math.log(3 / 4), 12)
  })

  it('records every module of a forward pass through the Gpt taps', () => {
    const model = Gpt({ vocabulary: 10, context: 8, width: 16, heads: 2, layers: 2 })
    const params = model.init(stream('taps'))
    const { activations } = recordActivations((ctx) => model.apply(params, [1, 2, 3, 4], ctx))
    for (const path of [
      'embedding.tokens',
      'embedding.positions',
      'embedding',
      'blocks.0.attentionNorm',
      'blocks.0.attention.weights',
      'blocks.0.attention',
      'blocks.0.residual',
      'blocks.0.feedForwardNorm',
      'blocks.0.feedForward.hidden',
      'blocks.1',
      'final',
      'logits',
    ])
      expect(path in activations, path).toBe(true)
    expect((activations['blocks.1.attention.weights'] as Tensor).shape).toEqual([2, 4, 4])
    expect((activations['blocks.0.feedForward.hidden'] as Tensor).shape).toEqual([4, 64])
    expect((activations.logits as Tensor).shape).toEqual([4, 10])
  })

  it('streams checkpoints from step 0 and learns a short reverse task', () => {
    const data = sequenceTasks(stream('reverse'), {
      task: 'reverse',
      n: 400,
      testN: 64,
      minLength: 2,
      maxLength: 3,
      symbols: 3,
    })
    const snaps = [...taskTrainingRun(data, { width: 16, heads: 2, layers: 1, steps: 120, every: 40, stepSize: 0.01 })]
    expect(snaps.map((s) => s.step)).toEqual([0, 40, 80, 120])
    const last = snaps.at(-1)!
    expect(last.checkpoints.map((c) => c.step)).toEqual([0, 40, 80, 120])
    expect(last.losses.length).toBe(121)
    expect(last.checkpoints[0].train.exact).toBeLessThan(0.2)
    expect(last.checkpoints.at(-1)!.test.token).toBeGreaterThan(0.8)
    // The checkpoint accuracy is that of its parameters.
    const model = Gpt(last.config)
    expect(taskAccuracy(model, last.checkpoints.at(-1)!.params, data.test, 256)).toEqual(last.checkpoints.at(-1)!.test)
  }, 120_000)

  it('is reproducible by seed', () => {
    const data = sequenceTasks(child(stream('seed'), 0), { task: 'copy', n: 64, testN: 16, maxLength: 3 })
    const a = [...taskTrainingRun(data, { width: 8, heads: 1, layers: 1, steps: 5, every: 5, seed: 3 })].at(-1)!
    const b = [...taskTrainingRun(data, { width: 8, heads: 1, layers: 1, steps: 5, every: 5, seed: 3 })].at(-1)!
    expect(a.losses).toEqual(b.losses)
  })
})
