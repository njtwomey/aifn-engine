/**
 * The learning-rate sweep against the same sweep composed by hand from `moons`, `Mlp` and `trainingLoop` (as the
 * LearningRateSweep figure does), its divergence flags, and the worker path.
 */
import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import type { Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { heNormal } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { trainingLoop } from 'aifn-compute/nn/training'
import { sgdRule } from 'aifn-compute/optim/first-order'
import { moons } from 'aifn-methods/data/synthetic'
import { learningRateSweep, tuningFunctions } from 'aifn-methods/neural/tuning'

const column = (y: Tensor) => fromData(Float64Array.from(toFlat(y)), [y.shape[0], 1])

describe('learningRateSweep', () => {
  it('matches the sweep composed from moons, Mlp and trainingLoop, stream for stream', () => {
    const [rates, steps, batch] = [[0.01, 0.1], 15, 8]
    const train = moons(stream('check/train'), { n: 51, noise: 0.3 })
    const valid = moons(stream('check/validation'), { n: 101, noise: 0.3 })
    const net = Mlp([2, 6, 1], { activation: 'relu', init: heNormal() })
    const expected = rates.map((rate) =>
      [1, 2].map((k) => {
        const final = run(
          trainingLoop({
            loss: (w: Params[], b: { x: Tensor; y: Tensor }) => binaryCrossEntropyWithLogits(net.apply(w, b.x), b.y),
            data: { x: train.x, y: column(train.y!) },
            batchSize: batch,
            optimizer: sgdRule({ stepSize: rate, momentum: 0.5 }),
          }),
          { params: net.init(stream(`check/init/${k}`)) },
          steps,
          { stream: stream(`check/order/${k}`) },
        )
        return Number(unwrap(binaryCrossEntropyWithLogits(net.apply(final.params, valid.x), column(valid.y!))))
      }),
    )
    const r = learningRateSweep({
      widths: [6],
      optimiser: 'sgd',
      momentum: 0.5,
      rates,
      seeds: 2,
      batch,
      steps,
      train: 51,
      validation: 101,
      noise: 0.3,
      seed: 'check',
    })
    expect(r.rates).toEqual(rates)
    expect(r.losses).toEqual(expected)
    expect(r.diverged).toEqual([
      [false, false],
      [false, false],
    ])
  })

  it('flags a run that diverges and reports NaN for it', () => {
    const r = learningRateSweep({ widths: [4], optimiser: 'sgd', rates: [1e4], seeds: 1, steps: 40, batch: 16 })
    expect(r.diverged).toEqual([[true]])
    expect(r.losses[0][0]).toBeNaN()
    expect(() => learningRateSweep({ seeds: 0 })).toThrow(DomainError)
    expect(() => learningRateSweep({ rates: [0] })).toThrow(DomainError)
  })
})

describe('worker path', () => {
  it('runs from a task and the result survives structured cloning', async () => {
    const options = { widths: [4], rates: [0.01, 0.1], seeds: 1, steps: 5, batch: 16 }
    const task = { $call: 'neural/tuning/learningRateSweep', args: [options] }
    const ns = (await import(`aifn-methods/${task.$call.slice(0, task.$call.lastIndexOf('/'))}`)) as Record<
      string,
      unknown
    >
    const f = ns[task.$call.slice(task.$call.lastIndexOf('/') + 1)] as (...a: unknown[]) => unknown
    const result = f(...task.args)
    expect(structuredClone(result)).toEqual(result)
    expect(result).toEqual(learningRateSweep(options))
    expect(Object.keys(tuningFunctions)).toEqual(['learningRateSweep'])
  })
})
