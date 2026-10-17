/** aifn-methods/neural/semi-supervised: `consistencyRun`, the plain-data run of `consistencyTraining`. */
import { describe, expect, it } from 'vitest'
import { consistencyRun, consistencyTraining, predictProbabilities } from 'aifn-methods/neural/semi-supervised'
import { stream } from 'aifn-compute/foundation/random'
import { tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { xavierUniform } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { adamRule } from 'aifn-compute/optim/first-order'

const problem = {
  labelledX: tensor([
    [-1, 0],
    [1, 0],
  ]),
  labelledY: [0, 1],
  unlabelledX: tensor([
    [-1.2, 0.3],
    [-0.8, -0.2],
    [0.9, 0.4],
    [1.1, -0.3],
    [-1, 0.6],
    [1, -0.6],
  ]),
  classes: 2,
  truth: [0, 0, 1, 1, 0, 1],
}

describe('consistencyRun', () => {
  it('keeps step 0, every `every` steps and the last, as plain arrays', () => {
    const shots = [...consistencyRun(problem, { method: { kind: 'mean-teacher' }, hidden: [6], steps: 12, every: 5 })]
    expect(shots.map((c) => c.step)).toEqual([0, 5, 10, 12])
    const last = shots.at(-1)!
    expect(last.probabilities).toBeInstanceOf(Float64Array)
    expect(last.probabilities.length).toBe(12)
    expect(last.teacherProbabilities!.length).toBe(12)
    expect(last.pseudo).toBeInstanceOf(Int32Array)
    expect(last.accuracy).toBeGreaterThanOrEqual(0)
    expect(last.evaluate.length).toBe(0)
  })

  it('reproduces consistencyTraining with the same network, seeds and Adam step', () => {
    const options = {
      method: { kind: 'fixmatch', threshold: 0.6 } as const,
      hidden: [6],
      steps: 8,
      every: 8,
      seed: 's',
    }
    const last = [...consistencyRun(problem, { ...options, rate: 0.02, evaluate: problem.labelledX })].at(-1)!
    const net = Mlp([2, 6, 2], { activation: 'tanh', init: xavierUniform() })
    const alg = consistencyTraining(problem, {
      model: net,
      method: options.method,
      steps: 8,
      optimizer: adamRule({ stepSize: 0.02 }),
    })
    const state = run(alg, { params: net.init(stream('s')) }, 8)
    expect(last.loss).toBe(state.loss)
    expect(Array.from(last.probabilities)).toEqual(
      Array.from(predictProbabilities(net, state.params, problem.unlabelledX)),
    )
    expect(Array.from(last.evaluate)).toEqual(Array.from(predictProbabilities(net, state.params, problem.labelledX)))
    expect(last.teacherProbabilities).toBeNull()
    expect(Number.isNaN(last.teacherAccuracy)).toBe(true)
  })
})
