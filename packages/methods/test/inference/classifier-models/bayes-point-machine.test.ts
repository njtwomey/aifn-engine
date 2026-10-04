import { describe, expect, it } from 'vitest'
import { bayesPointMachine, bayesPointMachinePredict } from 'aifn-methods/inference/classifier-models'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'

describe('Bayes point machine', () => {
  const x = [
    [1, 2, 1],
    [2, 1, 1],
    [1.5, 1.5, 1],
    [-1, -2, 1],
    [-2, -0.5, 1],
    [-1.5, -1, 1],
  ]
  const y = [1, 1, 1, -1, -1, -1]
  it.each(['probit', 'step'] as const)('%s: converges and separates the training data', (likelihood) => {
    const s = run(bayesPointMachine({ x, y, likelihood }), undefined, 1000)
    expect(s.converged).toBe(true)
    const p = toFlat(bayesPointMachinePredict(s, x))
    p.forEach((pi, i) => expect(y[i] > 0 ? pi > 0.5 : pi < 0.5).toBe(true))
  })
})
