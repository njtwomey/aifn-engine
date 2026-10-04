import { describe, it } from 'vitest'
import { bayesPointMachine } from 'aifn-methods/inference/classifier-models'
import { expectProtocol } from '../../protocol'

describe('bayesPointMachine', () => {
  it('follows the trace protocol (step and probit likelihoods)', () => {
    const x = [
      [1, 0.5],
      [0.8, -0.2],
      [-1, 0.3],
      [-0.6, -0.9],
      [0.2, 1.1],
      [-0.3, -1.2],
    ]
    const y = [1, 1, -1, -1, 1, -1]
    for (const likelihood of ['step', 'probit'] as const)
      expectProtocol(bayesPointMachine({ x, y, likelihood }), undefined, { n: 5 })
  })
})
