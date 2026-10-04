import { describe, it } from 'vitest'
import { caviNormalGamma } from 'aifn-methods/inference/conjugate-models'
import { expectProtocol } from '../../protocol'

describe('caviNormalGamma', () => {
  it('follows the trace protocol', () => {
    expectProtocol(caviNormalGamma([1.2, 0.7, 1.9, 1.4, 0.3, 1.1]), { expectedTau0: 0.5 }, { n: 6 })
  })
})
