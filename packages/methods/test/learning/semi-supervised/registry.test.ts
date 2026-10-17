/** The registry of `aifn-methods/learning/semi-supervised`: every entry is well formed and keyed by its export name. */
import { describe, it } from 'vitest'
import { semiSupervisedAlgorithms, semiSupervisedFunctions } from 'aifn-methods/learning/semi-supervised'
import { expectInfo } from '../../registry'

describe('semi-supervised registry', () => {
  it('has well-formed algorithm and function entries', () => {
    expectInfo(semiSupervisedAlgorithms, 'algorithm')
    expectInfo(semiSupervisedFunctions, 'function')
  })
})
