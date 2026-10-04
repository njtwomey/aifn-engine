import { describe, it } from 'vitest'
import { unsupervisedModelRegistry } from 'aifn-methods/unsupervised'
import { expectModelProtocol } from '../registry'

describe('unsupervisedModelRegistry', () => {
  it('fits every registered estimator and finds its declared capabilities', () => {
    expectModelProtocol(unsupervisedModelRegistry)
  })
})
