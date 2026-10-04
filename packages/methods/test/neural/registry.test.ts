import { describe, it } from 'vitest'
import { neuralModelRegistry } from 'aifn-methods/neural'
import { expectModelProtocol } from '../registry'

describe('neural model registry', () => {
  it('every registered language model fits and has its declared capabilities', () => {
    expectModelProtocol(neuralModelRegistry)
  })
})
