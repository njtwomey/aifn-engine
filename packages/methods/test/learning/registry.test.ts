import { describe, expect, it } from 'vitest'
import { defaults } from 'aifn-compute/foundation/space'
import { learningModelRegistry } from 'aifn-methods/learning'
import { unsupervisedModelRegistry } from 'aifn-methods/unsupervised'
import { neuralModelRegistry } from 'aifn-methods/neural'
import { MODEL_FIXTURES } from '../model-fixtures'
import { expectModelProtocol } from '../registry'

describe('learningModelRegistry', () => {
  it('fits every registered estimator and finds its declared capabilities', () => {
    expectModelProtocol(learningModelRegistry)
  })
  it('has a default for every hyperparameter', () => {
    for (const m of Object.values(learningModelRegistry)) expect(typeof defaults(m.info.hyper)).toBe('object')
  })
})

describe('model fixtures', () => {
  it('belong to registered estimators only', () => {
    const keys = new Set([
      ...Object.keys(learningModelRegistry),
      ...Object.keys(unsupervisedModelRegistry),
      ...Object.keys(neuralModelRegistry),
    ])
    for (const key of Object.keys(MODEL_FIXTURES)) expect(keys.has(key), key).toBe(true)
  })
})
