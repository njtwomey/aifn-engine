import { describe, expect, it } from 'vitest'
import { forestGrowth, randomForest } from 'aifn-methods/learning/trees-and-ensembles/bagging'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { classProbabilities, dataset } from 'aifn-compute/learning/estimators'
import { expectProtocol } from '../../../protocol'
import { close, X3, XQ, Y3 } from '../../shared'

describe('random forests', () => {
  it('random forests are reproducible, extendable and sensible', () => {
    const a = randomForest({ trees: 20 }).fit(dataset(X3, Y3), { stream: stream(5) })
    const b = randomForest({ trees: 20 }).fit(dataset(X3, Y3), { stream: stream(5) })
    expect(toFlat(classProbabilities(a.predictive(XQ)))).toEqual(toFlat(classProbabilities(b.predictive(XQ))))
    const c = randomForest({ trees: 25 }).fit(dataset(X3, Y3), { stream: stream(5) })
    expect(toFlat(classProbabilities(c.predictiveUpTo(XQ, 20)))).toEqual(toFlat(classProbabilities(a.predictive(XQ))))
    expect(a.outOfBag.accuracy).toBeGreaterThan(0.8)
    close([toFlat(a.featureImportances).reduce((u, v) => u + v, 0)], [1])
  })
})

describe('trace protocol', () => {
  it('forest growth (one tree per step, bootstrap draws from each step) follows it', () => {
    expectProtocol(forestGrowth({ x: X3, y: Y3, classes: 3, params: { maxFeatures: 1 } }), undefined, { n: 5 })
  })
})
