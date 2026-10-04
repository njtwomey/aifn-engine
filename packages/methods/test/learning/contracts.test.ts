/**
 * The learning area's fitted models against the model contract: every model is `kind: 'model'` with a name;
 * classifiers' `predictive` gives class probabilities whose argmax is `decide`; regressors' `expect` is their point
 * prediction and their `predictive` mean.
 */
import { describe, expect, it } from 'vitest'
import { gaussianProcessRegressor } from 'aifn-methods/learning/gaussian-processes'
import { gaussianNaiveBayes, linearDiscriminant } from 'aifn-methods/learning/generative-classifiers'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { linearRegression } from 'aifn-methods/learning/linear'
import { kNearestNeighbours, kNearestNeighboursRegression } from 'aifn-methods/learning/neighbours'
import { decisionTree, regressionTree } from 'aifn-methods/learning/trees-and-ensembles'
import { gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { randomForest } from 'aifn-methods/learning/trees-and-ensembles/bagging'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { classProbabilities, dataset } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { fx, X3, XQ, Y3 } from './shared'

describe('classifiers', () => {
  it('predictive rows are distributions and decide is their argmax', () => {
    const d = dataset(X3, Y3)
    const models = [
      kNearestNeighbours({ k: 5 }).fit(d),
      gaussianNaiveBayes().fit(d),
      linearDiscriminant().fit(d),
      decisionTree({ maxDepth: 3 }).fit(d),
      randomForest({ trees: 5 }).fit(d, { stream: stream(1) }),
      logisticRegression({ l2: 1 }).fit(d),
    ]
    for (const m of models) {
      expect(m.kind).toBe('model')
      expect(typeof m.name).toBe('string')
      const p = toRows(classProbabilities(m.predictive(XQ)))
      const decided = toFlat(m.decide(XQ))
      p.forEach((row, i) => {
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
        const best = Math.max(...row)
        expect(row[decided[i]]).toBeCloseTo(best, 12)
      })
    }
  })
})

describe('regressors', () => {
  it('expect is the point prediction and the predictive mean', () => {
    const d = dataset(X3, tensor(fx.yreg))
    const models = [
      linearRegression().fit(d),
      regressionTree({ maxDepth: 3 }).fit(d),
      kNearestNeighboursRegression({ k: 4 }).fit(d),
      gradientBoosting({ stages: 5, learningRate: 0.3, tree: { maxDepth: 2 } }).fit(d),
      gaussianProcessRegressor({ kernel: rbf(), noiseVariance: 0.1 }).fit(d),
    ]
    for (const m of models) {
      expect(m.kind).toBe('model')
      const e = toFlat(m.expect(XQ))
      expect(e.length).toBe(XQ.shape[0])
      toFlat(m.decide(XQ)).forEach((v, i) => expect(v).toBeCloseTo(e[i], 10))
      if ('predictive' in m && typeof m.predictive === 'function')
        toFlat(m.predictive(XQ).mean() as never).forEach((v: number, i: number) => expect(v).toBeCloseTo(e[i], 10))
    }
  })
})
