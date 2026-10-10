import { describe, expect, it } from 'vitest'
import {
  adaBoost,
  adaBoostSteps,
  gradientBoosting,
  gradientBoostingSteps,
} from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { expectProtocol } from '../../../protocol'
import { close, fx, X2, X3, XQ, Y2, Y3 } from '../../shared'

describe('ensembles', () => {
  it('AdaBoost (SAMME) matches scikit-learn', () => {
    const m = adaBoost({ rounds: 5 }).fit(dataset(X3, Y3))
    close(m.alphas, fx.ada.weights)
    close(m.errors, fx.ada.errors)
    expect(toFlat(m.decide(XQ))).toEqual(fx.ada.predict)
    expectProtocol(adaBoostSteps({ x: X3, y: Y3 }), undefined, {
      n: 6,
      record: { k: (s) => s.alphas.reduce((a, b) => a + b, 0) },
    })
  })
  it('gradient boosting matches scikit-learn (squared, binomial, multinomial)', () => {
    const r = gradientBoosting({ stages: 5, learningRate: 0.3, tree: { maxDepth: 2 } }).fit(
      dataset(X3, tensor(fx.yreg)),
    )
    fx.gbr.staged.forEach((row, s) => close(r.rawUpTo(XQ, s + 1), row))
    const b = gradientBoosting({ loss: 'logistic', stages: 5, learningRate: 0.3, tree: { maxDepth: 2 } }).fit(
      dataset(X2, Y2),
    )
    close(b.forward(XQ), fx.gbc.decision)
    close(b.predictive(XQ), fx.gbc.proba)
    const k = gradientBoosting({ loss: 'logistic', stages: 4, learningRate: 0.3, tree: { maxDepth: 2 } }).fit(
      dataset(X3, Y3),
    )
    close(k.predictive(XQ), fx.gbm.proba)
    expect(k.training.series.loss.data[4]).toBeLessThan(k.training.series.loss.data[0])
  })
  it('gradient boosting with all labels one class starts with finite log-odds', () => {
    const x = tensor([[1], [2], [3], [4]])
    const y = tensor([1, 1, 1, 1])
    const b = gradientBoosting({ loss: 'logistic', stages: 2 }).fit(dataset(x, y))
    expect(Number.isFinite(b.initial[0])).toBe(true)
    expect(Number.isFinite(b.training.final.loss)).toBe(true)
  })
})

describe('trace protocol', () => {
  it('gradient boosting (squared, logistic, stochastic) follows it', () => {
    expectProtocol(gradientBoostingSteps({ x: X3, y: tensor(fx.yreg), loss: 'squared' }), undefined, { n: 5 })
    expectProtocol(gradientBoostingSteps({ x: X3, y: Y3, loss: 'logistic', tree: { maxDepth: 2 } }), undefined, {
      n: 4,
    })
    expectProtocol(gradientBoostingSteps({ x: X3, y: tensor(fx.yreg), loss: 'squared', subsample: 0.5 }), undefined, {
      n: 5,
    })
  })
})
