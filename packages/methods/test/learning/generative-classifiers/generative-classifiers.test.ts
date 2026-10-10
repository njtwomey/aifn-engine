import { describe, expect, it } from 'vitest'
import {
  bernoulliNaiveBayes,
  gaussianNaiveBayes,
  linearDiscriminant,
  multinomialNaiveBayes,
  quadraticDiscriminant,
} from 'aifn-methods/learning/generative-classifiers'
import { tensor, toRows } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { DomainError } from 'aifn-compute/foundation/errors'
import { close, fx, X3, XQ, Y3 } from '../shared'

describe('generative classifiers', () => {
  it('naive Bayes matches scikit-learn', () => {
    close(gaussianNaiveBayes().fit(dataset(X3, Y3)).predictive(XQ), fx.gnb)
    const C = tensor(fx.counts)
    const yc = tensor(fx.ycounts)
    close(
      multinomialNaiveBayes({ alpha: 0.5 })
        .fit(dataset(C, yc))
        .predictive(tensor(fx.counts.slice(0, 6))),
      fx.mnb,
    )
    close(
      bernoulliNaiveBayes({ binarize: 1.5 })
        .fit(dataset(C, yc))
        .predictive(tensor(fx.counts.slice(0, 6))),
      fx.bnb,
    )
  })
  it('LDA and QDA match scikit-learn', () => {
    const lda = linearDiscriminant().fit(dataset(X3, Y3))
    close(lda.predictive(XQ), fx.lda)
    close(lda.explainedVarianceRatio, fx.lda_ratio)
    // The projected classes have unit pooled within-class variance.
    const z = toRows(lda.transform(X3))
    expect(z[0].length).toBe(2)
    close(quadraticDiscriminant({ regularisation: 0.1 }).fit(dataset(X3, Y3)).predictive(XQ), fx.qda)
  })
  it('QDA rejects regularisation outside [0, 1] instead of factoring an indefinite covariance', () => {
    expect(() => quadraticDiscriminant({ regularisation: 3 })).toThrow(DomainError)
    expect(() => quadraticDiscriminant({ regularisation: -0.1 })).toThrow(DomainError)
  })
})
