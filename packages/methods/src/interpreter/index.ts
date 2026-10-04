/**
 * `aifn-methods/interpreter`: models for programs run by `aifn-compute/interpreter`, in the `learn` namespace.
 *
 * - `learn.fitLinear(X, y, l2 = 0)`: least squares (ridge when l2 > 0) by `linearRegression`;
 * - `learn.fitLogistic(X, labels, l2 = 1)`: logistic regression by `logisticRegression`;
 * - `learningPrelude` holds them, and `prelude` is the core prelude with them added.
 *
 * Each returns plain data (weights, intercept, fitted values, a goodness of fit) and a `predict` function.
 */
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  corePrelude,
  params,
  makePrelude,
  toTensor,
  withPrelude,
  type Prelude,
  type PreludeEntry,
} from 'aifn-compute/interpreter'
import { dataset } from 'aifn-compute/learning/estimators'
import { linearRegression } from 'aifn-methods/learning/linear'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'

/** A design matrix from a program's X: a vector is one feature, a matrix is rows of features. */
function design(X: unknown, what: string): Tensor {
  const t = toTensor(X, what)
  if (t.shape.length === 1) return tensor(toFlat(t).map((v) => [v]))
  if (t.shape.length !== 2) throw new TypeError(`${what}: X must be a vector or a matrix (rows of features)`)
  return t
}

const flat = (t: unknown) => Array.from(toFlat(t as Tensor))

const fitLinear: PreludeEntry = {
  namespace: 'learn',
  name: 'fitLinear',
  source: 'aifn-methods/learning/linear',
  params: params('X, y, l2 = 0'),
  doc: 'Fit y ≈ X·w + b by least squares (ridge when l2 > 0); X is a vector or rows of features.',
  returns: '{ weights, intercept, fitted, r2, noiseSd, predict(X) }',
  impl(_ctx, X, y, l2 = 0) {
    const x = design(X, 'fitLinear')
    const target = toTensor(y, 'fitLinear')
    const model = linearRegression({ l2: l2 as number }).fit(dataset(x, target))
    const fitted = flat(model.decide(x))
    const ys = flat(target)
    const ybar = ys.reduce((a, b) => a + b, 0) / ys.length
    const tss = ys.reduce((a, v) => a + (v - ybar) ** 2, 0)
    return {
      weights: flat(model.weights),
      intercept: model.intercept,
      fitted,
      r2: 1 - model.rss / tss,
      noiseSd: model.noiseSd,
      predict: (Z: unknown) => flat(model.decide(design(Z, 'predict'))),
    }
  },
}

const fitLogistic: PreludeEntry = {
  namespace: 'learn',
  name: 'fitLogistic',
  source: 'aifn-methods/learning/generalised/glm',
  params: params('X, labels, l2 = 1'),
  doc: 'Fit a two-class logistic regression P(label = 1) = σ(X·w + b) with an L2 penalty; labels are 0 or 1.',
  returns: '{ weights, intercept, probabilities, accuracy, predict(X) }',
  impl(_ctx, X, labels, l2 = 1) {
    const x = design(X, 'fitLogistic')
    const target = toTensor(labels, 'fitLogistic')
    const model = logisticRegression({ l2: l2 as number }).fit(dataset(x, target))
    if (model.multinomial) throw new TypeError('fitLogistic: labels must be 0 or 1')
    const probabilities = flat(model.expect(x))
    const ys = flat(target)
    const accuracy = ys.reduce((a, v, i) => a + ((probabilities[i] > 0.5 ? 1 : 0) === v ? 1 : 0), 0) / ys.length
    return {
      weights: flat(model.weights),
      intercept: flat(model.intercept)[0],
      probabilities,
      accuracy,
      converged: model.converged,
      predict: (Z: unknown) => flat(model.expect(design(Z, 'predict'))),
    }
  },
}

/** The `learn` namespace: models fitted to a program's data. */
export const learningPrelude: Prelude = makePrelude(
  [{ name: 'learn', doc: 'Fit models to data: linear and logistic regression.', source: 'aifn-methods/learning' }],
  [fitLinear, fitLogistic],
)

/** The core prelude with the models added. */
export const prelude: Prelude = withPrelude(corePrelude, learningPrelude)
