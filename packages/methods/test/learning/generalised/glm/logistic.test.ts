/**
 * logisticRegression against scikit-learn's LogisticRegression (fixture `learning/linear`, generated with the
 * linear regression cases).
 */
import { describe, expect, it } from 'vitest'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { fromData, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { asTensor, dataset, evaluate, hasTraining } from 'aifn-compute/learning/estimators'
import { accuracy, auroc, logLoss } from 'aifn-compute/learning/metrics'
import { fixture } from '../../../fixtures'
import type { LinearFixture } from '../../linear/least-squares.test'

const fx = fixture<LinearFixture>('learning/linear')
const X = tensor(fx.x)
const XT = tensor(fx.x_test)

const close = (actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number) => {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tol)
}

describe('logisticRegression', () => {
  const yb = tensor(fx.binary.y)
  const binary = logisticRegression({ l2: fx.binary.l2 }).fit(dataset(X, yb))

  it('matches scikit-learn (binary)', () => {
    expect(binary.converged).toBe(true)
    close(toFlat(binary.weights), fx.binary.coef, 1e-6)
    expect(toFlat(binary.intercept)[0]).toBeCloseTo(fx.binary.intercept, 6)
    const d = binary.predictive(XT)
    close(toFlat(asTensor(d.params.probs)), fx.binary.proba, 1e-7)
    close(toFlat(binary.decide(XT)), fx.binary.predict, 0.5)
    close(toFlat(binary.expect(XT)), fx.binary.proba, 1e-7)
  })

  it('matches scikit-learn (multinomial), with intercepts summing to zero', () => {
    const m = logisticRegression({ l2: fx.multinomial.l2 }).fit(dataset(X, tensor(fx.multinomial.y)))
    expect(m.multinomial).toBe(true)
    expect(m.classes).toBe(3)
    close(toFlat(m.weights), fx.multinomial.coef.flat(), 1e-6)
    close(toFlat(m.intercept), fx.multinomial.intercept, 1e-6)
    expect(toFlat(m.intercept).reduce((s, v) => s + v, 0)).toBeCloseTo(0, 10)
    close(toFlat(asTensor(m.predictive(XT).params.probs)), fx.multinomial.proba.flat(), 1e-7)
    close(toFlat(m.decide(XT)), fx.multinomial.predict, 0.5)
    expect(m.score(XT).shape).toEqual([5, 3])
  })

  it('keeps a monotone training trace that converges in a few Newton steps', () => {
    expect(hasTraining(binary)).toBe(true)
    const t = binary.training
    expect(t.meta.stopped).toBe('done')
    // Step 0 starts IRLS from the family's initial mean, not from coefficients (its deviance is not a loss at any w);
    // every Newton step after it decreases the penalised loss.
    expect('coefficients' in t.steps[0] && t.steps[0].coefficients).toBeNull()
    const loss = toFlat(t.series.loss)
    for (let k = 2; k < loss.length; k++) expect(loss[k]).toBeLessThanOrEqual(loss[k - 1] + 1e-12)
    expect(t.steps.length).toBeLessThan(15)
    expect(binary.loss).toBeCloseTo(loss.at(-1)!, 12)
  })

  it('is evaluated by the registered classification metrics', () => {
    const r = evaluate(binary, dataset(X, yb), [logLoss, auroc, accuracy])
    const p = toFlat(asTensor(binary.predictive(X).params.probs))
    const y = toFlat(yb)
    const ll = -Array.from(y).reduce((s, yi, i) => s + Math.log(yi ? p[i] : 1 - p[i]), 0) / y.length
    expect(r.logLoss).toBeCloseTo(ll, 12)
    expect(r.auroc).toBeCloseTo(0.7275, 12) // scikit-learn's roc_auc_score
    expect(r.accuracy).toBeCloseTo(0.625, 12)
  })

  it('reports failure to converge on separable data without a penalty', () => {
    const m = logisticRegression({ l2: 0, maxSteps: 8 }).fit(
      dataset(tensor([[-2], [-1], [1], [2]]), tensor([0, 0, 1, 1])),
    )
    expect(m.converged).toBe(false)
    expect(m.training.meta.stopped).toBe('limit')
  })

  it('rejects labels that are not 0 … K−1', () => {
    expect(() => logisticRegression().fit(dataset(X, fromData(new Float64Array(40).fill(0.5))))).toThrow(/labels/)
  })
})
