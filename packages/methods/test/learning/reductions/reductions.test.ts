import { describe, expect, it } from 'vitest'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import {
  codeDistance,
  dichotomyTree,
  exhaustiveCode,
  nestedDichotomies,
  oneVersusOne,
  oneVersusOneCode,
  oneVersusRest,
  outputCode,
  randomCode,
  randomDichotomyTree,
} from 'aifn-methods/learning/reductions'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { classProbabilities, dataset } from 'aifn-compute/learning/estimators'
import { DomainError } from 'aifn-compute/foundation/errors'
import { close, fx, X3, XQ, Y3 } from '../shared'

describe('multiclass reductions', () => {
  const base = logisticRegression({ l2: 1, tolerance: 1e-14 })
  it('one-versus-rest and one-versus-one match scikit-learn with logistic regression', () => {
    const ovr = oneVersusRest(base).fit(dataset(X3, Y3))
    close(ovr.score(XQ), fx.ovr.decision, 1e-6)
    close(classProbabilities(ovr.predictive!(XQ)), fx.ovr.proba, 1e-6)
    const ovo = oneVersusOne(base).fit(dataset(X3, Y3))
    close(ovo.score(XQ), fx.ovo.decision, 1e-6)
  })
  it('codes: sizes and distances', () => {
    expect(oneVersusOneCode(4).shape).toEqual([4, 6])
    expect(exhaustiveCode(4).shape).toEqual([4, 7])
    expect(codeDistance(exhaustiveCode(5))).toBe(8)
    const r = randomCode(stream(1), 5, 10)
    expect(r.shape).toEqual([5, 10])
    expect(toFlat(randomCode(stream(1), 5, 10))).toEqual(toFlat(r))
  })
  it('output codes with the one-versus-rest code agree with one-versus-rest', () => {
    const ecoc = outputCode(base, exhaustiveCode(3), { decoding: 'loss' }).fit(dataset(X3, Y3))
    const acc = toFlat(ecoc.decide(X3)).filter((c, i) => c === fx.y3[i]).length / 60
    expect(acc).toBeGreaterThan(0.85)
  })
  it('nested dichotomies give normalised probabilities', () => {
    for (const shape of ['balanced', 'chain'] as const) {
      const m = nestedDichotomies(base, (K) => dichotomyTree(K, shape)).fit(dataset(X3, Y3))
      for (const row of toRows(classProbabilities(m.predictive(XQ))))
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    }
  })
  it('code and tree builders reject too few classes instead of looping', () => {
    expect(() => randomCode(stream(0), 1, 3)).toThrow(DomainError)
    expect(() => randomCode(stream(0), 3, -1)).toThrow(DomainError)
    expect(() => dichotomyTree(0)).toThrow(DomainError)
    expect(() => randomDichotomyTree(stream(0), 0)).toThrow(DomainError)
    expect(dichotomyTree(1)).toBe(0)
  })
})
