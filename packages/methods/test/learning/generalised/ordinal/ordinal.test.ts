import { describe, expect, it } from 'vitest'
import { ordinalRegression } from 'aifn-methods/learning/generalised/ordinal'
import { ordinalConcordanceIndex, rankedProbabilityScore } from 'aifn-compute/learning/metrics'
import { learningModelRegistry } from 'aifn-methods/learning'
import { fromData, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { quadraticWeightedKappa } from 'aifn-compute/learning/metrics'
import type { OrdinalLinkName, OrdinalModel } from 'aifn-compute/probability/likelihoods'
import { stream } from 'aifn-compute/foundation/random'
import { fixture } from '../../../fixtures'

type Fit = { beta: number[]; theta: number[]; loglik: number; proba: number[][] }
type Fx = { x: number[][]; y: number[]; k: number; xq: number[][] } & Record<string, Fit>
const F = fixture<Fx>('learning/generalised/ordinal')
const X = tensor(F.x)
const Y = fromData(Int32Array.from(F.y), [F.y.length])

function close(got: ArrayLike<number>, want: number[], tol: number) {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) expect(Math.abs(got[i] - want[i])).toBeLessThan(tol)
}

describe('ordinalRegression', () => {
  const cases: [OrdinalModel, OrdinalLinkName][] = [
    ['cumulative', 'logit'],
    ['cumulative', 'probit'],
    ['continuation-ratio', 'logit'],
    ['adjacent-category', 'logit'],
  ]
  for (const [model, link] of cases)
    it(`${model}/${link}: the maximum-likelihood fit matches a direct scipy fit`, () => {
      const want = F[`${model}/${link}`]
      const m = ordinalRegression({ model, link }).fit(dataset(X, Y))
      expect(m.converged).toBe(true)
      expect(m.logLikelihood).toBeCloseTo(want.loglik, 6)
      close(toFlat(m.coefficients), want.beta, 1e-4)
      close(toFlat(m.thresholds), want.theta, 1e-4)
      close(toFlat(m.probabilities(tensor(F.xq))), want.proba.flat(), 1e-5)
    })

  it('predicts a categorical law over the classes; E[y], decisions and scores are consistent', () => {
    const m = ordinalRegression().fit(dataset(X, Y))
    const P = toFlat(m.probabilities(X))
    const K = F.k
    const e = toFlat(m.expect(X) as Tensor)
    const d = toFlat(m.decide(X))
    for (let i = 0; i < F.y.length; i++) {
      let s = 0
      let mean = 0
      let best = 0
      for (let k = 0; k < K; k++) {
        s += P[i * K + k]
        mean += k * P[i * K + k]
        if (P[i * K + k] > P[i * K + best]) best = k
      }
      expect(s).toBeCloseTo(1, 12)
      expect(e[i]).toBeCloseTo(mean, 10)
      expect(d[i]).toBe(best)
    }
    // The fitted model ranks and calibrates better than the class-frequency forecast.
    const c = ordinalConcordanceIndex(F.y, m.score(X))
    expect(c).toBeGreaterThan(0.75)
    const counts = new Array(K).fill(0)
    for (const v of F.y) counts[v]++
    const flat = F.y.map(() => counts.map((n) => n / F.y.length))
    const Pm = Array.from({ length: F.y.length }, (_, i) => Array.from(P.slice(i * K, (i + 1) * K)))
    expect(rankedProbabilityScore(F.y, Pm)).toBeLessThan(rankedProbabilityScore(F.y, flat))
    expect(quadraticWeightedKappa(F.y, Array.from(d))).toBeGreaterThan(0.4)
  })

  it('the L2 penalty shrinks β; the training trace decreases', () => {
    const free = ordinalRegression().fit(dataset(X, Y))
    const ridge = ordinalRegression({ l2: 50 }).fit(dataset(X, Y))
    const norm = (t: Tensor) => Math.hypot(...toFlat(t))
    expect(norm(ridge.coefficients)).toBeLessThan(norm(free.coefficients))
    const loss = toFlat(free.training.series.loss)
    expect(loss[loss.length - 1]).toBeLessThan(loss[0])
  })

  it('is registered with its capabilities and rejects bad labels', () => {
    const info = learningModelRegistry.ordinalRegression.info
    expect(info.capabilities).toEqual(['forward', 'decide', 'predictive', 'expect', 'score', 'sample'])
    expect(() => ordinalRegression().fit(dataset(X, tensor(F.y.map((v) => v + 0.5))))).toThrow(/class indices/)
    expect(() => ordinalRegression({ model: 'adjacent-category', link: 'probit' })).toThrow(/logit/)
  })

  it('sampling follows the protocol shape', () => {
    const m = ordinalRegression().fit(dataset(X, Y))
    expect(m.sample(stream('ordinal'), tensor(F.xq), 3).shape).toEqual([3, 5])
  })
})
