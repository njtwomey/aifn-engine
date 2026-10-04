/**
 * The ordinal family beyond the latent-variable models: threshold losses and their linear model, Frank and Hall's
 * binary decomposition, and the deep ordinal heads. Golden values from fixtures/gen/learning/generalised/ordinal.py.
 */
import { describe, expect, it } from 'vitest'
import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { ordinalMeanAbsoluteError } from 'aifn-compute/learning/metrics'
import {
  allThresholdLoss,
  binaryDecomposition,
  deepOrdinalRegression,
  differenceExceedance,
  immediateThresholdLoss,
  ordinalRegression,
  thresholdClasses,
  thresholdOrdinalRegression,
  thresholdPenalty,
} from 'aifn-methods/learning/generalised/ordinal'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { learningModelRegistry } from 'aifn-methods/learning'
import { fixture } from '../../../fixtures'

type ThresholdFit = { w: number[]; theta: number[]; objective: number }
type Fx = {
  x: number[][]
  y: number[]
  k: number
  xq: number[][]
  'frank-hall': { q: number[][]; proba: number[][] }
} & Record<string, ThresholdFit>
const F = fixture<Fx>('learning/generalised/ordinal')
const X = tensor(F.x)
const Y = fromData(Int32Array.from(F.y), [F.y.length])
const num = (v: Value) => (typeof v === 'number' ? v : toFlat(v as Tensor)[0])

function close(got: ArrayLike<number>, want: number[], tol: number) {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++) expect(Math.abs(got[i] - want[i]), `[${i}]`).toBeLessThan(tol)
}

describe('threshold losses (Rennie & Srebro, 2005)', () => {
  it('penalties: hinge, smooth hinge, logistic, modified least squares', () => {
    const z = tensor([-1, 0, 0.5, 1, 2])
    close(toFlat(thresholdPenalty(z, 'hinge') as Tensor), [2, 1, 0.5, 0, 0], 1e-15)
    close(toFlat(thresholdPenalty(z, 'smooth-hinge') as Tensor), [1.5, 0.5, 0.125, 0, 0], 1e-15)
    close(toFlat(thresholdPenalty(z, 'modified-least-squares') as Tensor), [4, 1, 0.25, 0, 0], 1e-15)
    close(
      toFlat(thresholdPenalty(z, 'logistic') as Tensor),
      [-1, 0, 0.5, 1, 2].map((v) => Math.log1p(Math.exp(-v))),
      1e-14,
    )
  })

  it('all-threshold charges every threshold on the wrong side; immediate only the two beside the class', () => {
    // θ = (−1, 0, 1), s = 0.5. Label 0: all wants θ_k > s for every k; immediate only θ₀.
    const theta = tensor([-1, 0, 1])
    const h = (z: number) => Math.max(0, 1 - z)
    const at = (y: number, f: typeof allThresholdLoss) =>
      num(f(tensor([0.5]), [y], theta, { penalty: 'hinge', reduction: 'sum' }))
    expect(at(0, allThresholdLoss)).toBeCloseTo(h(-1 - 0.5) + h(0 - 0.5) + h(1 - 0.5), 14)
    expect(at(0, immediateThresholdLoss)).toBeCloseTo(h(-1 - 0.5), 14)
    expect(at(2, allThresholdLoss)).toBeCloseTo(h(0.5 + 1) + h(0.5 - 0) + h(1 - 0.5), 14)
    expect(at(2, immediateThresholdLoss)).toBeCloseTo(h(0.5 - 0) + h(1 - 0.5), 14)
    expect(at(3, immediateThresholdLoss)).toBeCloseTo(h(0.5 - 1), 14)
    // With the hinge, the all-threshold loss bounds the absolute error of ŷ = #{θ_k < s}.
    const s = [-2, -0.5, 0.2, 0.7, 3]
    const pred = thresholdClasses(s, [-1, 0, 1])
    expect(Array.from(pred)).toEqual([0, 1, 2, 2, 3])
    for (let i = 0; i < s.length; i++)
      for (let y = 0; y <= 3; y++)
        expect(num(allThresholdLoss(tensor([s[i]]), [y], theta, { penalty: 'hinge' }))).toBeGreaterThanOrEqual(
          Math.abs(pred[i] - y) - 1e-12,
        )
  })

  it('differentiates in the scores and the thresholds', () => {
    const { grad } = valueAndGrad(
      (t: Value) => allThresholdLoss(tensor([0.3, -0.2]), [1, 2], t),
      {},
    )(tensor([-0.5, 0.1, 0.9]))
    expect(toFlat(grad as Tensor).every(Number.isFinite)).toBe(true)
  })

  const cases: [string, 'all' | 'immediate', 'logistic' | 'smooth-hinge', boolean][] = [
    ['threshold/all/logistic', 'all', 'logistic', true],
    ['threshold/all/smooth-hinge', 'all', 'smooth-hinge', true],
    ['threshold/immediate/logistic', 'immediate', 'logistic', false],
  ]
  for (const [key, construction, penalty, ordered] of cases)
    it(`${construction}/${penalty}: the fit matches a direct scipy fit`, () => {
      const want = F[key]
      const m = thresholdOrdinalRegression({ construction, penalty, ordered }).fit(dataset(X, Y))
      expect(m.converged).toBe(true)
      expect(m.objective).toBeCloseTo(want.objective, 6)
      close(toFlat(m.coefficients), want.w, 1e-4)
      close(toFlat(m.thresholds), want.theta, 1e-4)
    })

  it('immediate thresholds go out of order on a sparse class; ordered: true keeps them increasing at a cost', () => {
    const free = thresholdOrdinalRegression({ construction: 'immediate', ordered: false }).fit(dataset(X, Y))
    const kept = thresholdOrdinalRegression({ construction: 'immediate' }).fit(dataset(X, Y))
    const t = toFlat(kept.thresholds)
    for (let k = 1; k < t.length; k++) expect(t[k]).toBeGreaterThan(t[k - 1])
    expect(kept.objective).toBeGreaterThanOrEqual(free.objective - 1e-9)
    const d = toFlat(kept.decide(X))
    expect(ordinalMeanAbsoluteError(F.y, Array.from(d))).toBeLessThan(1)
  })
})

describe('binary decomposition (Frank & Hall, 2001)', () => {
  it('matches scikit-learn logistic regressions per split, differenced and clipped', () => {
    const m = binaryDecomposition({ l2: 1 }).fit(dataset(X, Y))
    close(toFlat(m.exceedance(tensor(F.xq))), F['frank-hall'].q.flat(), 1e-6)
    close(toFlat(m.probabilities(tensor(F.xq))), F['frank-hall'].proba.flat(), 1e-6)
  })

  it('repairs incoherent exceedances by clipping or sorting (the note’s example)', () => {
    // q = (0.9, 0.3, 0.4) differences to (0.1, 0.6, −0.1, 0.4).
    close(differenceExceedance([0.9, 0.3, 0.4], 1, 4, 'clip'), [0.1 / 1.1, 0.6 / 1.1, 0, 0.4 / 1.1], 1e-14)
    close(differenceExceedance([0.9, 0.3, 0.4], 1, 4, 'sort'), [0.1, 0.5, 0.1, 0.3], 1e-14)
  })

  it('takes any Bernoulli-predictive base estimator', () => {
    const viaBase = binaryDecomposition({ base: logisticRegression({ l2: 1 }) }).fit(dataset(X, Y))
    const direct = binaryDecomposition({ l2: 1 }).fit(dataset(X, Y))
    close(toFlat(viaBase.probabilities(tensor(F.xq))), Array.from(toFlat(direct.probabilities(tensor(F.xq)))), 1e-6)
  })
})

describe('deep ordinal regression', () => {
  it('CORAL: the biases end decreasing, so P(y > k) decreases in k; it fits the data', () => {
    const m = deepOrdinalRegression({ head: 'coral', hidden: [8], steps: 400 }).fit(dataset(X, Y), {
      stream: stream('coral'),
    })
    const b = toFlat(m.cutpoints)
    for (let k = 1; k < b.length; k++) expect(b[k]).toBeLessThan(b[k - 1])
    const q = toFlat(m.exceedance(X))
    for (let i = 0; i < F.y.length; i++)
      for (let k = 1; k < 3; k++) expect(q[i * 3 + k]).toBeLessThanOrEqual(q[i * 3 + k - 1])
    const loss = toFlat(m.training.series.loss)
    expect(loss[loss.length - 1]).toBeLessThan(0.7 * loss[0])
    expect(ordinalMeanAbsoluteError(F.y, Array.from(toFlat(m.decide(X))))).toBeLessThan(0.6)
  })

  it('cumulative head with no hidden layer approaches the maximum-likelihood ordinal regression', () => {
    const deep = deepOrdinalRegression({ head: 'cumulative', hidden: [], steps: 3000, learningRate: 0.05 }).fit(
      dataset(X, Y),
      { stream: stream('cumulative') },
    )
    const mle = ordinalRegression().fit(dataset(X, Y))
    close(toFlat(deep.cutpoints), Array.from(toFlat(mle.thresholds)), 2e-2)
    close(toFlat(deep.probabilities(tensor(F.xq))), Array.from(toFlat(mle.probabilities(tensor(F.xq)))), 1e-2)
  })

  it('is reproducible from its stream', () => {
    const fit = () =>
      toFlat(
        deepOrdinalRegression({ hidden: [4], steps: 30 })
          .fit(dataset(X, Y), { stream: stream(3) })
          .probabilities(tensor(F.xq)),
      )
    expect(fit()).toEqual(fit())
  })
})

describe('registry', () => {
  it('registers the family with its capabilities', () => {
    expect(learningModelRegistry.thresholdOrdinalRegression.info.capabilities).toEqual(['forward', 'decide', 'score'])
    for (const key of ['binaryDecomposition', 'deepOrdinalRegression'])
      expect(learningModelRegistry[key].info.capabilities).toEqual([
        'forward',
        'decide',
        'predictive',
        'expect',
        'score',
        'sample',
      ])
  })
})
