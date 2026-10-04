/**
 * Weak supervision against a direct NumPy Dawid–Skene EM (`fixtures/learning/weak-supervision.json`) and laws on data
 * with known truth: EM never lowers the likelihood and recovers the workers' confusions; the label model recovers the
 * labelling functions' accuracies and beats majority vote; Elkan–Noto recovers the label frequency and the prior; nnPU
 * beats the naive positive-versus-unlabelled classifier; the LLP, complementary-label and MIL models learn the classes;
 * confident learning reads the noise matrix off perfect predictions.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromRows, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  complementaryLabels,
  crowdLabels,
  instanceBags,
  labellingFunctions,
  positiveUnlabelled,
  proportionBags,
} from 'aifn-methods/data/synthetic'
import {
  attentionMil,
  complementaryClassifier,
  confidentLearning,
  dawidSkene,
  dawidSkeneSteps,
  elkanNoto,
  labelModel,
  majorityVote,
  proportionClassifier,
  puClassifier,
  votesOf,
  weakSupervisionAlgorithms,
  weakSupervisionFunctions,
} from 'aifn-methods/learning/weak-supervision'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type Run = { steps: number; posteriors: number[][]; priors: number[]; confusions: number[][][]; logLikelihood: number }
const W = fixture<{ votes: number[][]; classes: number; majority: number[][]; runs: Run[] }>(
  'learning/weak-supervision',
)

const close = (got: ArrayLike<number>, want: number[], tol: number) =>
  want.forEach((w, i) => expect(Math.abs(got[i] - w), `${i}: ${got[i]} vs ${w}`).toBeLessThanOrEqual(tol))
const argmaxRows = (t: Tensor) => {
  const [n, K] = t.shape
  const d = toFlat(t)
  return Array.from({ length: n }, (_, i) => {
    let b = 0
    for (let k = 1; k < K; k++) if (d[i * K + k] > d[i * K + b]) b = k
    return b
  })
}
const accuracy = (pred: ArrayLike<number>, truth: ArrayLike<number>) =>
  Array.from(pred).filter((p, i) => p === truth[i]).length / truth.length

describe('label models', () => {
  const votes = votesOf(W.votes)
  it('majority vote matches the reference, ties split', () =>
    close(toFlat(majorityVote(votes, W.classes)), W.majority.flat(), 1e-15))

  for (const r of W.runs)
    it(`Dawid–Skene matches a direct EM after ${r.steps} step(s)`, () => {
      const t = trace(dawidSkeneSteps(votes, W.classes), undefined, r.steps)
      const s = t.final
      close(toFlat(s.posteriors), r.posteriors.flat(), 1e-10)
      close(toFlat(s.priors), r.priors, 1e-12)
      close(toFlat(s.confusions), r.confusions.flat(2), 1e-12)
      expect(s.logLikelihood).toBeCloseTo(r.logLikelihood, 9)
    })

  it('EM never lowers the log-likelihood and recovers the workers’ confusions on crowd data', () => {
    const d = crowdLabels(stream(2), { n: 600, classes: 3, workers: 8, perItem: 4 })
    const v = votesOf(d.votes)
    const t = trace(dawidSkeneSteps(v, 3), undefined, 50)
    const ll = t.steps.map((s) => s.logLikelihood)
    for (let i = 1; i < ll.length; i++) expect(ll[i]).toBeGreaterThanOrEqual(ll[i - 1] - 1e-9)
    const s = dawidSkene(v, 3)
    const est = toFlat(s.confusions)
    const truth = toFlat(d.confusions)
    let worst = 0
    for (let w = 0; w < 8; w++)
      for (let k = 0; k < 3; k++)
        worst = Math.max(worst, Math.abs(est[(w * 3 + k) * 3 + k] - truth[(w * 3 + k) * 3 + k]))
    expect(worst).toBeLessThan(0.15)
    const y = toFlat(d.y)
    expect(accuracy(argmaxRows(s.posteriors), y)).toBeGreaterThanOrEqual(accuracy(argmaxRows(majorityVote(v, 3)), y))
  })

  it('the label model recovers the functions’ accuracies and coverages and beats majority vote', () => {
    const d = labellingFunctions(stream(4), { n: 2000, functions: 8, accuracy: [0.55, 0.95] })
    const v = votesOf(d.votes)
    const lm = labelModel(v, 2)
    close(lm.accuracy, Array.from(toFlat(d.accuracy)), 0.06)
    close(lm.coverage, Array.from(toFlat(d.coverage)), 0.04)
    const y = toFlat(d.y)
    expect(accuracy(argmaxRows(lm.posteriors), y)).toBeGreaterThan(accuracy(argmaxRows(majorityVote(v, 2)), y))
  })
})

describe('positive–unlabelled learning', () => {
  const d = positiveUnlabelled(stream(6), { n: 600, prior: 0.4, labelFrequency: 0.3 })
  const s = toFlat(d.labelled)
  const y = toFlat(d.y)
  it('Elkan–Noto recovers the label frequency and the class prior', () => {
    const en = elkanNoto(d.x, s)
    expect(Math.abs(en.labelFrequency - 0.3)).toBeLessThan(0.08)
    expect(Math.abs(en.prior - 0.4)).toBeLessThan(0.1)
    const pred = Array.from(en.positive(d.x), (p) => (p > 0.5 ? 1 : 0))
    expect(accuracy(pred, y)).toBeGreaterThan(0.85)
  })
  it('nnPU and uPU beat the naive unlabelled-as-negative classifier', () => {
    const acc = (risk: 'naive' | 'unbiased' | 'non-negative') => {
      const m = puClassifier(d.x, s, { prior: 0.4, risk })
      return accuracy(
        Array.from(toFlat(m.predict(d.x)), (p) => (p > 0.5 ? 1 : 0)),
        y,
      )
    }
    const naive = acc('naive')
    expect(acc('non-negative')).toBeGreaterThan(naive + 0.1)
    expect(acc('unbiased')).toBeGreaterThan(naive + 0.1)
  })
})

describe('proportions, complementary labels and multiple instances', () => {
  it('the proportion classifier learns the instance classes from bag proportions', () => {
    const d = proportionBags(stream(7), { bags: 25, bagSize: 16 })
    const m = proportionClassifier(d.x, toFlat(d.bag), d.proportions)
    expect(accuracy(argmaxRows(m.predict(d.x)), toFlat(d.y))).toBeGreaterThan(0.85)
  })
  it('the complementary-label classifier learns three classes', () => {
    const d = complementaryLabels(stream(8), { n: 450 })
    const m = complementaryClassifier(d.x, toFlat(d.complementary), 3)
    expect(accuracy(argmaxRows(m.predict(d.x)), toFlat(d.y))).toBeGreaterThan(0.85)
  })
  it('attention MIL classifies bags and attends to the witnesses', () => {
    const d = instanceBags(stream(9), { bags: 60 })
    const bags = toFlat(d.bag)
    const B = d.bagLabel.shape[0]
    const m = attentionMil(d.x, bags, toFlat(d.bagLabel))
    const r = m.predict(d.x, bags, B)
    expect(
      accuracy(
        Array.from(r.bags, (p) => (p > 0.5 ? 1 : 0)),
        toFlat(d.bagLabel),
      ),
    ).toBeGreaterThan(0.9)
    // In positive bags, witnesses carry more attention than the other instances.
    const inst = toFlat(d.instanceLabel)
    const lab = toFlat(d.bagLabel)
    let w = 0
    let nw = 0
    let o = 0
    let no = 0
    bags.forEach((b, i) => {
      if (lab[b] !== 1) return
      if (inst[i] === 1) ((w += r.attention[i]), nw++)
      else ((o += r.attention[i]), no++)
    })
    expect(w / nw).toBeGreaterThan(2 * (o / no))
  })
})

describe('confident learning', () => {
  it('perfect predictions give the empirical joint of noisy and true labels', () => {
    const truth = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    const noisy = [0, 0, 1, 0, 1, 1, 2, 1, 2, 0]
    const probs = fromRows(truth.map((t) => [0, 1, 2].map((k) => (k === t ? 1 : 0))))
    const r = confidentLearning(noisy, probs)
    const Q = toFlat(r.joint)
    // Q[i][j] = P(ỹ = i, y = j): (0,0) 3, (1,0) 1, (1,1) 3, (2,1) 1, (2,2) 1, (0,2) 1, out of 10.
    expect(Array.from(Q).map((v) => Math.round(v * 10))).toEqual([3, 0, 1, 1, 3, 0, 0, 1, 1])
    expect(r.issues).toEqual([2, 6, 9])
  })
})

describe('registry', () => {
  it('entries are well formed', () => {
    expectInfo(weakSupervisionAlgorithms, 'algorithm')
    expectInfo(weakSupervisionFunctions, 'function')
  })
})
