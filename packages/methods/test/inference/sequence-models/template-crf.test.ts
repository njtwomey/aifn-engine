import { describe, expect, it } from 'vitest'
import {
  crfNegLogLikelihood,
  crfppRegularisation,
  crfProblem,
  crfTraining,
  crfTrainingRun,
  firingFeatures,
  templateCrf,
  templateCrfMarginals,
  templateCrfPosterior,
  templateCrfPotentials,
  templateCrfScore,
  templateCrfViterbi,
  topFeatures,
  transitionWeights,
  type LabelledSequence,
} from 'aifn-methods/inference/sequence-models'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { parseTemplates } from 'aifn-compute/text/features'
import { fixture } from '../../fixtures'
import { expectProtocol } from '../../protocol'

type Golden = {
  templates: string
  c2: number
  train: { rows: string[][]; labels: string[] }[]
  state: [string, string, number][]
  transitions: [string, string, number][]
  test: { rows: string[][]; viterbi: string[]; labels: string[]; marginals: number[][] }[]
}
const golden = fixture<Golden>('inference/sequence-models')

// A small problem with bigram features conjoined with an observation, for the exact checks.
const toy: LabelledSequence[] = [
  {
    rows: [
      ['a', 'v'],
      ['b', 'c'],
      ['a', 'v'],
      ['c', 'c'],
    ],
    labels: ['X', 'Y', 'X', 'Z'],
  },
  {
    rows: [
      ['b', 'c'],
      ['b', 'c'],
      ['a', 'v'],
    ],
    labels: ['Y', 'Y', 'X'],
  },
  {
    rows: [
      ['c', 'c'],
      ['a', 'v'],
      ['b', 'c'],
      ['a', 'v'],
      ['b', 'c'],
    ],
    labels: ['Z', 'X', 'Y', 'Z', 'Y'],
  },
]
const toyTemplates = parseTemplates('U00:%x[0,0]\nU01:%x[-1,0]\nU02:%x[0,0]/%x[1,1]\nB\nB01:%x[0,1]')

function randomCrf(seed: number) {
  const p = crfProblem(toyTemplates, toy)
  const w = Array.from(toFlat(uniform(child(stream(seed), 'w'), -1.5, 1.5, { shape: [p.dimension] })))
  return { p, crf: templateCrf(p.index, p.labels, w) }
}

/** Every labelling of length N over K labels. */
function* labellings(N: number, K: number): Generator<number[]> {
  const y = new Array<number>(N).fill(0)
  for (;;) {
    yield [...y]
    let i = N - 1
    while (i >= 0 && y[i] === K - 1) y[i--] = 0
    if (i < 0) return
    y[i]++
  }
}

describe('a linear-chain CRF over feature templates', () => {
  it('Z, the marginals and the Viterbi path match brute-force enumeration', () => {
    const { crf } = randomCrf(1)
    const rows = toy[2].rows
    const { logUnary, logPairwise } = templateCrfPotentials(crf, rows)
    const K = crf.labels.length
    const N = rows.length
    const U = toFlat(logUnary)
    const P = toFlat(logPairwise)
    const score = (y: number[]) =>
      y.reduce((a, k, n) => a + U[n * K + k] + (n > 0 ? P[(n - 1) * K * K + y[n - 1] * K + k] : 0), 0)
    let Z = 0
    let best = -Infinity
    let argbest: number[] = []
    const marg = new Float64Array(N * K)
    for (const y of labellings(N, K)) {
      const s = score(y)
      Z += Math.exp(s)
      y.forEach((k, n) => (marg[n * K + k] += Math.exp(s)))
      if (s > best) [best, argbest] = [s, y]
    }
    const m = templateCrfMarginals(crf, rows)
    expect(m.logZ).toBeCloseTo(Math.log(Z), 10)
    toFlat(m.marginals).forEach((p, i) => expect(p).toBeCloseTo(marg[i] / Z, 10))
    const v = templateCrfViterbi(crf, rows)
    expect(Array.from(toFlat(v.path))).toEqual(argbest)
    expect(v.logProbability).toBeCloseTo(best, 10)
    expect(v.labels).toEqual(argbest.map((k) => crf.labels[k]))
    // Posterior decoding is the argmax of the enumerated marginals; its score is the path's.
    const d = templateCrfPosterior(crf, rows)
    const argmax = Array.from({ length: N }, (_, n) => {
      const row = Array.from({ length: K }, (_, k) => marg[n * K + k])
      return row.indexOf(Math.max(...row))
    })
    expect(Array.from(toFlat(d.path))).toEqual(argmax)
    expect(d.logScore).toBeCloseTo(score(argmax), 10)
    expect(templateCrfScore(crf, rows, argbest)).toBeCloseTo(best, 10)
    expect(d.expectedCorrect).toBeCloseTo(
      argmax.reduce((a, k, n) => a + marg[n * K + k] / Z, 0),
      10,
    )
  })

  it('the gradient of the NLL matches central finite differences', () => {
    const { p, crf } = randomCrf(2)
    const w = crf.weights
    const g = new Float64Array(w.length)
    crfNegLogLikelihood(w, p.encoded, p.U, p.K, g)
    const h = 1e-5
    for (let i = 0; i < w.length; i++) {
      const up = Float64Array.from(w)
      const dn = Float64Array.from(w)
      up[i] += h
      dn[i] -= h
      const fd = (crfNegLogLikelihood(up, p.encoded, p.U, p.K) - crfNegLogLikelihood(dn, p.encoded, p.U, p.K)) / (2 * h)
      expect(Math.abs(g[i] - fd)).toBeLessThan(1e-6)
    }
  })

  it('reports the features firing at a position, with their template cells and weights', () => {
    const { crf } = randomCrf(3)
    const rows = toy[0].rows
    const f = firingFeatures(crf, rows, 1)
    expect(f.map((x) => x.string)).toEqual(['U00:b', 'U01:a', 'U02:b/v', 'B', 'B01:c'])
    expect(f[1].macros).toEqual([{ row: -1, column: 0, start: 4, end: 12 }])
    // ψ_1 is the sum of the unigram weights that fire.
    const U = toFlat(templateCrfPotentials(crf, rows).logUnary)
    for (let k = 0; k < crf.labels.length; k++)
      expect(U[1 * crf.labels.length + k]).toBeCloseTo(
        f.filter((x) => x.kind === 'unigram').reduce((a, x) => a + x.weights[k], 0),
        12,
      )
    expect(firingFeatures(crf, rows, 0).some((x) => x.kind === 'bigram')).toBe(false)
    expect(transitionWeights(crf)?.length).toBe(3)
    expect(topFeatures(crf, 2, 0)).toHaveLength(2)
  })

  it('firingFeatures throws DomainError for a row missing a column a template reads', () => {
    const { crf } = randomCrf(3)
    const rows = toy[0].rows.map((r) => r.slice(0, 1))
    expect(() => firingFeatures(crf, rows, 1)).toThrow(/firingFeatures: .*column 1/)
    expect(() => firingFeatures(crf, rows, 1)).toThrow(expect.objectContaining({ name: 'DomainError' }))
  })

  it('matches CRFsuite (L-BFGS, L2): weights, held-out marginals and Viterbi tags', () => {
    const templates = parseTemplates(golden.templates)
    const train: LabelledSequence[] = golden.train
    const p = crfProblem(templates, train, { labels: golden.test[0].labels })
    const s = run(crfTraining(p, { c2: golden.c2, tolerance: 1e-7 }), undefined, 2000)
    expect(s.converged).toBe(true)
    const crf = templateCrf(p.index, p.labels, toFlat(s.weights))
    const K = p.K
    const k = new Map(p.labels.map((y, i) => [y, i]))
    for (const [attr, label, w] of golden.state) {
      const u = p.index.unigramIds.get(attr)!
      expect(Math.abs(crf.weights[u * K + k.get(label)!] - w)).toBeLessThan(2e-5)
    }
    const T = transitionWeights(crf)!
    for (const [a, b, w] of golden.transitions) expect(Math.abs(T[k.get(a)!][k.get(b)!] - w)).toBeLessThan(2e-5)
    for (const t of golden.test) {
      expect(templateCrfViterbi(crf, t.rows).labels).toEqual(t.viterbi)
      const m = toFlat(templateCrfMarginals(crf, t.rows).marginals)
      t.marginals.flat().forEach((q, i) => expect(Math.abs(m[i] - q)).toBeLessThan(1e-4))
    }
  })

  it('OWL-QN zeroes weights as c₁ grows; SGD and Adam lower the objective', () => {
    const templates = parseTemplates(golden.templates)
    const p = crfProblem(templates, golden.train)
    const active = [0.05, 1, 5].map(
      (c1) => run(crfTraining(p, { optimizer: 'owlqn', c1, c2: 0.01 }), undefined, 300).active,
    )
    expect(active[1]).toBeLessThan(active[0])
    expect(active[2]).toBeLessThan(active[1])
    expect(active[2]).toBeGreaterThan(0)
    const full = run(crfTraining(p, { optimizer: 'sgd', batchSize: 0, stepSize: 0.5, c2: 0.01 }), undefined, 20)
    expect(full.nll).toBeLessThan(run(crfTraining(p, { optimizer: 'sgd', batchSize: 0, c2: 0.01 }), undefined, 0).nll)
    for (const optimizer of ['sgd', 'adam'] as const) {
      const alg = crfTraining(p, { optimizer, c2: 0.01 })
      const s0 = alg.init(undefined, stream(0))
      const s = run(alg, undefined, 5)
      expect(s.objective).toBeLessThan(0.5 * s0.objective)
    }
    expect(crfppRegularisation(2)).toEqual({ c1: 0, c2: 0.25 })
  })

  it('follows the Algorithm protocol, and the run yields from step 0', () => {
    const p = crfProblem(toyTemplates, toy)
    expectProtocol(crfTraining(p, { optimizer: 'owlqn', c1: 0.1 }), undefined, { record: { f: (s) => s.objective } })
    expectProtocol(crfTraining(p, { optimizer: 'adam' }), undefined, { record: { f: (s) => s.objective } })
    const snaps = [...crfTrainingRun(toyTemplates, toy, { maxSteps: 5 })]
    expect(snaps[0].step).toBe(0)
    expect(snaps.at(-1)!.history.nll.length).toBe(snaps.at(-1)!.step + 1)
    expect(snaps.at(-1)!.history.nll.at(-1)!).toBeLessThan(snaps[0].history.nll[0])
  })
})
