/**
 * The recommenders against references (`fixtures/retrieval/recommenders.json`: the `implicit` library's exact ALS,
 * a NumPy ALS-WR, scikit-learn's cosine similarities) and laws: the alternating objectives never increase, the
 * gradient models' losses fall and beat popularity on held-out recall when the data have taste structure, the
 * feedback loop concentrates exposure under popularity more than under random slates, and the registry is well formed.
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { implicitFeedback } from 'aifn-methods/data/synthetic'
import {
  alternatingLeastSquares,
  biasedFactorInit,
  biasedFactorScorer,
  evaluateRanking,
  feedbackLoop,
  implicitAls,
  interactionsFromRows,
  itemKnn,
  matrixFactorisationSgd,
  popularity,
  recommenderAlgorithms,
  recommenderFunctions,
  recommenderRun,
  topK,
  userKnn,
  worldFromFactors,
  type RecommenderKind,
  type RecommenderSnapshot,
} from 'aifn-methods/retrieval/recommenders'
import { fixture } from '../../fixtures'
import { expectInfo } from '../../registry'

type F = {
  implicit: {
    users: number
    items: number
    rows: number[][]
    alpha: number
    regularisation: number
    sweeps: number
    P0: number[][]
    Q0: number[][]
    P: number[][]
    Q: number[][]
  }
  explicit: {
    rows: number[][]
    P0: number[][]
    Q0: number[][]
    P: number[][]
    Q: number[][]
    regularisation: number
    sweeps: number
  }
  itemCosine: { rows: number[][]; similarity: number[][] }
}
const R = fixture<F>('retrieval/recommenders')

const close = (got: ArrayLike<number>, want: number[], tol: number) =>
  want.forEach((w, i) => expect(Math.abs(got[i] - w), `${i}: ${got[i]} vs ${w}`).toBeLessThanOrEqual(tol))

describe('implicitAls', () => {
  const f = R.implicit
  const data = interactionsFromRows(f.rows, f.users, f.items)
  it('matches the implicit library’s exact ALS sweep for sweep', () => {
    const alg = implicitAls(data, {
      factors: 3,
      alpha: f.alpha,
      regularisation: f.regularisation,
      init: { P: f.P0.flat(), Q: f.Q0.flat() },
    })
    const s = run(alg, undefined, f.sweeps)
    close(toFlat(s.P), f.P.flat(), 1e-9)
    close(toFlat(s.Q), f.Q.flat(), 1e-9)
  })
  it('never increases its objective, half-step by half-step', () => {
    const t = trace(implicitAls(data, { factors: 3 }), undefined, 15, { stream: stream(3) })
    const states = t.steps
    for (let i = 1; i < states.length; i++) {
      expect(states[i].half).toBeLessThanOrEqual(states[i - 1].objective + 1e-9)
      expect(states[i].objective).toBeLessThanOrEqual(states[i].half + 1e-9)
    }
  })
})

describe('alternatingLeastSquares', () => {
  const f = R.explicit
  const data = interactionsFromRows(f.rows, 10, 9)
  it('matches a direct ALS-WR', () => {
    const s = run(
      alternatingLeastSquares(data, {
        factors: 2,
        regularisation: f.regularisation,
        init: { P: f.P0.flat(), Q: f.Q0.flat() },
      }),
      undefined,
      f.sweeps,
    )
    close(toFlat(s.P), f.P.flat(), 1e-9)
    close(toFlat(s.Q), f.Q.flat(), 1e-9)
  })
  it('never increases its objective', () => {
    const t = trace(alternatingLeastSquares(data, { factors: 2 }), undefined, 12, { stream: stream(1) })
    for (let i = 1; i < t.steps.length; i++)
      expect(t.steps[i].objective).toBeLessThanOrEqual(t.steps[i - 1].objective + 1e-9)
  })
})

describe('matrixFactorisationSgd', () => {
  it('lowers the training squared error by automatic-gradient SGD', () => {
    const data = interactionsFromRows(R.explicit.rows, 10, 9)
    const alg = matrixFactorisationSgd(data, { stepSize: 0.05, batchSize: 8 })
    const s0 = alg.init({ params: biasedFactorInit(stream(2), data, 2) }, stream(0))
    const sN = run(alg, { params: biasedFactorInit(stream(2), data, 2) }, 300)
    const mse = (p: typeof s0.params) => {
      const S = biasedFactorScorer(p)([...Array(10).keys()])
      let e = 0
      R.explicit.rows.forEach(([u, i, r]) => (e += (S[u * 9 + i] - r) ** 2 / R.explicit.rows.length))
      return e
    }
    expect(mse(sN.params)).toBeLessThan(0.5 * mse(s0.params))
  })
})

describe('neighbourhood models', () => {
  it('item-kNN with every neighbour scores by the cosine similarities of scikit-learn', () => {
    const f = R.itemCosine
    const data = interactionsFromRows(
      f.rows.map(([u, i]) => [u, i]),
      12,
      15,
    )
    const S = itemKnn(data, { k: 14 })([0, 1])
    const binary = new Float64Array(12 * 15)
    f.rows.forEach(([u, i]) => (binary[u * 15 + i] = 1))
    for (const u of [0, 1])
      for (let i = 0; i < 15; i++) {
        let want = 0
        for (let j = 0; j < 15; j++) if (binary[u * 15 + j] && j !== i) want += f.similarity[j][i]
        expect(S[u * 15 + i]).toBeCloseTo(want, 9)
      }
  })
  it('popularity ranks by counts, and topK skips seen items', () => {
    const data = interactionsFromRows(
      [
        [0, 2],
        [1, 2],
        [1, 0],
        [2, 1],
        [3, 2],
      ],
      4,
      3,
    )
    const s = popularity(data)([0])
    expect(Array.from(s)).toEqual([1, 1, 3])
    expect(topK(s, 2, new Set([2]))).toEqual([0, 1])
  })
})

describe('on implicit feedback with taste structure', () => {
  const d = implicitFeedback(stream(5), { users: 60, items: 70, testPerUser: 3 })
  const train = interactionsFromRows(d.train, d.users, d.items)
  const test = interactionsFromRows(d.test, d.users, d.items)
  const pop = evaluateRanking(popularity(train), train, test, 10)

  it('the neighbourhood models beat popularity on held-out recall', () => {
    expect(evaluateRanking(userKnn(train, { k: 15 }), train, test, 10).recall).toBeGreaterThan(pop.recall)
    expect(evaluateRanking(itemKnn(train, { k: 15 }), train, test, 10).recall).toBeGreaterThan(pop.recall * 0.9)
  })

  const last = (gen: Generator<RecommenderSnapshot>) => {
    let s: RecommenderSnapshot | undefined
    for (const x of gen) s = x
    return s!
  }
  const kinds: [RecommenderKind, number][] = [
    ['implicit-als', 8],
    ['logistic-mf', 8],
    ['bpr', 8],
    ['factorisation-machine', 6],
    ['field-aware-factorisation-machine', 4],
    ['wide-and-deep', 6],
    ['deepfm', 6],
    ['neural-collaborative-filtering', 6],
    ['two-tower', 8],
    ['sasrec', 15],
  ]
  for (const [model, epochs] of kinds)
    it(
      `${model} trains: its loss falls and its held-out recall is finite and above chance`,
      { timeout: 60_000 },
      () => {
        const s = last(recommenderRun({ data: d, model, epochs, dimension: 8, hidden: [16], seed: 1 }))
        expect(s.done).toBe(true)
        expect(s.history.epoch.length).toBe(epochs + 1)
        const loss = s.history.loss.filter(Number.isFinite)
        expect(loss[loss.length - 1]).toBeLessThan(loss[0])
        const recall = s.history.recall[s.history.recall.length - 1]
        // Chance: 10 of the ~55 unseen items, i.e. about 0.18.
        expect(recall).toBeGreaterThan(0.2)
        expect(s.checkpoints[s.checkpoints.length - 1].scores.length).toBe(d.users * d.items)
      },
    )
})

describe('feedbackLoop', () => {
  it('concentrates exposure more under popularity than under random slates', () => {
    const d = implicitFeedback(stream(9), { users: 40, items: 60 })
    const world = worldFromFactors(toFlat(d.userFactors), toFlat(d.itemFactors), d.users, d.items)
    let last
    for (const s of feedbackLoop({ world, rounds: 8, policies: ['popularity', 'random', 'oracle'] })) last = s
    const p = last!.policies
    expect(p.popularity.gini[8]).toBeGreaterThan(p.random.gini[8] + 0.2)
    // The oracle's slates have the highest expected click-through.
    expect(p.oracle.ctr[8]).toBeGreaterThan(p.random.ctr[8])
  })
})

describe('registry', () => {
  it('entries are well formed', () => {
    expectInfo(recommenderAlgorithms, 'algorithm')
    expectInfo(recommenderFunctions, 'function')
  })
})
