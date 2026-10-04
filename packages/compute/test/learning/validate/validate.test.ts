/**
 * aifn-compute/learning/validate: splitters (scikit-learn 1.9 split indices inlined where comparable), cross-validation over
 * registered metrics (with the capability type check), and grid, random and nested searches over a `Space`. The
 * estimators are small hand-written ones (ridge regression, a nearest-centroid classifier).
 */
import { describe, expect, it } from 'vitest'
import { int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { add, fromData, matmul, mul, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  bernoulliPredictive,
  type Decides,
  type Estimator,
  type Predicts,
  type Supervised,
} from 'aifn-compute/learning/estimators'
import { accuracy, logLoss, meanSquaredError } from 'aifn-compute/learning/metrics'
import { solve } from 'aifn-compute/numerics/linalg'
import {
  assignment,
  crossValidate,
  expandingWindow,
  gridSearch,
  groupKFold,
  kFold,
  leaveOneOut,
  nested,
  randomSearch,
  repeated,
  rollingOrigin,
  shuffleSplit,
  stratifiedKFold,
  TEST,
  TRAIN,
  UNUSED,
  type Split,
} from 'aifn-compute/learning/validate'

/** Ridge regression with an unpenalised intercept, by the normal equations. */
type Ridge = Decides<Tensor, Tensor> & { l2: number; weights: number[] }
const ridge = (l2: number): Estimator<Supervised<Tensor, Tensor>, Ridge> => ({
  name: 'ridge',
  params: { l2 },
  fit({ x, y }) {
    const rows = toRows(x).map((r) => [...r, 1])
    const d = rows[0].length
    const a = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => rows.reduce((s, r) => s + r[i] * r[j], 0) + (i === j && i < d - 1 ? l2 : 0)),
    )
    const yv = toFlat(y)
    const b = Array.from({ length: d }, (_, i) => rows.reduce((s, r, k) => s + r[i] * yv[k], 0))
    const weights = Array.from(toFlat(solve(tensor(a), tensor(b)) as Tensor))
    const decide = (z: Tensor) =>
      tensor(toRows(z).map((r) => r.reduce((s, v, j) => s + v * weights[j], weights[d - 1])))
    return { l2, weights, decide }
  },
})

/** Nearest centroid on two classes; P(y = 1 | x) = σ(‖x − μ₀‖² − ‖x − μ₁‖²) / 2). */
type Centroid = Decides<Tensor, Tensor> & Predicts<Tensor, ReturnType<typeof bernoulliPredictive>>
const centroid = (): Estimator<Supervised<Tensor, Tensor>, Centroid> => ({
  name: 'centroid',
  fit({ x, y }) {
    const rows = toRows(x)
    const labels = toFlat(y)
    const centre = (c: number) => {
      const members = rows.filter((_, i) => labels[i] === c)
      return members[0].map((_, j) => members.reduce((s, r) => s + r[j], 0) / members.length)
    }
    const [m0, m1] = [centre(0), centre(1)]
    const d2 = (r: number[], m: number[]) => r.reduce((s, v, j) => s + (v - m[j]) ** 2, 0)
    const p = (z: Tensor) =>
      fromData(
        Float64Array.from(toRows(z), (r) => 1 / (1 + Math.exp(-(d2(r, m0) - d2(r, m1)) / 2))),
        [z.shape[0]],
      )
    return {
      decide: (z) =>
        fromData(
          Int32Array.from(toFlat(p(z)), (q) => (q >= 0.5 ? 1 : 0)),
          [z.shape[0]],
        ),
      predictive: (z) => bernoulliPredictive(p(z)),
    }
  },
})

const tests = (splits: Split[]) => splits.map((s) => toFlat(s.test))
const trains = (splits: Split[]) => splits.map((s) => toFlat(s.train))

/** Every row is tested exactly once and never trained on in its own fold. */
function isPartition(splits: Split[], n: number) {
  const seen = new Array(n).fill(0)
  for (const s of splits) {
    for (const i of toFlat(s.test)) seen[i]++
    const test = new Set(toFlat(s.test))
    for (const i of toFlat(s.train)) expect(test.has(i)).toBe(false)
    expect(toFlat(s.train).length + toFlat(s.test).length).toBe(n)
  }
  expect(seen).toEqual(new Array(n).fill(1))
}

describe('splitters', () => {
  it('k-fold matches scikit-learn and partitions; shuffled is a deterministic partition', () => {
    expect(tests(kFold({ k: 3 }).split({ n: 10 }))).toEqual([
      [0, 1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ])
    const shuffled = kFold({ k: 4, shuffle: true })
    isPartition(shuffled.split({ n: 23 }, stream('k')), 23)
    expect(tests(shuffled.split({ n: 23 }, stream('k')))).toEqual(tests(shuffled.split({ n: 23 }, stream('k'))))
    expect(() => shuffled.split({ n: 23 })).toThrow(/stream/)
    isPartition(leaveOneOut().split({ n: 5 }), 5)
  })

  it('stratified k-fold matches scikit-learn and keeps class proportions', () => {
    const y = tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1])
    expect(tests(stratifiedKFold({ k: 3 }).split({ n: 12, y }))).toEqual([
      [0, 1, 3, 7],
      [2, 4, 5, 8],
      [6, 9, 10, 11],
    ])
    const labels = Array.from({ length: 60 }, (_, i) => (i % 5 === 0 ? 'rare' : 'common'))
    const splits = stratifiedKFold({ k: 4, shuffle: true }).split({ n: 60, y: labels }, stream('strat'))
    isPartition(splits, 60)
    for (const s of splits) expect(toFlat(s.test).filter((i) => labels[i] === 'rare').length).toBe(3)
  })

  it('grouped k-fold matches scikit-learn and never splits a group', () => {
    const groups = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4]
    const splits = groupKFold({ k: 3 }).split({ n: 15, groups })
    expect(tests(splits)).toEqual([
      [9, 10, 11, 12, 13],
      [0, 1, 2, 3, 14],
      [4, 5, 6, 7, 8],
    ])
    for (const s of splits) {
      const testGroups = new Set(toFlat(s.test).map((i) => groups[i]))
      for (const i of toFlat(s.train)) expect(testGroups.has(groups[i])).toBe(false)
    }
  })

  it('time-series splits match scikit-learn and never train on the future', () => {
    const splits = expandingWindow({ splits: 3, gap: 1 }).split({ n: 10 })
    expect(trains(splits)).toEqual([
      [0, 1, 2],
      [0, 1, 2, 3, 4],
      [0, 1, 2, 3, 4, 5, 6],
    ])
    expect(tests(splits)).toEqual([
      [4, 5],
      [6, 7],
      [8, 9],
    ])
    const rolling = rollingOrigin({ window: 4, horizon: 2, gap: 1 }).split({ n: 12 })
    for (const s of rolling) {
      expect(toFlat(s.train).length).toBe(4)
      expect(Math.max(...toFlat(s.train))).toBeLessThan(Math.min(...toFlat(s.test)) - 1)
    }
    expect(rolling.length).toBe(3)
  })

  it('shuffle-split and repeated k-fold', () => {
    const ss = shuffleSplit({ splits: 4, testSize: 0.25, trainSize: 0.5 }).split({ n: 20 }, stream('ss'))
    for (const s of ss) expect([toFlat(s.test).length, toFlat(s.train).length]).toEqual([5, 10])
    const rep = repeated(kFold({ k: 5, shuffle: true }), 3).split({ n: 20 }, stream('rep'))
    expect(rep.length).toBe(15)
    isPartition(rep.slice(5, 10), 20)
    expect(tests(rep.slice(0, 5))).not.toEqual(tests(rep.slice(5, 10)))
  })

  it('assignment matrix', () => {
    expect([TEST, TRAIN, UNUSED]).toEqual([1, 0, -1])
    const a = assignment(shuffleSplit({ splits: 2, testSize: 1, trainSize: 2 }).split({ n: 4 }, stream('a')), 4)
    expect(a.shape).toEqual([2, 4])
    for (const row of toRows(a)) expect(row.slice().sort()).toEqual([-1, 0, 0, 1])
  })
})

describe('crossValidate', () => {
  const s = stream('cv')
  const x = normals(child(s, 'x'), [80, 3])
  const y = tensor(toFlat(add(matmul(x, tensor([1, -2, 0])), mul(0.5, normals(child(s, 'e'), [80]))) as Tensor))
  const labels = tensor(toFlat(y).map((v) => (v > 0 ? 1 : 0)))
  const data = { kind: 'dataset', x, y: labels } as const

  it('keeps folds, models, predictions, metrics and out-of-fold decisions', () => {
    const cv = crossValidate(centroid(), data, kFold({ k: 4 }), [accuracy, logLoss])
    expect(cv.splitter).toBe(kFold({ k: 4 }).name)
    expect(cv.folds.length).toBe(4)
    expect(cv.assignment.shape).toEqual([4, 80])
    expect(cv.scores.accuracy.shape).toEqual([4])
    expect(cv.mean.accuracy).toBeGreaterThan(0.7)
    expect(cv.mean.accuracy).toBeCloseTo(toFlat(cv.scores.accuracy).reduce((a, b) => a + b, 0) / 4, 12)
    expect(cv.outOfFold.decide?.shape).toEqual([80])
    expect(cv.directions).toEqual({ accuracy: 'higher', logLoss: 'lower' })
    // Each fold's metric is the metric of its model on its test rows.
    const f = cv.folds[1]
    const test = toFlat(f.test)
    const xt = tensor(test.map((i) => toRows(x)[i]))
    const yt = test.map((i) => toFlat(labels)[i])
    expect(f.metrics.accuracy).toBeCloseTo(accuracy(yt, toFlat(f.model.decide(xt))), 12)
    // Out-of-fold decisions are those of the model that did not train on the row.
    const oof = toFlat(cv.outOfFold.decide!)
    expect(test.map((i) => oof[i])).toEqual(Array.from(toFlat(f.model.decide(xt))))
    const trained = crossValidate(centroid(), data, kFold({ k: 4 }), [accuracy], { trainMetrics: true })
    expect(trained.folds[0].trainMetrics?.accuracy).toBeGreaterThan(0.7)
  })

  it('rejects metrics the model cannot serve (types)', () => {
    const decider: Estimator<Supervised<Tensor, Tensor>, Decides<Tensor, Tensor>> = {
      name: 'always-one',
      fit: () => ({ decide: (z) => tensor(new Array(z.shape[0]).fill(1)) }),
    }
    // @ts-expect-error: log loss needs a predictive; the fitted model only decides.
    expect(() => crossValidate(decider, data, kFold({ k: 2 }), [logLoss])).toThrow(/predictive/)
    expect(crossValidate(decider, data, kFold({ k: 2 }), [accuracy]).folds.length).toBe(2)
  })
})

describe('search over a Space, and nested cross-validation', () => {
  const s = stream('search')
  const x = normals(child(s, 'x'), [40, 8])
  const y = tensor(toFlat(add(matmul(x, tensor([1, 0.5, 0, 0, 0, 0, 0, 0])), normals(child(s, 'e'), [40])) as Tensor))
  const data = { kind: 'dataset', x, y } as const

  it('grid search keeps the full table, ranks by the metric and refits the best', () => {
    const search = gridSearch((p) => ridge(p.l2), space({ l2: oneOf([0, 1, 10, 100]) }), {
      metrics: [meanSquaredError],
    })
    const r = search.run(data, kFold({ k: 5 }))
    expect(r.rows.length).toBe(4)
    expect(r.scores.shape).toEqual([4, 5])
    expect(r.metric).toBe('meanSquaredError')
    expect(r.direction).toBe('lower')
    expect(r.best.rank).toBe(1)
    expect(r.best.mean).toBe(Math.min(...r.rows.map((row) => row.mean)))
    expect(r.rows.map((row) => row.params.l2)).toEqual([0, 1, 10, 100])
    expect(r.model?.l2).toBe(r.best.params.l2)
    // A log-scale real gives `points` values from min to max.
    const logGrid = gridSearch((p) => ridge(p.l2), space({ l2: real(0.01, 100, { scale: 'log' }) }), {
      metrics: [meanSquaredError],
      points: 5,
    })
    const l2s = logGrid.candidates().map((p) => p.l2)
    ;[0.01, 0.1, 1, 10, 100].forEach((v, i) => expect(l2s[i]).toBeCloseTo(v, 12))
  })

  it('random search draws candidates from the stream, inside the space', () => {
    const sp = space({ l2: real(0.01, 100, { scale: 'log' }), pad: int(0, 3) })
    const search = randomSearch((p) => ridge(p.l2 + p.pad), sp, { metrics: [meanSquaredError], iterations: 5 })
    const a = search.run(data, kFold({ k: 4 }), stream('rs'))
    const b = search.run(data, kFold({ k: 4 }), stream('rs'))
    expect(a.rows.length).toBe(5)
    expect(a.rows.map((r) => r.params)).toEqual(b.rows.map((r) => r.params))
    expect(search.candidates(stream('x'))).not.toEqual(search.candidates(stream('y')))
    for (const r of a.rows) {
      expect(r.params.l2 >= 0.01 && r.params.l2 <= 100).toBe(true)
      expect(Number.isInteger(r.params.pad)).toBe(true)
    }
    expect(() => randomSearch((p) => ridge(p.l2), sp, { metrics: [] })).toThrow(/metric/)
  })

  it('nested cross-validation reports both levels and the optimism', () => {
    const search = gridSearch((p) => ridge(p.l2), space({ l2: oneOf([0, 3, 30]) }), { metrics: [meanSquaredError] })
    const result = nested(kFold({ k: 4 }), kFold({ k: 3 }), search, data)
    expect(result.perFold.length).toBe(4)
    expect(result.outer.folds[0].model.search.rows.length).toBe(3)
    expect(result.unnested.rows.length).toBe(3)
    expect(Number.isFinite(result.optimism)).toBe(true)
    expect(result.optimism).toBeCloseTo(result.nestedScore - result.unnestedScore, 12)
  })
})
