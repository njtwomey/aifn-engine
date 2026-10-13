/**
 * `randomForestRegressor`: exactly scikit-learn's `DecisionTreeRegressor` in every tree without bootstrap (every
 * feature searched), and in law scikit-learn's `RandomForestRegressor` with bootstrap (the mean and the spread over
 * trees at query points, against a 2000-tree forest in `fixtures/gen/optimisation/bayesian.py`); the moments are the
 * trees' mean and population variance, plus the leaves' variances with `total`; reproducible and extendable.
 */
import { describe, expect, it } from 'vitest'
import { randomForestRegressor } from 'aifn-methods/learning/trees-and-ensembles/bagging'
import { applyTree } from 'aifn-methods/learning/trees-and-ensembles'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { ShapeError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../../fixtures'
import { close, fx, sameTree, X3, XQ } from '../../shared'

type ForestFx = { forest: { x: number[][]; y: number[]; xq: number[][]; mean: number[]; variance: number[] } }
const RF = fixture<ForestFx>('optimisation/bayesian').forest

describe('randomForestRegressor', () => {
  it('without bootstrap, every tree is scikit-learn’s regression tree and the spread is 0', () => {
    const m = randomForestRegressor({ trees: 3, bootstrap: false, maxDepth: 3 }).fit(dataset(X3, tensor(fx.yreg)))
    for (const tree of m.trees) sameTree(tree, fx.rtree)
    close(m.decide(XQ), fx.rtree_predict)
    const p = toFlat(m.treePredictions(XQ))
    expect(m.treePredictions(XQ).shape).toEqual([3, XQ.shape[0]])
    for (let b = 0; b < 3; b++) close(Array.from(p.slice(b * 12, b * 12 + 12)), fx.rtree_predict)
    expect(Array.from(toFlat(m.moments(XQ).variance))).toEqual(new Array(12).fill(0))
    expect(Number.isNaN(m.outOfBag.mse)).toBe(true)
  })

  it('with bootstrap, matches scikit-learn’s forest in law: the mean and the spread over trees', () => {
    const m = randomForestRegressor({ trees: 400 }).fit(dataset(tensor(RF.x), tensor(RF.y)), { stream: stream(4) })
    const { mean, variance } = m.moments(tensor(RF.xq))
    toFlat(mean).forEach((v, i) => expect(Math.abs(v - RF.mean[i]), `point ${i}`).toBeLessThan(0.03))
    toFlat(variance).forEach((v, i) => expect(Math.abs(v / RF.variance[i] - 1), `point ${i}`).toBeLessThan(0.3))
    expect(m.outOfBag.r2).toBeGreaterThan(0.7)
    close([toFlat(m.featureImportances).reduce((a, b) => a + b, 0)], [1])
  })

  it('moments are the trees’ mean and population variance, plus the leaf variances with total', () => {
    const m = randomForestRegressor({ trees: 7, maxDepth: 2 }).fit(dataset(tensor(RF.x), tensor(RF.y)), {
      stream: stream(1),
    })
    const q = tensor(RF.xq)
    const p = toFlat(m.treePredictions(q))
    const { mean, variance } = m.moments(q)
    const total = toFlat(m.moments(q, { total: true }).variance)
    for (let i = 0; i < 4; i++) {
      const col = [0, 1, 2, 3, 4, 5, 6].map((b) => p[b * 4 + i])
      const mu = col.reduce((a, b) => a + b) / 7
      expect(toFlat(mean)[i]).toBeCloseTo(mu, 12)
      expect(toFlat(variance)[i]).toBeCloseTo(col.reduce((a, b) => a + (b - mu) ** 2, 0) / 7, 12)
      const leafVar = m.trees.reduce((a, t) => a + t.nodes[toFlat(applyTree(t, q))[i]].impurity, 0) / 7
      expect(total[i]).toBeCloseTo(toFlat(variance)[i] + leafVar, 12)
    }
    expect(toFlat(m.forward(q))).toEqual(toFlat(mean))
  })

  it('is reproducible, and a forest of more trees extends one of fewer', () => {
    const data = dataset(tensor(RF.x), tensor(RF.y))
    const a = randomForestRegressor({ trees: 5 }).fit(data, { stream: stream(9) })
    const b = randomForestRegressor({ trees: 8 }).fit(data, { stream: stream(9) })
    const q = tensor(RF.xq)
    expect(Array.from(toFlat(b.treePredictions(q)).slice(0, 20))).toEqual(Array.from(toFlat(a.treePredictions(q))))
    expect(
      toFlat(
        randomForestRegressor({ trees: 5 })
          .fit(data, { stream: stream(9) })
          .decide(q),
      ),
    ).toEqual(toFlat(a.decide(q)))
  })

  it('checks its inputs', () => {
    expect(() => randomForestRegressor().fit(dataset(tensor(RF.x), tensor([1, 2])))).toThrow(ShapeError)
    const m = randomForestRegressor({ trees: 2 }).fit(dataset(tensor(RF.x), tensor(RF.y)))
    expect(() => m.decide(tensor([[1, 2, 3]]))).toThrow(ShapeError)
  })
})
