/**
 * The regularised linear paths on the diabetes data against scikit-learn (the `diabetes` cases of fixture
 * `learning/linear`): `elasticNet` against `ElasticNet` and `Lasso`, `elasticNetPath` against `enet_path`, the ridge
 * degrees of freedom and `ridgePath` against `Ridge`, `leastAngleRegression` against `lars_path`, and
 * `linearSgdSteps` against the spread of `SGDRegressor` over twenty seeds. As in the field notes, the features are
 * standardised with `standardScaler` and the target centred for the paths.
 */
import { describe, expect, it } from 'vitest'
import { diabetes } from 'aifn-methods/data/real'
import { standardScaler } from 'aifn-methods/learning/preprocessing'
import {
  effectiveDof,
  elasticNet,
  elasticNetPath,
  leastAngleRegression,
  linearFunctions,
  linearRegression,
  linearSgdSteps,
  ridgePath,
} from 'aifn-methods/learning/linear'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { dataset } from 'aifn-compute/learning/estimators'
import { fixture } from '../../fixtures'

type Fixture = {
  data: { mean: number[]; sd: number[]; target_mean: number; first: number[]; first_target: number }
  elastic_net: {
    alpha: number
    l1_ratio: number
    standardised: boolean
    coef: number[]
    intercept: number
    objective: number
  }[]
  paths: { l1_ratio: number; alphas: number[]; coef: number[][]; nonzero: number[] }[]
  ridge: { lambdas: number[]; coef: number[][]; intercept: number[]; dof: number[] }
  lars: Record<
    'lar' | 'lasso',
    {
      alphas: number[]
      active: number[]
      coef: number[][]
      l1_norm: number[]
      events: { knot: number; feature: number; kind: 'enter' | 'drop' }[]
    }
  >
  sgd: {
    penalty: 'l2' | 'l1' | 'elasticnet'
    alpha: number
    l1_ratio: number
    epochs: number
    mean: number[]
    sd: number[]
    intercept_mean: number
    intercept_sd: number
    zeros: number
    optimum: number[]
  }[]
}
const fx = fixture<{ diabetes: Fixture }>('learning/linear').diabetes

const data = diabetes()
const x = standardScaler().fit({ x: data.x }).transform(data.x)
const target = toFlat(data.y!)
const mean = target.reduce((a, b) => a + b, 0) / target.length
const y = fromData(Float64Array.from(target, (v) => v - mean))

const expectClose = (actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number, what: string): void => {
  expect(actual.length, what).toBe(expected.length)
  for (let i = 0; i < expected.length; i++)
    expect(Math.abs(actual[i] - expected[i]), `${what}[${i}]: ${actual[i]} vs ${expected[i]}`).toBeLessThan(tol)
}
const rows = (t: Tensor): number[][] => toRows(t) as number[][]

describe('diabetes data', () => {
  it("is scikit-learn's raw copy, column for column", () => {
    expect(data.x.shape).toEqual([442, 10])
    expect(data.meta.featureNames).toEqual(['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])
    expect(data.meta.sha256).toBe('4733febee697862c22139cdac87478a300ce0d101593deb07ed6c0f3328a99cd')
    expectClose(rows(data.x)[0], fx.data.first, 1e-12, 'first row')
    expect(target[0]).toBe(fx.data.first_target)
    expect(mean).toBeCloseTo(fx.data.target_mean, 10)
    const cols = rows(data.x)
    const colMean = fx.data.mean.map((_, j) => cols.reduce((a, r) => a + r[j], 0) / 442)
    expectClose(colMean, fx.data.mean, 1e-9, 'column means')
  })
})

describe('elasticNet', () => {
  it('matches ElasticNet and Lasso to 1e-6, with exact zeros', () => {
    for (const c of fx.elastic_net) {
      const design = c.standardised ? x : data.x
      const m = elasticNet({ alpha: c.alpha, l1Ratio: c.l1_ratio }).fit(dataset(design, data.y!))
      const what = `alpha ${c.alpha}, l1Ratio ${c.l1_ratio}${c.standardised ? '' : ', raw'}`
      expect(m.converged, what).toBe(true)
      expectClose(toFlat(m.weights), c.coef, 1e-6, `${what}: weights`)
      expect(Math.abs(m.intercept - c.intercept), `${what}: intercept`).toBeLessThan(1e-6)
      expect(Math.abs(m.objective - c.objective) / c.objective, `${what}: objective`).toBeLessThan(1e-10)
      const w = toFlat(m.weights)
      // Exact zeros where scikit-learn has them.
      c.coef.forEach((v, j) => expect(w[j] === 0, `${what}: zero at ${j}`).toBe(v === 0))
      expect(m.nonzero).toBe(c.coef.filter((v) => v !== 0).length)
      expectClose(
        toFlat(m.decide(x)).slice(0, 1),
        [
          m.intercept +
            toFlat(x)
              .slice(0, 10)
              .reduce((a, v, j) => a + v * w[j], 0),
        ],
        1e-9,
        'decide',
      )
    }
  })

  it('a warm start from the solution converges at once', () => {
    const first = elasticNet({ alpha: 1 }).fit(dataset(x, y))
    const again = elasticNet({ alpha: 1, warmStart: first.weights }).fit(dataset(x, y))
    expect(again.steps).toBe(1)
    expectClose(toFlat(again.weights), toFlat(first.weights), 1e-9, 'weights')
  })

  it('rejects a penalty that is not positive and a mixing outside [0, 1]', () => {
    expect(() => elasticNet({ alpha: 0 })).toThrow(/alpha/)
    expect(() => elasticNet({ alpha: 1, l1Ratio: 1.5 })).toThrow(/l1Ratio/)
    expect(() => elasticNet({ alpha: 1, warmStart: tensor([1, 2]) }).fit(dataset(x, y))).toThrow(/warmStart/)
  })
})

describe('elasticNetPath', () => {
  it('matches enet_path for l1Ratio 0.1, 0.5, 0.9 and 1 on its default grid', () => {
    for (const p of fx.paths) {
      const path = elasticNetPath(x, y, { l1Ratio: p.l1_ratio })
      expect(path.converged).toBe(true)
      expectClose(path.alphas, p.alphas, 1e-9 * p.alphas[0], `l1Ratio ${p.l1_ratio}: alphas`)
      expect(path.coef.shape).toEqual([10, 81])
      rows(path.coef).forEach((row, j) => expectClose(row, p.coef[j], 1e-6, `l1Ratio ${p.l1_ratio}: feature ${j}`))
      expect(path.nonzero).toEqual(p.nonzero)
      for (const b of path.intercept) expect(Math.abs(b)).toBeLessThan(1e-9)
    }
  })

  it('agrees with a single fit at each penalty, and takes explicit penalties in any order', () => {
    const path = elasticNetPath(x, y, { l1Ratio: 0.5, alphas: [0.01, 1, 0.1] })
    expect(path.alphas).toEqual([1, 0.1, 0.01])
    path.alphas.forEach((alpha, t) => {
      const w = toFlat(elasticNet({ alpha, l1Ratio: 0.5 }).fit(dataset(x, y)).weights)
      expectClose(
        rows(path.coef).map((r) => r[t]),
        w,
        1e-6,
        `alpha ${alpha}`,
      )
    })
    expect(() => elasticNetPath(x, y, { l1Ratio: 0 })).toThrow(/alphas/)
  })
})

describe('ridge', () => {
  it('linearRegression reports the effective degrees of freedom', () => {
    fx.ridge.lambdas.forEach((l2, t) => {
      const m = linearRegression({ l2 }).fit(dataset(x, data.y!))
      expect(Math.abs(m.dof - fx.ridge.dof[t]), `lambda ${l2}`).toBeLessThan(1e-9)
      expectClose(
        toFlat(m.weights),
        fx.ridge.coef.map((r) => r[t]),
        1e-6,
        `lambda ${l2}: weights`,
      )
    })
    const ols = linearRegression().fit(dataset(x, y))
    expect(ols.dof).toBe(10)
    expect(effectiveDof(toFlat(ols.singularValues), 0)).toBe(10)
  })

  it('ridgePath matches Ridge along 71 penalties from 1e5 to 1e-2', () => {
    const path = ridgePath(x, data.y!, fx.ridge.lambdas)
    rows(path.coef).forEach((row, j) => expectClose(row, fx.ridge.coef[j], 1e-6, `feature ${j}`))
    expectClose(path.intercept, fx.ridge.intercept, 1e-6, 'intercepts')
    expectClose(path.dof, fx.ridge.dof, 1e-9, 'dof')
    expect(() => ridgePath(x, y, [-1])).toThrow(/non-negative/)
  })
})

describe('leastAngleRegression', () => {
  for (const method of ['lar', 'lasso'] as const) {
    it(`matches lars_path(method='${method}'): knots, weights and the order of events`, () => {
      const expected = fx.lars[method]
      const path = leastAngleRegression(x, y, { method })
      expectClose(path.alphas, expected.alphas, 1e-9, 'alphas')
      expect(path.coef.shape).toEqual([10, expected.alphas.length])
      rows(path.coef).forEach((row, j) => expectClose(row, expected.coef[j], 1e-6, `feature ${j}`))
      expectClose(path.l1Norm, expected.l1_norm, 1e-6, 'l1 norms')
      expect(path.events).toEqual(expected.events)
      expect(path.alphas.at(-1)).toBe(0)
    })
  }

  it('the lasso knots lie on the coordinate-descent lasso path', () => {
    const path = leastAngleRegression(x, y, { method: 'lasso' })
    const knots = path.alphas.slice(1, -1)
    const cd = elasticNetPath(x, y, { alphas: knots })
    rows(cd.coef).forEach((row, j) => expectClose(row, rows(path.coef)[j].slice(1, -1), 1e-6, `feature ${j}`))
  })
})

describe('linearSgdSteps', () => {
  for (const c of fx.sgd) {
    it(`${c.penalty}, alpha ${c.alpha}: within SGDRegressor's run-to-run spread after ${c.epochs} epochs`, () => {
      const finals = [1, 2, 3].map((seed) =>
        run(linearSgdSteps({ x, y, penalty: c.penalty, alpha: c.alpha, l1Ratio: c.l1_ratio }), {}, c.epochs, {
          stream: child(stream(seed), 'sgd'),
        }),
      )
      for (const s of finals) {
        expect(s.epoch).toBe(c.epochs)
        const w = toFlat(s.weights)
        w.forEach((v, j) =>
          expect(Math.abs(v - c.mean[j]), `weight ${j}: ${v} vs ${c.mean[j]} ± ${c.sd[j]}`).toBeLessThan(
            5 * c.sd[j] + 0.05,
          ),
        )
        expect(Math.abs(s.intercept - c.intercept_mean)).toBeLessThan(5 * c.intercept_sd + 0.05)
        // The truncated penalty leaves exact zeros where SGDRegressor does.
        if (c.penalty === 'l1') expect(s.zeros).toBeGreaterThanOrEqual(c.zeros)
      }
      // And near the penalised optimum, on average over the runs.
      const avg = c.optimum.map((_, j) => finals.reduce((a, s) => a + toFlat(s.weights)[j], 0) / finals.length)
      expectClose(avg, c.optimum, 0.5, 'mean of the runs')
    })
  }

  it('the subgradient never reaches zero; the truncated penalty does', () => {
    const common = { x, y, penalty: 'l1' as const, alpha: 1 }
    const at = (l1Handling: 'truncated' | 'subgradient') =>
      run(linearSgdSteps({ ...common, l1Handling }), {}, 30, { stream: stream(4) })
    expect(at('truncated').zeros).toBeGreaterThan(0)
    expect(at('subgradient').zeros).toBe(0)
  })

  it('records the objective per epoch, and is deterministic in its stream', () => {
    const alg = linearSgdSteps({ x, y, penalty: 'elasticnet', alpha: 0.1, schedule: 'constant', eta0: 0.001 })
    const a = run(alg, {}, 5, { stream: stream(7) })
    const b = run(alg, {}, 5, { stream: stream(7) })
    expect(toFlat(a.weights)).toEqual(toFlat(b.weights))
    expect(a.samples).toBe(5 * 442)
    expect(a.eta).toBe(0.001)
    const start = alg.init({}, stream(0))
    expect(start.zeros).toBe(10)
    expect(a.objective).toBeLessThan(start.objective)
  })
})

describe('registry', () => {
  it('lists the paths, selections, evidence and hypergradient as functions', () => {
    expect(Object.keys(linearFunctions).sort()).toEqual([
      'bayesianLinearEvidence',
      'bestSubset',
      'elasticNetPath',
      'evidenceMaximisation',
      'forwardStagewise',
      'groupLassoPath',
      'leastAngleRegression',
      'ridgeHypergradient',
      'ridgePath',
      'ruleFitPath',
      'stepwise',
    ])
  })
})
