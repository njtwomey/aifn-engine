/**
 * The selection study: its fast penalised smoothers against the engine's generic path (`pspline`, and
 * `polynomialFeatures` then `linearRegression`, searched by `nested` on the real inputs), the laws it shows (the best
 * cross-validation score is optimistic beside nested cross-validation, the training error more so, the oracle is never
 * worse, the one-standard-error choice is never more flexible), determinism, and a run through a worker task tree.
 */
import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn-compute/foundation/random'
import { DomainError } from 'aifn-compute/foundation/errors'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { pipeline } from 'aifn-compute/learning/compose'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { oneStandardError } from 'aifn-compute/learning/selection'
import { gridSearch, kFold, nested } from 'aifn-compute/learning/validate'
import { pspline } from 'aifn-compute/numerics/interpolate'
import { linearRegression } from 'aifn-methods/learning/linear'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { drawTrainingSet } from 'aifn-methods/theory'
import { modelSelectionFunctions, selectionStudy } from 'aifn-methods/theory/model-selection'
import { evaluateTask } from './simulation-task'

const vector = (t: Tensor) => fromData(Float64Array.from(dense.data(t)), [t.shape[0]])
const column = (x: Float64Array) => fromData(x, [x.length, 1])

/** A P-spline on $[-1, 1]$ as an estimator of the real inputs, as the notes build it. */
const psplineLearner = (lambda: number, segments: number) => ({
  name: 'p-spline',
  fit: (d: { x: Tensor; y: Tensor }) => {
    const fit = pspline(vector(d.x), vector(d.y), { lambda, segments, range: [-1, 1] })
    return { decide: (x: Tensor) => fit.evaluate(vector(x)) }
  },
})

const ridgePolynomial = (lambda: number, degree: number) =>
  pipeline(polynomialFeatures({ degree, includeBias: false }), linearRegression({ l2: lambda }))

/**
 * Data set 0 of a study by the generic path: the same streams, the real inputs, the engine's estimators.
 *
 * @returns The minimum CV score, its index, the 1-SE index, the nested score, the true error and the test score.
 */
function generic(make: (lambda: number) => { fit: (d: never) => { decide: (x: Tensor) => Tensor } }, grid: number[]) {
  const s = stream(11)
  const k = kFold({ k: 4, shuffle: true })
  const fresh = drawTrainingSet(child(s, 'fresh'), 'sine', 400, 0.3)
  const rs = child(s, 'repeat', 0)
  const all = drawTrainingSet(child(rs, 'data'), 'sine', 40, 0.3)
  const dev = dataset(column(all.x.slice(0, 30)), fromData(all.y.slice(0, 30)))
  const search = gridSearch(({ lambda }) => make(lambda) as never, space({ lambda: oneOf(grid) }), {
    metrics: [meanSquaredError],
  })
  const result = nested(k, k, search, dev, { stream: child(rs, 'cv') })
  const rule = oneStandardError(result.unnested, { simplicity: (i) => -grid[i] })
  const model = result.unnested.model as { decide: (x: Tensor) => Tensor }
  return {
    bestIndex: rule.best,
    oneSeIndex: rule.chosen,
    bestScore: result.unnested.best.mean,
    nestedScore: result.nestedScore,
    trueError: meanSquaredError(fromData(fresh.y), model.decide(column(fresh.x))),
    testScore: meanSquaredError(fromData(all.y.slice(30)), model.decide(column(all.x.slice(30)))),
    rowLambdas: result.unnested.rows.map((r) => r.params.lambda),
  }
}

describe('selectionStudy', () => {
  it('equals the generic path for both learners', () => {
    const grid = [0.001, 0.1, 10, 1000]
    for (const [learner, make] of [
      ['pspline', (l: number) => psplineLearner(l, 15)],
      ['ridge-polynomial', (l: number) => ridgePolynomial(l, 10)],
    ] as const) {
      const g = generic(make as never, grid)
      const k = kFold({ k: 4, shuffle: true })
      const study = selectionStudy(stream(11), {
        rows: 40,
        test: 10,
        fresh: 400,
        grid,
        inner: k,
        outer: k,
        repeats: 1,
        learner,
      })
      const r = study.perRepeat
      expect(g.rowLambdas).toEqual(grid)
      expect(r.bestIndex[0]).toBe(g.bestIndex)
      expect(r.oneSeIndex[0]).toBe(g.oneSeIndex)
      expect(r.bestScore[0]).toBeCloseTo(g.bestScore, 9)
      expect(r.nestedScore[0]).toBeCloseTo(g.nestedScore, 9)
      expect(r.trueError[0]).toBeCloseTo(g.trueError, 9)
      expect(r.testScore[0]).toBeCloseTo(g.testScore, 9)
    }
  })

  it('shows the optimism of the best score and of the training error', () => {
    const k = kFold({ k: 5, shuffle: true })
    const grid = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 1e4]
    const r = selectionStudy(stream(13), { rows: 60, test: 20, fresh: 2000, grid, inner: k, outer: k, repeats: 100 })
    const p = r.perRepeat
    for (let i = 0; i < 100; i++) {
      expect(p.oracleTrueError[i]).toBeLessThanOrEqual(p.trueError[i])
      expect(p.oracleTrueError[i]).toBeLessThanOrEqual(p.oneSeTrueError[i])
      expect(p.oneSeIndex[i]).toBeGreaterThanOrEqual(p.bestIndex[i])
      expect(p.nestedSe[i]).toBeGreaterThan(0)
      expect(p.testSe[i]).toBeGreaterThan(0)
    }
    // The training error is optimistic, and the best inner score is lower than the nested estimate of the same search.
    expect(r.summary.trainingError.bias).toBeLessThan(0)
    const gap = Float64Array.from(p.nestedScore, (v, i) => v - p.bestScore[i])
    const mean = gap.reduce((a, b) => a + b, 0) / gap.length
    const sd = Math.sqrt(gap.reduce((a, b) => a + (b - mean) ** 2, 0) / (gap.length - 1))
    expect(mean / (sd / Math.sqrt(gap.length))).toBeGreaterThan(2)
    // The summary is the arithmetic of the per-repeat values.
    const m = (v: Float64Array) => v.reduce((a, b) => a + b, 0) / v.length
    expect(r.summary.testScore.mean).toBeCloseTo(m(p.testScore), 12)
    expect(r.summary.testScore.bias).toBeCloseTo(m(p.testScore) - m(p.trueError), 12)
    expect(r.summary.trueError).toEqual({ mean: r.summary.trueError.mean, bias: 0, sd: 0 })
    // The true error includes the noise variance.
    expect(r.summary.oracleTrueError.mean).toBeGreaterThan(0.09)
  })

  it('is fixed by its stream, runs in parts, and rejects bad options', () => {
    const k = kFold({ k: 3, shuffle: true })
    const o = { rows: 30, test: 6, fresh: 200, grid: [0.01, 1, 100], inner: k, outer: k }
    const whole = selectionStudy(stream(13), { ...o, repeats: 3 })
    const head = selectionStudy(stream(13), { ...o, repeats: 2 })
    const tail = selectionStudy(stream(13), { ...o, repeats: 1, first: 2 })
    expect(Array.from(whole.perRepeat.nestedScore)).toEqual([
      ...head.perRepeat.nestedScore,
      ...tail.perRepeat.nestedScore,
    ])
    expect(Array.from(whole.perRepeat.oracleIndex)).toEqual([
      ...head.perRepeat.oracleIndex,
      ...tail.perRepeat.oracleIndex,
    ])
    expect(() => selectionStudy(stream(0), { ...o, test: 29 })).toThrow(DomainError)
    expect(() => selectionStudy(stream(0), { ...o, grid: [] })).toThrow(DomainError)
    expect(() => selectionStudy(stream(0), { ...o, learner: 'lasso' as never })).toThrow(DomainError)
  })

  it('runs as a worker task, with a plain, cloneable result', async () => {
    const splitter = { $call: 'learning/validate/kFold', args: [{ k: 3, shuffle: true }] }
    const options = {
      rows: 30,
      test: 6,
      fresh: 200,
      grid: [0.01, 1, 100],
      learner: { kind: 'ridge-polynomial', degree: 6 },
      inner: splitter,
      outer: splitter,
      repeats: 2,
    }
    const task = {
      $call: 'theory/model-selection/selectionStudy',
      args: [{ $call: 'foundation/random/stream', args: [14] }, options],
    }
    const result = await evaluateTask(task)
    expect(structuredClone(result)).toEqual(result)
    const k = kFold({ k: 3, shuffle: true })
    expect(result).toEqual(
      selectionStudy(stream(14), { ...options, learner: { kind: 'ridge-polynomial', degree: 6 }, inner: k, outer: k }),
    )
  })

  it('registers the study as a simulation', () => {
    expect(Object.keys(modelSelectionFunctions)).toEqual(['selectionStudy'])
    expect(modelSelectionFunctions.selectionStudy.info.role).toBe('simulation')
  })
})
