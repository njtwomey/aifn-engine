/**
 * Subset selection on the standardised diabetes data (the `subsets` cases of fixture `learning/linear`): `bestSubset`
 * against brute force over all 1024 subsets with numpy's `lstsq`, `stepwise` against a numpy greedy search, and
 * `forwardStagewise` against a numpy loop and, for small steps, against the lasso path of `leastAngleRegression`.
 */
import { describe, expect, it } from 'vitest'
import { diabetes } from 'aifn-methods/data/real'
import { standardScaler } from 'aifn-methods/learning/preprocessing'
import {
  bestSubset,
  forwardStagewise,
  leastAngleRegression,
  linearFunctions,
  stepwise,
} from 'aifn-methods/learning/linear'
import { fromData, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'

type Step = { action: 'start' | 'add' | 'remove'; feature: number; active: number[]; rss: number; criterion: number }
type Fixture = {
  all: { sizes: number[]; subsets: number[][]; rss: number[] }
  best: { size: number; subset: number[]; rss: number; coef: number[]; intercept: number }[]
  stepwise: { direction: 'forward' | 'backward' | 'both'; criterion: 'rss' | 'aic' | 'bic'; steps: Step[] }[]
  stagewise: { epsilon: number; steps: number; coef: number[][]; features: number[]; rss: number[] }
}
const fx = fixture<{ subsets: Fixture }>('learning/linear').subsets

const data = diabetes()
const x = standardScaler().fit({ x: data.x }).transform(data.x)
const y = data.y!

const expectRelative = (actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number, what: string): void => {
  expect(actual.length, what).toBe(expected.length)
  for (let i = 0; i < expected.length; i++)
    expect(Math.abs(actual[i] - expected[i]), `${what}[${i}]: ${actual[i]} vs ${expected[i]}`).toBeLessThan(
      tol * Math.max(1, Math.abs(expected[i])),
    )
}

describe('bestSubset', () => {
  it('finds every subset’s RSS and the best subset of each size, as brute force with lstsq', () => {
    const best = bestSubset(x, y)
    expect(best.all.sizes).toEqual(fx.all.sizes)
    expect(best.all.subsets).toEqual(fx.all.subsets)
    expectRelative(best.all.rss, fx.all.rss, 1e-10, 'all rss')
    expect(best.sizes).toEqual(fx.best.map((b) => b.size))
    expect(best.subsets).toEqual(fx.best.map((b) => b.subset))
    expectRelative(
      best.rss,
      fx.best.map((b) => b.rss),
      1e-10,
      'best rss',
    )
    const coef = toRows(best.coef)
    fx.best.forEach((b, k) => {
      expectRelative(
        coef.map((row) => row[k]),
        b.coef,
        1e-8,
        `coef of size ${k}`,
      )
      expect(best.intercept[k]).toBeCloseTo(b.intercept, 8)
    })
  })

  it('stops at maxSize, handles a collinear column, and refuses a search that is too large', () => {
    const small = bestSubset(x, y, { maxSize: 2 })
    expect(small.sizes).toEqual([0, 1, 2])
    expect(small.all.rss).toHaveLength(1 + 10 + 45)
    // A copy of column 2 adds nothing: the best pair containing both has the RSS of column 2 alone.
    const rows = toRows(x)
    const doubled = tensor(rows.map((r) => [r[2], r[2], r[3]]))
    const d = bestSubset(doubled, y)
    const i = d.all.subsets.findIndex((s) => s.length === 2 && s[0] === 0 && s[1] === 1)
    expect(d.all.rss[i]).toBeCloseTo(d.all.rss[1], 6)
    expect(() => bestSubset(x, y, { maxSize: 11 })).toThrow(DomainError)
    expect(() => bestSubset(fromData(new Float64Array(30 * 25), [30, 25]), fromData(new Float64Array(30)))).toThrow(
      DomainError,
    )
  })
})

describe('stepwise', () => {
  it('makes the moves of a numpy greedy search, for each direction and criterion', () => {
    for (const run of fx.stepwise) {
      const what = `${run.direction} ${run.criterion}`
      const s = stepwise(x, y, { direction: run.direction, criterion: run.criterion })
      expect(
        s.steps.map((t) => [t.action, t.feature, t.active]),
        what,
      ).toEqual(run.steps.map((t) => [t.action, t.feature, t.active]))
      expectRelative(
        s.steps.map((t) => t.rss),
        run.steps.map((t) => t.rss),
        1e-10,
        `${what} rss`,
      )
      expectRelative(
        s.steps.map((t) => t.criterion),
        run.steps.map((t) => t.criterion),
        1e-10,
        `${what} criterion`,
      )
    }
  })

  it('visits a model of every size going forward by RSS, never better than the best subset', () => {
    const s = stepwise(x, y)
    const best = bestSubset(x, y)
    expect(s.steps.map((t) => t.active.length)).toEqual(best.sizes)
    s.steps.forEach((t, k) => expect(t.rss).toBeGreaterThanOrEqual(best.rss[k] * (1 - 1e-12)))
    // The weights of each model are its least-squares fit: the full model's are those of the best subset of size 10.
    const coef = toRows(s.coef)
    const full = toRows(best.coef)
    for (let j = 0; j < 10; j++) expect(coef[j][10]).toBeCloseTo(full[j][10], 8)
  })

  it('throws for an unknown direction, both with RSS, or a bad start', () => {
    expect(() => stepwise(x, y, { direction: 'sideways' as 'forward' })).toThrow(DomainError)
    expect(() => stepwise(x, y, { direction: 'both', criterion: 'rss' })).toThrow(DomainError)
    expect(() => stepwise(x, y, { direction: 'both', start: [12] })).toThrow(DomainError)
  })
})

describe('forwardStagewise', () => {
  it('takes the steps of a numpy loop', () => {
    const { epsilon, steps } = fx.stagewise
    const path = forwardStagewise(x, y, { epsilon, steps })
    expect(path.features).toEqual(fx.stagewise.features)
    expect(path.converged).toBe(fx.stagewise.features.length < steps)
    expectRelative(toFlat(path.coef), fx.stagewise.coef.flat(), 1e-9, 'coef')
    expectRelative(path.rss, fx.stagewise.rss, 1e-9, 'rss')
  })

  it('tends to the lasso path while that path is monotone, as epsilon shrinks', () => {
    const lasso = leastAngleRegression(x, y, { method: 'lasso' })
    const knots = toRows(lasso.coef)
    // The last knot up to which no lasso weight has shrunk in magnitude: there the lasso is the monotone lasso, the
    // limit of stagewise (Hastie, Taylor, Tibshirani and Walther, 2007). Here a weight turns back after knot 7.
    let monotone = 0
    while (
      monotone + 1 < lasso.alphas.length &&
      knots.every((row) => Math.abs(row[monotone + 1]) >= Math.abs(row[monotone]) - 1e-12)
    )
      monotone++
    expect(monotone).toBe(7)
    for (const epsilon of [0.2, 0.02]) {
      const path = forwardStagewise(x, y, { epsilon, steps: 100_000 })
      expect(path.converged).toBe(true)
      const coef = toRows(path.coef)
      let worst = 0
      for (let t = 1; t <= monotone; t++) {
        // The stagewise point with the same l1 norm as the knot.
        const s = path.l1Norm.findIndex((v) => v >= lasso.l1Norm[t])
        for (let j = 0; j < 10; j++) worst = Math.max(worst, Math.abs(coef[j][s] - knots[j][t]))
      }
      expect(worst, `epsilon ${epsilon}`).toBeLessThan(epsilon)
    }
  })

  it('throws for a step that is not positive', () => {
    expect(() => forwardStagewise(x, y, { epsilon: 0 })).toThrow(DomainError)
    expect(() => forwardStagewise(x, y, { steps: 1.5 })).toThrow(DomainError)
  })

  it('is registered', () => {
    expect(linearFunctions.bestSubset).toBe(bestSubset)
    expect(linearFunctions.stepwise).toBe(stepwise)
    expect(linearFunctions.forwardStagewise).toBe(forwardStagewise)
  })
})
