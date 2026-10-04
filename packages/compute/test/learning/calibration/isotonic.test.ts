/**
 * Isotonic regression by pool adjacent violators against scikit-learn's IsotonicRegression and
 * scipy.optimize.isotonic_regression (`fixtures/learning/calibration.json`), and the step-through form's invariants.
 */
import { describe, expect, it } from 'vitest'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { isotonicRegression, poolAdjacentViolatorsSteps } from 'aifn-compute/learning/calibration'
import { fixture } from '../../fixtures'

type Fit = { x: number[]; y: number[]; weights: number[]; increasing: boolean; fit: number[] }
type Pav = { y: number[]; weights: number[]; fit: number[]; blocks: number[] }
const F = fixture<{ isotonicRegression: Fit[]; poolAdjacentViolatorsSteps: Pav[] }>('learning/calibration')

describe('isotonicRegression', () => {
  it.each(F.isotonicRegression.map((c, i) => [i, c] as const))('matches scikit-learn (case %i)', (_, c) => {
    const { fit } = isotonicRegression(c.y, { x: c.x, weights: c.weights, increasing: c.increasing })
    Array.from(toFlat(fit)).forEach((v, i) => expect(v).toBeCloseTo(c.fit[i], 10))
  })

  it('is the identity on a sorted sequence and a constant on a reversed one', () => {
    expect(Array.from(toFlat(isotonicRegression([1, 2, 3]).fit))).toEqual([1, 2, 3])
    expect(Array.from(toFlat(isotonicRegression([3, 2, 1]).fit))).toEqual([2, 2, 2])
  })
})

describe('poolAdjacentViolatorsSteps', () => {
  it.each(F.poolAdjacentViolatorsSteps.map((c, i) => [i, c] as const))('matches scipy (case %i)', (_, c) => {
    const s = run(poolAdjacentViolatorsSteps(c.y, { weights: c.weights }), undefined, 2 * c.y.length)
    expect(s.done).toBe(true)
    Array.from(toFlat(s.fit)).forEach((v, i) => expect(v).toBeCloseTo(c.fit[i], 10))
    expect(Array.from(toFlat(s.starts))).toEqual(c.blocks.slice(0, -1))
  })

  it('keeps every block but the last in order, and finishes in at most 2n − 1 steps', () => {
    const y = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    const tr = trace(poolAdjacentViolatorsSteps(y), undefined, 100, { keep: 'all' })
    expect(tr.final.done).toBe(true)
    expect(tr.final.t).toBeLessThanOrEqual(2 * y.length - 1)
    for (const s of tr.steps) {
      const v = Array.from(toFlat(s.values))
      for (let k = 1; k < v.length - 1; k++) expect(v[k - 1]).toBeLessThanOrEqual(v[k])
    }
  })
})
