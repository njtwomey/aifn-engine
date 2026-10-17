/**
 * Threshold learners on the site's self-check: the ERM interval (0.3, 0.6) with midpoint 0.45, a reversed pair with
 * one error, the low-density learner finding the trough between clusters at 0.2 and 0.8, and the median rule.
 */
import { describe, expect, it } from 'vitest'
import {
  ermInterval,
  lowDensityThreshold,
  medianThreshold,
  thresholdErm,
  thresholdErrors,
  unlabelledDensity,
} from 'aifn-methods/learning/semi-supervised'

const s1 = { x: [0.6, 0.1, 0.9, 0.3], y: [1, 0, 1, 0] }

describe('supervised threshold', () => {
  it('finds the interval (0.3, 0.6) with no errors and its midpoint 0.45', () => {
    expect(ermInterval(s1, [0, 1])).toEqual({ lo: 0.3, hi: 0.6, errors: 0 })
    expect(thresholdErm(s1, [0, 1])).toBeCloseTo(0.45, 12)
  })

  it('takes the first interval (0, 0.2) with one error for a reversed pair', () => {
    expect(ermInterval({ x: [0.2, 0.7], y: [1, 0] }, [0, 1])).toEqual({ lo: 0, hi: 0.2, errors: 1 })
    expect(thresholdErrors({ x: [0.2, 0.7], y: [1, 0] }, 0.5)).toBe(2)
    expect(() => ermInterval({ x: [0], y: [2] }, [0, 1])).toThrow(/0 or 1/)
  })
})

describe('semi-supervised threshold', () => {
  it('finds the trough between unlabelled clusters at 0.2 and 0.8', () => {
    const un: number[] = []
    for (let i = 0; i < 50; i++) un.push(0.2 + (i - 25) * 0.002, 0.8 + (i - 25) * 0.002)
    const t = lowDensityThreshold({ x: [0.1, 0.9], y: [0, 1] }, unlabelledDensity(un, [0, 1]), [0, 1])
    expect(Math.abs(t - 0.5)).toBeLessThan(0.02)
  })

  it('falls back to the supervised midpoint without a density', () => {
    expect(unlabelledDensity([0.5], [0, 1])).toBeNull()
    expect(lowDensityThreshold(s1, null, [0, 1])).toBe(thresholdErm(s1, [0, 1]))
  })

  it('the median rule is the midpoint under a uniform distribution function', () => {
    expect(
      medianThreshold(
        s1,
        (x) => x,
        (p) => p,
        [0, 1],
      ),
    ).toBeCloseTo(0.45, 12)
  })
})
