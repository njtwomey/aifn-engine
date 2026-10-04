import { describe, expect, it } from 'vitest'
import { trackingMetrics } from 'aifn-compute/inference/filtering'

describe('trackingMetrics', () => {
  it('reads lag, overshoot, noise, errors and coverage off a hand case', () => {
    // Truth steps from 0 to 10 at index 3; the estimate takes three values to cover 90 % and overshoots by 1.
    const truth = [0, 0, 0, 10, 10, 10, 10, 10]
    const estimate = [1, -1, 0, 4, 8, 9, 11, 10]
    const sd = [2, 2, 2, 2, 2, 2, 2, 2]
    const m = trackingMetrics(estimate, truth, sd, { change: 3 })
    expect(m.lag).toBe(3)
    expect(m.overshoot).toBe(1)
    expect(m.noise).toBeCloseTo(Math.sqrt((1 + 0) / 2), 12)
    expect(m.rmseBefore).toBeCloseTo(Math.sqrt(2 / 3), 12)
    expect(m.rmseAfter).toBeCloseTo(Math.sqrt((36 + 4 + 1 + 1 + 0) / 5), 12)
    // |error| ≤ 3.92 everywhere except −6 at index 3.
    expect(m.coverage).toBeCloseTo(7 / 8, 12)
  })
  it('a downward step, no change, and no sds', () => {
    const m = trackingMetrics([5, 5, 2, -4], [5, 5, -5, -5], null, { change: 2, level: 0.5 })
    expect(m.lag).toBe(2)
    expect(m.overshoot).toBe(0)
    expect(m.coverage).toBeNaN()
    const n = trackingMetrics([1, 2], [0, 0])
    expect(n.lag).toBeNaN()
    expect(n.rmseBefore).toBeCloseTo(Math.sqrt(2.5), 12)
  })
})
