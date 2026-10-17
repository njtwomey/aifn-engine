/**
 * aifn-methods/neural/self-supervised: the objectives of a one-number linear code (no site check; tests written from
 * the code: the reconstruction optimum is the leading principal direction, InfoNCE is log n and agreement 0 at the
 * collapsed encoder).
 */
import { describe, expect, it } from 'vitest'
import {
  inputCovariance,
  linearCodeLandscape,
  linearCodeObjective,
  thresholdProbeAccuracy,
  type LinearCodeProblem,
} from 'aifn-methods/neural/self-supervised'

// Twelve points spread along (2, 1) with a little spread across it, and two jittered views.
const n = 12
const x = new Float64Array(2 * n)
for (let i = 0; i < n; i++) {
  const t = (i - 5.5) / 3
  const s = (i % 3) - 1
  x[2 * i] = 2 * t - 0.2 * s
  x[2 * i + 1] = t + 0.4 * s
}
const a = x.map((v, i) => v + 0.03 * Math.sin(i))
const b = x.map((v, i) => v - 0.03 * Math.cos(i))
const p: LinearCodeProblem = { x, a, b, n, d: 2 }

describe('inputCovariance', () => {
  it('is the covariance with divisor n', () => {
    const S = inputCovariance(p)
    const m = [0, 1].map((j) => x.filter((_, i) => i % 2 === j).reduce((s, v) => s + v, 0) / n)
    let s01 = 0
    for (let i = 0; i < n; i++) s01 += ((x[2 * i] - m[0]) * (x[2 * i + 1] - m[1])) / n
    expect(S[1]).toBeCloseTo(s01, 12)
    expect(S[2]).toBe(S[1])
  })
})

describe('linearCodeObjective', () => {
  it('reconstruction is smallest on the leading principal direction, equal there to the smaller eigenvalue', () => {
    const [s00, s01, , s11] = inputCovariance(p)
    const half = (s00 + s11) / 2
    const r = Math.hypot((s00 - s11) / 2, s01)
    const [top, bottom] = [half + r, half - r]
    const lead = [s01, top - s00]
    expect(linearCodeObjective(p, lead, 'reconstruction')).toBeCloseTo(bottom, 10)
    for (let k = 0; k < 36; k++) {
      const w = [Math.cos((k * Math.PI) / 36), Math.sin((k * Math.PI) / 36)]
      expect(linearCodeObjective(p, w, 'reconstruction')).toBeGreaterThanOrEqual(bottom - 1e-12)
    }
  })

  it('reconstruction does not depend on the scale of w, and is NaN at w = 0', () => {
    const w = [0.3, -0.7]
    expect(linearCodeObjective(p, [3 * w[0], 3 * w[1]], 'reconstruction')).toBeCloseTo(
      linearCodeObjective(p, w, 'reconstruction'),
      12,
    )
    expect(Number.isNaN(linearCodeObjective(p, [0, 0], 'reconstruction'))).toBe(true)
  })

  it('at the collapsed encoder: InfoNCE is log n, agreement 0, the variance hinge 2μ', () => {
    expect(linearCodeObjective(p, [0, 0], 'contrastive')).toBeCloseTo(Math.log(n), 12)
    expect(linearCodeObjective(p, [0, 0], 'agreement')).toBe(0)
    expect(linearCodeObjective(p, [0, 0], 'variance', { mu: 1.5 })).toBe(3)
  })

  it('InfoNCE falls below log n for an encoder that tells the inputs apart', () => {
    expect(linearCodeObjective(p, [4, 2], 'contrastive')).toBeLessThan(Math.log(n))
  })
})

describe('linearCodeLandscape', () => {
  it('evaluates the grid and reports its smallest value', () => {
    const { axis, z, best, bestValue } = linearCodeLandscape(p, 'agreement', 1, 5)
    expect(axis).toEqual([-1, -0.5, 0, 0.5, 1])
    expect(z[2][2]).toBe(0)
    expect(bestValue).toBe(0)
    expect(best).toEqual([0, 0])
    expect(() => linearCodeLandscape({ ...p, d: 3 }, 'agreement', 1, 5)).toThrow(/two-dimensional/)
  })
})

describe('thresholdProbeAccuracy', () => {
  const xs = Float64Array.from([0, 1, 1, 1, 2, 0, 3, 0])
  it('is 1 for a separating code and the majority share for a constant one', () => {
    expect(thresholdProbeAccuracy(xs, [0, 0, 1, 1], [1, 0], 2)).toBe(1)
    // Every code ties at w = 0: no threshold may split the tie (the site's version reported 1 here).
    expect(thresholdProbeAccuracy(xs, [0, 0, 1, 1], [0, 0], 2)).toBe(0.5)
    expect(thresholdProbeAccuracy(xs, [0, 1, 1, 1], [0, 0], 2)).toBe(0.75)
  })
})
