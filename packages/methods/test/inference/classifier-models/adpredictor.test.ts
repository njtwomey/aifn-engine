import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { simulatedImpressions } from 'aifn-methods/data/synthetic'
import { adPredictor, adPredictorProbability, adPredictorUpdate } from 'aifn-methods/inference/classifier-models'

describe('AdPredictor', () => {
  it('a click raises the active weights’ means and shrinks their variances, leaving the rest', () => {
    const m = adPredictor(4)
    const u = adPredictorUpdate(m, [0, 2], true)
    expect(u.mean[0]).toBeGreaterThan(0)
    expect(u.mean[2]).toBeCloseTo(u.mean[0], 12)
    expect(u.variance[0]).toBeLessThan(1)
    expect(u.mean[1]).toBe(0)
    expect(u.variance[1]).toBe(1)
    const n = adPredictorUpdate(m, [0, 2], false)
    expect(n.mean[0]).toBeCloseTo(-u.mean[0], 12)
  })
  it('learns the weights of simulated impressions and predicts calibrated clicks', () => {
    const d = simulatedImpressions(stream(3), { n: 8000 })
    let m = adPredictor(d.features)
    d.impressions.forEach((a, r) => (m = adPredictorUpdate(m, a, d.clicks[r])))
    let num = 0
    let den1 = 0
    let den2 = 0
    for (let i = 1; i < d.features; i++) {
      num += m.mean[i] * d.weights[i]
      den1 += m.mean[i] ** 2
      den2 += d.weights[i] ** 2
    }
    expect(num / Math.sqrt(den1 * den2)).toBeGreaterThan(0.9)
    // Mean predicted probability on the last impressions matches their click rate.
    const last = d.impressions.slice(-2000)
    const p = last.reduce((s, a) => s + adPredictorProbability(m, a), 0) / last.length
    const rate = d.clicks.slice(-2000).filter(Boolean).length / last.length
    expect(Math.abs(p - rate)).toBeLessThan(0.03)
  })
})
