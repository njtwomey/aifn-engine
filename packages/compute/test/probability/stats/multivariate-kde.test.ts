import { describe, expect, it } from 'vitest'
import { normals, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { multivariateKde } from 'aifn-compute/probability/stats'

describe('multivariateKde', () => {
  it('matches scipy.stats.gaussian_kde (Scott factor) in two dimensions', () => {
    // scipy 1.x: gaussian_kde(x.T)(q.T) for the points below; factor 6^(−1/6).
    const x = fromData(Float64Array.of(0, 0, 1, 0.5, 0.3, -0.4, -0.7, 0.9, 1.5, 1.2, -1, -0.8), [6, 2])
    const q = fromData(Float64Array.of(0, 0, 0.5, 0.5, 2, -1), [3, 2])
    const k = multivariateKde(x, q)
    expect(k.factor).toBeCloseTo(0.7418363755904023, 12)
    const d = toFlat(k.density)
    expect(d[0]).toBeCloseTo(0.17484300024166893, 12)
    expect(d[1]).toBeCloseTo(0.15630266399289958, 12)
    expect(d[2] / 7.550786803460316e-5).toBeCloseTo(1, 10)
  })

  it('law: integrates to 1 over a wide grid, and log density is finite far away', () => {
    const x = normals(stream(1), [200, 2])
    const g = 80
    const h = 16 / g
    const pts = new Float64Array(2 * g * g)
    for (let i = 0; i < g; i++)
      for (let j = 0; j < g; j++) {
        pts[2 * (i * g + j)] = -8 + (j + 0.5) * h
        pts[2 * (i * g + j) + 1] = -8 + (i + 0.5) * h
      }
    const k = multivariateKde(x, fromData(pts, [g * g, 2]))
    const total = toFlat(k.density).reduce((s, v) => s + v * h * h, 0)
    expect(total).toBeCloseTo(1, 3)
    const far = multivariateKde(x, fromData(Float64Array.of(60, 60), [1, 2]))
    expect(Number.isFinite(toFlat(far.logDensity)[0])).toBe(true)
    expect(toFlat(far.density)[0]).toBe(0)
  })

  it('cross-validation follows narrow separated modes where Scott oversmooths', () => {
    // Four tight clusters (sd 0.05) at (±2, ±2): the global covariance is about 4I, so Scott's factor blurs each one.
    const s = stream(4)
    const n = 200
    const pts = new Float64Array(2 * n)
    const z = toFlat(normals(s, [n, 2]))
    for (let i = 0; i < n; i++) {
      pts[2 * i] = (i % 2 ? 2 : -2) + 0.05 * z[2 * i]
      pts[2 * i + 1] = (i % 4 < 2 ? 2 : -2) + 0.05 * z[2 * i + 1]
    }
    const x = fromData(pts, [n, 2])
    const at = fromData(Float64Array.of(2, 2), [1, 2])
    const scott = multivariateKde(x, at)
    const cv = multivariateKde(x, at, { bandwidth: 'cross-validation' })
    expect(cv.factor).toBeLessThan(scott.factor / 3)
    // The true density at a cluster centre is ¼ · 1/(2π·0.05²) ≈ 15.9.
    expect(toFlat(cv.density)[0] / 15.9).toBeGreaterThan(0.6)
    expect(toFlat(scott.density)[0] / 15.9).toBeLessThan(0.2)
  })
})
