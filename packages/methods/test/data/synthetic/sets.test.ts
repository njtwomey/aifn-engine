/** Set datasets (aifn-methods/data/synthetic), against the site's self-check (set-networks). */
import { describe, expect, it } from 'vitest'
import { numberSets, shapeClouds } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'

describe('numberSets', () => {
  const d = numberSets(stream('check'), { count: 50, minSize: 2, maxSize: 5 })
  it('sizes in [2, 5]; targets are the sum, maximum and population variance', () => {
    expect(d.sets.every((s) => s.length >= 2 && s.length <= 5)).toBe(true)
    d.sets.forEach((v, b) => {
      const m = v.reduce((a, x) => a + x, 0) / v.length
      expect(d.targets.sum[b]).toBeCloseTo(m * v.length, 12)
      expect(d.targets.max[b]).toBe(Math.max(...v))
      expect(d.targets.variance[b]).toBeCloseTo(v.reduce((a, x) => a + (x - m) ** 2, 0) / v.length, 12)
    })
  })
  it('set b does not change when more sets are drawn', () => {
    expect(numberSets(stream('check'), { count: 3, minSize: 2, maxSize: 5 }).sets).toEqual(d.sets.slice(0, 3))
    expect(() => numberSets(stream(0), { count: 1, minSize: 3, maxSize: 2 })).toThrow()
  })
})

describe('shapeClouds', () => {
  it('labels cycle [0, 1, 2, 0, 1, 2]; a noise-free circle has one radius', () => {
    const c = shapeClouds(stream('check'), { count: 6, points: 40, noise: 0 })
    expect(c.labels).toEqual([0, 1, 2, 0, 1, 2])
    const radii = c.clouds[0].map(([x, y]) => Math.hypot(x, y))
    expect(Math.max(...radii) - Math.min(...radii)).toBeLessThan(1e-12)
    expect(c.clouds.every((cloud) => cloud.length === 40)).toBe(true)
  })
})
