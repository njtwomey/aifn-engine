import { describe, expect, it } from 'vitest'
import { alignment, getMetric, uniformity } from 'aifn-compute/learning/metrics'

const circle = (n: number) =>
  Array.from({ length: n }, (_, i) => [Math.cos((2 * Math.PI * i) / n), Math.sin((2 * Math.PI * i) / n)])

describe('alignment', () => {
  it('is the mean squared distance between paired rows (α = 2), and the mean distance for α = 1', () => {
    const x = [
      [1, 0],
      [0, 1],
    ]
    const y = [
      [0, 1],
      [0, 1],
    ]
    expect(alignment(x, y)).toBeCloseTo(1, 12)
    expect(alignment(x, y, { alpha: 1 })).toBeCloseTo(Math.SQRT2 / 2, 12)
    expect(alignment(x, x)).toBe(0)
  })

  it('rejects inputs of different shapes', () => {
    expect(() => alignment([[1, 0]], [[1, 0, 0]])).toThrow(/alignment/)
  })
})

describe('uniformity', () => {
  it('is 0 for coincident points and matches the closed form for two antipodal points', () => {
    expect(
      uniformity([
        [1, 0],
        [1, 0],
        [1, 0],
      ]),
    ).toBeCloseTo(0, 12)
    // One pair at squared distance 4: log exp(−2 · 4) = −8.
    expect(
      uniformity([
        [1, 0],
        [-1, 0],
      ]),
    ).toBeCloseTo(-8, 12)
  })

  it('falls as points spread out over the circle', () => {
    const clumped = circle(12).map(([c, s]) => [Math.cos(0.1 * Math.atan2(s, c)), Math.sin(0.1 * Math.atan2(s, c))])
    expect(uniformity(circle(12))).toBeLessThan(uniformity(clumped))
  })

  it('agrees with a direct mean for t = 1', () => {
    const x = [
      [0.6, 0.8],
      [1, 0],
      [0, -1],
    ]
    const d2 = (a: number[], b: number[]) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    const direct = Math.log((Math.exp(-d2(x[0], x[1])) + Math.exp(-d2(x[0], x[2])) + Math.exp(-d2(x[1], x[2]))) / 3)
    expect(uniformity(x, { t: 1 })).toBeCloseTo(direct, 12)
  })

  it('is registered, lower is better', () => {
    expect(getMetric('uniformity').info.direction).toBe('lower')
    expect(getMetric('alignment').info.direction).toBe('lower')
  })
})
