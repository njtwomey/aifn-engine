/**
 * Chebyshev series (aifn-compute/numerics/interpolate), against closed forms (numpy's `chebval` and `chebinterpolate`
 * conventions) and the site's self-check (chebyshev-graph-filter).
 */
import { describe, expect, it } from 'vitest'
import { chebyshevInterpolation, chebyshevNodes, chebyshevSeries } from 'aifn-compute/numerics/interpolate'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'

describe('chebyshevSeries', () => {
  it('chebval([1, 2, 3], x) = 1 + 2x + 3(2x^2 - 1)', () => {
    for (const x of [-1, -0.3, 0.5, 1, 2])
      expect(chebyshevSeries([1, 2, 3], x)).toBeCloseTo(1 + 2 * x + 3 * (2 * x * x - 1), 12)
  })
  it('maps [a, b] to [-1, 1], keeps the shape of a vector argument', () => {
    const y = chebyshevSeries([0, 1], tensor([0, 2, 4]), [0, 4])
    expect(y.shape).toEqual([3])
    expect(Array.from(toFlat(y))).toEqual([-1, 0, 1])
  })
  it('throws for no coefficients or an empty interval', () => {
    expect(() => chebyshevSeries([], 0)).toThrow()
    expect(() => chebyshevSeries([1], 0, [1, 1])).toThrow()
  })
})

describe('chebyshevInterpolation', () => {
  it('x^2 = (T0 + T2)/2 on [-1, 1]', () => {
    const c = toFlat(chebyshevInterpolation((x) => x * x, 2))
    ;[0.5, 0, 0.5].forEach((v, k) => expect(c[k]).toBeCloseTo(v, 14))
  })
  it('K = 2 reproduces a quadratic on [0, 4] at 0, 0.3, 1.7, 2.9 and 4 (site check)', () => {
    const g = (l: number) => 1 - 0.5 * l + 0.1 * l * l
    const c = chebyshevInterpolation(g, 2, [0, 4])
    for (const l of [0, 0.3, 1.7, 2.9, 4]) expect(chebyshevSeries(c, l, [0, 4])).toBeCloseTo(g(l), 12)
  })
  it('interpolates at the Chebyshev nodes', () => {
    const f = (x: number) => Math.exp(x) * Math.sin(3 * x)
    const c = chebyshevInterpolation(f, 5, [-2, 1])
    for (const x of toFlat(chebyshevNodes(6, -2, 1))) expect(chebyshevSeries(c, x, [-2, 1])).toBeCloseTo(f(x), 12)
  })
  it('matches the site candidate exactly on [0, lambda max]', () => {
    // The candidate's formulas, verbatim.
    const lambdaMax = 6
    const g = (l: number) => Math.exp(-l)
    const m = 4
    const values = Array.from({ length: m }, (_, j) => g((lambdaMax / 2) * (Math.cos((Math.PI * (j + 0.5)) / m) + 1)))
    const theta = Array.from({ length: m }, (_, k) => {
      let s = 0
      for (let j = 0; j < m; j++) s += values[j] * Math.cos((Math.PI * k * (j + 0.5)) / m)
      return ((k === 0 ? 1 : 2) / m) * s
    })
    expect(Array.from(toFlat(chebyshevInterpolation(g, 3, [0, lambdaMax])))).toEqual(theta)
  })
})
