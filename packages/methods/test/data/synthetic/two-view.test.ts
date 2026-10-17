/**
 * `twoViewDataset`: uniform marginals, the closed-form Bayes error of one view, and the rotated copy at dependence 1
 * (the numbers of the site's self-training and co-training self-check).
 */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { classCoordinate, twoViewDataset } from 'aifn-methods/data/synthetic'

describe('twoViewDataset', () => {
  it('has a uniform first coordinate and the closed-form Bayes error 0.0866 of one view (sample 0.0847)', () => {
    const d = twoViewDataset(stream(3), { n: 40000, sharpness: 8, dependence: 0 })
    const a = toFlat(d.a)
    const y = toFlat(d.y)
    let m = 0
    let v = 0
    let err = 0
    for (let i = 0; i < y.length; i++) {
      m += a[2 * i]
      v += a[2 * i] ** 2
      if ((a[2 * i] > 0 ? 1 : 0) !== y[i]) err++
    }
    m /= y.length
    v /= y.length
    expect(Math.abs(m)).toBeLessThan(0.01)
    expect(Math.abs(v - 1 / 3)).toBeLessThan(0.01)
    expect(d.truth.viewBayesError).toBeCloseTo((Math.LN2 - Math.log1p(Math.exp(-8))) / 8, 12)
    expect(d.truth.viewBayesError).toBeCloseTo(0.0866, 4)
    expect(Math.abs(err / y.length - d.truth.viewBayesError)).toBeLessThan(0.005)
    expect(d.y.dtype).toBe('int32')
    expect(d.x.shape).toEqual([40000, 4])
  })

  it('copies view a rotated, b = (-a2, a1), at dependence 1', () => {
    const d = twoViewDataset(stream(3), { n: 1000, dependence: 1 })
    const a = toFlat(d.a)
    const b = toFlat(d.b)
    expect(Array.from(a).every((u, i) => (i % 2 === 0 ? b[i + 1] === u : b[i - 1] === -u))).toBe(true)
    expect(d.copied.every((c) => c === 1)).toBe(true)
  })

  it('inverts the class distribution function: the median of class 1 solves e^{su} = (1 + e^{-s}) e^{s/2} - 1', () => {
    const u = classCoordinate(0.5, 1, 8)
    expect(Math.exp(8 * u)).toBeCloseTo((1 + Math.exp(-8)) * Math.exp(4) - 1, 9)
    expect(classCoordinate(0.5, 0, 8)).toBe(-u)
    expect(classCoordinate(0, 1, 8)).toBeCloseTo(-1, 12)
    expect(classCoordinate(1, 1, 8)).toBeCloseTo(1, 12)
  })

  it('refuses a dependence outside [0, 1]', () => {
    expect(() => twoViewDataset(stream(1), { dependence: 2 })).toThrow(/dependence/)
  })
})
