import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { expectile } from 'aifn-compute/probability/stats'
import { curve1d, type Curve1dCase } from 'aifn-methods/data/synthetic'
import { datasetRegistry, type Curve1dTruth } from 'aifn-methods/data'

const CASES: Curve1dCase[] = ['sine', 'skewed', 'counts', 'binary', 'gamma']
const at = (v: number) => fromData(Float64Array.of(v), [1, 1])

describe('curve1d', () => {
  it('draws n sorted inputs on [0, 1] with responses in the family support, and is registered', () => {
    expect(datasetRegistry.curve1d).toBeDefined()
    for (const c of CASES) {
      const d = curve1d(stream(`curve-${c}`), { case: c, n: 200 })
      const x = toFlat(d.x as Tensor)
      const y = toFlat(d.y as Tensor)
      expect(x.length).toBe(200)
      for (let i = 1; i < x.length; i++) expect(x[i]).toBeGreaterThanOrEqual(x[i - 1])
      if (c === 'counts') expect(y.every((v) => Number.isInteger(v) && v >= 0)).toBe(true)
      if (c === 'binary') expect(y.every((v) => v === 0 || v === 1)).toBe(true)
      if (c === 'gamma') expect(y.every((v) => v > 0)).toBe(true)
    }
  })

  it('the ½-expectile is the mean; expectiles increase with τ', () => {
    for (const c of CASES) {
      const t = curve1d(stream('e'), { case: c, n: 10 }).meta!.truth as Curve1dTruth
      const x = at(0.37)
      expect(toFlat(t.expectile(x, 0.5))[0]).toBeCloseTo(toFlat(t.mean(x))[0], 3)
      expect(toFlat(t.expectile(x, 0.9))[0]).toBeGreaterThan(toFlat(t.expectile(x, 0.6))[0])
    }
  })

  it('normal noise: e₀.₉ = μ + 0.862σ, and 80.6% of the population lies below it', () => {
    const t = curve1d(stream('n'), { case: 'sine', n: 10, noiseShape: 'normal' }).meta!.truth as Curve1dTruth
    const x = at(0.2)
    const e = toFlat(t.expectile(x, 0.9))[0]
    expect((e - toFlat(t.mean(x))[0]) / toFlat(t.sdAt(x))[0]).toBeCloseTo(0.8616, 2)
    expect(t.shareBelow(0.9)).toBeCloseTo(0.806, 2)
  })

  it('a Bernoulli expectile is τp/(τp + (1 − τ)(1 − p)); a large sample agrees with the truth', () => {
    const t = curve1d(stream('b'), { case: 'binary', n: 10 }).meta!.truth as Curve1dTruth
    const x = at(0.5)
    const p = toFlat(t.mean(x))[0]
    expect(toFlat(t.expectile(x, 0.8))[0]).toBeCloseTo((0.8 * p) / (0.8 * p + 0.2 * (1 - p)), 10)
    // Skewed noise: the sample expectile of many draws at one x matches.
    const s = curve1d(stream('s'), { case: 'skewed', n: 10 }).meta!.truth as Curve1dTruth
    const rows = fromData(new Float64Array(40000).fill(0.6), [40000, 1])
    const draws = s.predictive(rows).sample(stream('draws')) as Tensor
    expect(expectile(draws, 0.9)).toBeCloseTo(toFlat(s.expectile(at(0.6), 0.9))[0], 1)
  })
})
