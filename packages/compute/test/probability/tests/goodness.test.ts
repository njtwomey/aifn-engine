/** Kolmogorov–Smirnov: statistics and p-values against scipy.stats.kstest / ks_2samp / kstwo (scipy 1.x). */
import { describe, expect, it } from 'vitest'
import { normalCdf } from 'aifn-compute/numerics/special'
import { tensor } from 'aifn-compute/foundation/tensor'
import { kolmogorovLimitSf, kolmogorovSf, ksStatistic, ksTest } from 'aifn-compute/probability/tests'

const Phi = (x: number) => normalCdf(x) as number
const x = [0.1, -0.4, 0.3, 1.2, -1.5, 0.8, 2.1, -0.2, 0.05, -0.9]
const y = [0.5, 1.1, 1.9, 0.7, 2.4, 1.3, 0.2, 1.8]
const rel = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThanOrEqual(tol * Math.abs(b))

describe('one sample against a cdf', () => {
  it('matches kstest(x, "norm"): statistic, location, sign, exact and asymptotic p', () => {
    const t = ksTest(x, Phi)
    expect(t.statistic).toBeCloseTo(0.1445782583896758, 14)
    expect(t.location).toBe(-0.4)
    expect(t.sign).toBe(-1)
    rel(t.pValue, 0.9659757061882045, 1e-9)
    rel(ksTest(x, Phi, { method: 'asymp' }).pValue, 0.9850116471543398, 1e-9)
    expect(ksStatistic(tensor(x), Phi).statistic).toBe(t.statistic)
  })
  it('larger samples, including the far tail', () => {
    const z = Array.from({ length: 40 }, (_, i) => Math.sin(i * 1.7) * 1.5)
    const t = ksTest(z, Phi)
    expect(t.statistic).toBeCloseTo(0.13507580292071997, 13)
    rel(t.pValue, 0.42163626292114764, 1e-8)
    const w = Array.from({ length: 300 }, (_, i) => Math.sin(i * 0.37) * 2)
    const tail = ksTest(w, Phi)
    expect(tail.statistic).toBeCloseTo(0.21126337471526957, 13)
    rel(tail.pValue, 3.1591271190500495e-12, 0.1)
  })
  it('the null laws match kstwo.sf and kstwobign.sf', () => {
    rel(kolmogorovSf(0.3, 10), 0.27053557479999946, 1e-9)
    // n = 1000 accumulates rounding in the 101-square matrix power: ~3e-7 relative.
    rel(kolmogorovSf(0.05, 1000), 0.013012074781090332, 1e-6)
    rel(kolmogorovSf(0.2, 50), 0.03143877776953241, 1e-8)
    rel(kolmogorovLimitSf(1), 0.26999967167735456, 1e-9)
    rel(kolmogorovLimitSf(0.5), 0.9639452436648751, 1e-12)
    rel(kolmogorovLimitSf(0.2), 0.999999999999495, 1e-12)
    // n d ≥ 100: Stephens-corrected limit, against kstwo.sf(0.02, 5000) = 0.03613941395325637.
    rel(kolmogorovSf(0.02, 5000), 0.03613941395325637, 3e-3)
    expect(kolmogorovSf(0, 5)).toBe(1)
    expect(kolmogorovSf(1, 5)).toBe(0)
  })
})

describe('two samples', () => {
  it('matches ks_2samp exactly, and its asymp method', () => {
    const t = ksTest(x, y)
    expect(t.statistic).toBeCloseTo(0.6, 14)
    expect(t.location).toBe(0.1)
    expect(t.sign).toBe(1)
    expect(t.method).toMatch(/exact/)
    rel(t.pValue, 0.04986516751222633, 1e-9)
    const a = ksTest(x, y, { method: 'asymp' })
    rel(a.pValue, 0.06739999999999993, 1e-8)
    const u = Array.from({ length: 30 }, (_, i) => Math.sin(i * 0.9) + 0.01 * i)
    const v = Array.from({ length: 25 }, (_, i) => Math.cos(i * 1.3) + 0.4)
    const b = ksTest(u, v)
    expect(b.statistic).toBeCloseTo(0.28, 14)
    rel(b.pValue, 0.1953653771055992, 1e-9)
    rel(ksTest(u, v, { method: 'asymp' }).pValue, 0.18346750030158931, 1e-8)
  })
  it('identical samples give D = 0 and p = 1; NaN is refused', () => {
    expect(ksTest(x, x).statistic).toBe(0)
    expect(ksTest(x, x).pValue).toBe(1)
    expect(() => ksTest([1, NaN], Phi)).toThrow(/non-finite/)
  })
})
