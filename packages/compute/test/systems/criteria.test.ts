import { describe, expect, test } from 'vitest'
import { closedLoopPolesAt, nyquist, rootLocus, routhArray, transferFunction } from 'aifn-compute/systems'
import { roots } from 'aifn-compute/numerics/polynomial'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toComplexFlat, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../fixtures'

type Locus = { num: number[]; den: number[]; gains: number[]; re: number[][]; im: number[][] }
type NyquistCase = { num: number[]; den: number[]; dt: number | null; count: number; P: number }
const F = fixture<{ rootLocus: Locus[]; nyquist: NyquistCase[] }>('systems')

const rhpCount = (c: number[]) => toComplexFlat(roots(c)).filter((z) => z.re > 1e-9).length

describe('Routh–Hurwitz', () => {
  test('counts right-half-plane roots of random polynomials as the roots do', () => {
    const s = stream('routh')
    for (let trial = 0; trial < 200; trial++) {
      const degree = 2 + (trial % 6)
      const c = [1, ...toFlat(uniform(s, -3, 3, { shape: [degree] }) as Tensor)]
      const r = routhArray(c)
      expect(r.rightHalfPlane).toBe(rhpCount(c))
      expect(r.stable).toBe(rhpCount(c) === 0)
    }
  })

  test('a stable cubic, a zero leading entry (ε) and a row of zeros (auxiliary polynomial)', () => {
    expect(routhArray([1, 6, 11, 6]).stable).toBe(true)
    // s⁴ + s³ + 2s² + 2s + 3: the s² row starts with 0; two roots in the right half-plane.
    const eps = routhArray([1, 1, 2, 2, 3])
    expect(eps.epsilonRows.length).toBe(1)
    expect(eps.rightHalfPlane).toBe(2)
    // (s² + 1)(s + 2)(s + 3): the s¹ row vanishes; two roots on the imaginary axis, none on the right.
    const aux = routhArray([1, 5, 7, 5, 6])
    expect(aux.auxiliaryRows.length).toBe(1)
    expect(aux.rightHalfPlane).toBe(0)
    expect(aux.imaginaryAxis).toBe(2)
    expect(aux.stable).toBe(false)
    // (s² − 4)(s + 1)…: s⁵ + 2s⁴ − 3s³ − 6s² − 4s − 8 has one right root (s = 2), and ±i on the axis.
    const mixed = routhArray([1, 2, -3, -6, -4, -8])
    expect(mixed.rightHalfPlane).toBe(rhpCount([1, 2, -3, -6, -4, -8]))
  })
})

describe('root locus', () => {
  test('closed-loop poles at each gain match python-control', () => {
    for (const c of F.rootLocus) {
      const L = transferFunction(c.num, c.den)
      const r = rootLocus(L, { gains: c.gains })
      c.gains.forEach((k, i) => {
        const got = r.branches.map((b) => b[i]).sort((u, v) => u.re - v.re || u.im - v.im)
        const want = c.re[i].map((re, j) => ({ re, im: c.im[i][j] })).sort((u, v) => u.re - v.re || u.im - v.im)
        expect(got.length).toBe(want.length)
        got.forEach((z, j) => {
          expect(Math.abs(z.re - want[j].re)).toBeLessThan(1e-6 * (1 + k))
          expect(Math.abs(z.im - want[j].im)).toBeLessThan(1e-6 * (1 + k))
        })
      })
    }
  })

  test('asymptotes, breakaway point and imaginary-axis crossing of 1/(s(s + 1)(s + 2))', () => {
    const r = rootLocus(transferFunction([1], [1, 3, 2, 0]))
    expect(r.asymptotes.centroid).toBeCloseTo(-1, 12)
    expect(r.asymptotes.angles.map((a) => (a * 180) / Math.PI)).toEqual([60, 180, 300].map((v) => expect.closeTo(v, 9)))
    // dk/ds = 0: 3s² + 6s + 2 = 0, the root between 0 and −1 with k > 0.
    expect(r.breakaway.length).toBe(1)
    expect(r.breakaway[0].s).toBeCloseTo(-1 + 1 / Math.sqrt(3), 9)
    // Routh: the s¹ row vanishes at k = 6, with poles at ±i√2.
    expect(r.crossings.length).toBe(1)
    expect(r.crossings[0].gain).toBeCloseTo(6, 6)
    expect(r.crossings[0].at.im).toBeCloseTo(Math.SQRT2, 5)
    // Branches are continuous: no jump larger than a few percent of the locus scale between neighbouring gains.
    for (const b of r.branches)
      for (let i = 1; i < b.length; i++)
        expect(Math.hypot(b[i].re - b[i - 1].re, b[i].im - b[i - 1].im)).toBeLessThan(2)
    expect(closedLoopPolesAt(transferFunction([1], [1, 3, 2, 0]), 0).length).toBe(3)
  })

  test('gains given as a tensor are read, not dropped; negative gains are refused (review G2)', () => {
    const L = transferFunction([1], [1, 3, 2, 0])
    const r = rootLocus(L, { gains: fromData(Float64Array.from([1, 6.5, 10]), [3]) })
    expect(toFlat(r.gains)).toEqual([0, 1, 6.5, 10])
    expect(r.crossings.length).toBe(1)
    expect(r.crossings[0].gain).toBeCloseTo(6, 6)
    expect(() => rootLocus(L, { gains: [-1, 2] })).toThrow(/non-negative/)
  })
})

describe('Nyquist', () => {
  test('encirclements and closed-loop stability match python-control', () => {
    for (const c of F.nyquist) {
      // python-control's discrete polynomials are in descending powers of z; aifn's in ascending z⁻¹, so the
      // numerator is shifted by the relative degree.
      const L =
        c.dt === null
          ? transferFunction(c.num, c.den)
          : transferFunction([...new Array<number>(c.den.length - c.num.length).fill(0), ...c.num], c.den, { dt: c.dt })
      const n = nyquist(L)
      expect(n.openLoopUnstable).toBe(c.P)
      expect(n.encirclements).toBe(c.count)
      expect(n.closedLoopUnstable).toBe(c.count + c.P)
    }
  })

  test('agrees with Routh on the closed loop of k/(s + 1)³ across the critical gain 8', () => {
    for (const k of [2, 7.5, 8.5, 30]) {
      const n = nyquist(transferFunction([k], [1, 3, 3, 1]))
      const r = routhArray([1, 3, 3, 1 + k])
      expect(n.closedLoopUnstable).toBe(r.rightHalfPlane)
    }
  })

  test('the indentation around integrators gives a closed contour', () => {
    const n = nyquist(transferFunction([2, 1], [1, 1, 0, 0]))
    expect(n.stable).toBe(true)
    const re = toFlat(n.re)
    expect(Math.max(...re.map(Math.abs))).toBeGreaterThan(1e3)
  })
})
