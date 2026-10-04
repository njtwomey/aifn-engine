import { describe, expect, it } from 'vitest'
import { placePoles } from 'aifn-compute/systems'
import { ackermann } from 'aifn-compute/dynamics/control'
import { imagPart, realPart, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { fixture } from '../fixtures'

type PlaceCase = {
  A: number[][]
  B: number[][]
  real: number[]
  complexRe: number[]
  complexIm: number[]
  K: number[][]
  iterations: number
  condKnv: number
  condYtReal: number
  condYt: number
}
const F = fixture<{ place: Record<string, PlaceCase> }>('systems').place

/** The achieved poles, sorted (re, im) for comparison. */
function sortedPoles(re: ArrayLike<number>, im: ArrayLike<number>): [number, number][] {
  return Array.from(re, (r, i) => [r, im[i]] as [number, number]).sort((a, b) => a[0] - b[0] || a[1] - b[1])
}

describe('placePoles (Kautsky–Nichols–Van Dooren)', () => {
  it.each(Object.keys(F))('matches scipy place_poles KNV0 on real poles (%s)', (key) => {
    const c = F[key]
    const r = placePoles({ A: c.A, B: c.B }, c.real)
    toRows(r.K).forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(c.K[i][j], 6)))
    expect(r.iterations).toBe(c.iterations)
    expect(r.converged).toBe(true)
    expect(r.conditioning).toBeCloseTo(c.condKnv, 6)
    const got = sortedPoles(toFlat(realPart(r.closedLoop)), toFlat(imagPart(r.closedLoop)))
    const want = sortedPoles(c.real, new Array(c.real.length).fill(0))
    got.forEach(([re, im], i) => {
      expect(re).toBeCloseTo(want[i][0], 8)
      expect(im).toBeCloseTo(0, 8)
    })
  })

  it.each(Object.keys(F))('places complex-conjugate poles with a well-conditioned X (%s)', (key) => {
    const c = F[key]
    const poles = c.complexRe.map((re, i) => ({ re, im: c.complexIm[i] }))
    const r = placePoles({ A: c.A, B: c.B }, poles)
    const got = sortedPoles(toFlat(realPart(r.closedLoop)), toFlat(imagPart(r.closedLoop)))
    const want = sortedPoles(c.complexRe, c.complexIm)
    got.forEach(([re, im], i) => {
      expect(re).toBeCloseTo(want[i][0], 7)
      expect(im).toBeCloseTo(want[i][1], 7)
    })
    expect(r.converged).toBe(true)
    // Within a small factor of scipy's YT method (which has its own complex update) on the same poles.
    expect(r.conditioning).toBeLessThan(1.25 * c.condYt)
    // The real form keeps K real: A − BK has the requested poles in conjugate pairs.
    expect(toFlat(r.K).every(Number.isFinite)).toBe(true)
  })

  it('maximising |det X| makes the placement robust: better conditioned than the starting eigenvectors', () => {
    const c = F.random6x3
    const one = placePoles({ A: c.A, B: c.B }, c.real, { maxIterations: 1, rtol: 0 })
    const many = placePoles({ A: c.A, B: c.B }, c.real, { maxIterations: 50 })
    expect(many.conditioning).toBeLessThanOrEqual(one.conditioning + 1e-9)
    expect(one.converged).toBe(false)
  })

  it('agrees with Ackermann for one input (the gain is unique)', () => {
    const A = [
      [0, 1, 0],
      [0, 0, 1],
      [-6, -11, -6],
    ]
    const B = [[0], [0], [1]]
    const poles = [-2, { re: -1, im: 1 }, { re: -1, im: -1 }]
    const r = placePoles({ A, B }, poles)
    const k = ackermann({ A, B }, poles)
    toFlat(r.K).forEach((v, i) => expect(v).toBeCloseTo(toFlat(k.K!)[i], 8))
    expect(r.iterations).toBe(0)
  })

  it('solves by least squares when B has full row rank', () => {
    const A = [
      [1, 2],
      [3, 4],
    ]
    const B = [
      [1, 0],
      [0, 1],
    ]
    const r = placePoles({ A, B }, [
      { re: -1, im: 2 },
      { re: -1, im: -2 },
    ])
    const got = sortedPoles(toFlat(realPart(r.closedLoop)), toFlat(imagPart(r.closedLoop)))
    expect(got[0][0]).toBeCloseTo(-1, 10)
    expect(got[0][1]).toBeCloseTo(-2, 10)
    expect(got[1][1]).toBeCloseTo(2, 10)
  })

  it('rejects unpaired complex poles, the wrong count and poles repeated beyond rank(B)', () => {
    const c = F.kautsky
    expect(() => placePoles({ A: c.A, B: c.B }, [{ re: -1, im: 1 }, -2, -3, -4])).toThrow(/conjugates/)
    expect(() => placePoles({ A: c.A, B: c.B }, [-1, -2, -3])).toThrow(/need 4 poles/)
    expect(() => placePoles({ A: c.A, B: c.B }, [-1, -1, -1, -2])).toThrow(/repeated/)
  })
})
