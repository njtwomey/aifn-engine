import { describe, expect, it } from 'vitest'
import {
  halton,
  latinHypercube,
  maxGap,
  randomShift,
  scrambledHalton,
  scrambledSobol,
  sobol,
  starDiscrepancy,
} from 'aifn-compute/numerics/quadrature'
import { fork, stream } from 'aifn-compute/foundation/random'
import { toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { fixture } from '../../fixtures'

type Fixture = { designs: { starDiscrepancy: Record<string, { points: number[][]; value: number }> } }
const F = fixture<Fixture>('numerics/quadrature').designs

const rows = (t: Tensor) => toRows(t) as number[][]
const column = (x: number[][], j: number) => x.map((p) => p[j])

/** Whether each of the n cells [k/n, (k + 1)/n) holds exactly one of the n values. */
function stratified(values: number[], n: number): boolean {
  const cells = values.map((v) => Math.floor(v * n)).sort((a, b) => a - b)
  return cells.length === n && cells.every((c, k) => c === k)
}

/** Whether the points are a (t, m, 2)-net in base 2 on coordinates a and b. */
function isNet(x: number[][], m: number, t: number, [a, b]: [number, number]): boolean {
  for (let k1 = 0; k1 <= m - t; k1++) {
    const k2 = m - t - k1
    const counts = new Map<number, number>()
    for (const p of x) {
      const key = Math.floor(p[a] * 2 ** k1) * 2 ** k2 + Math.floor(p[b] * 2 ** k2)
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    if (counts.size !== 2 ** (m - t) || [...counts.values()].some((c) => c !== 2 ** t)) return false
  }
  return true
}

/** The star discrepancy by counting every box corner of the grid (the definition, for any d). */
function bruteForce(x: number[][]): number {
  const n = x.length
  const d = x[0].length
  const axes = Array.from({ length: d }, (_, j) => [...new Set([...column(x, j), 1])].sort((p, q) => p - q))
  let best = 0
  const visit = (j: number, t: number[]) => {
    if (j === d) {
      const volume = t.reduce((a, v) => a * v, 1)
      const open = x.filter((p) => p.every((v, i) => v < t[i])).length
      const closed = x.filter((p) => p.every((v, i) => v <= t[i])).length
      best = Math.max(best, volume - open / n, closed / n - volume)
      return
    }
    for (const v of axes[j]) visit(j + 1, [...t, v])
  }
  visit(0, [])
  return best
}

describe('latinHypercube', () => {
  it('puts one point in each slice of each axis', () => {
    const x = rows(latinHypercube(stream(1), 37, 5))
    for (let j = 0; j < 5; j++) expect(stratified(column(x, j), 37)).toBe(true)
    expect(x.flat().every((v) => v >= 0 && v < 1)).toBe(true)
    // Jittered points are not at the centres; centred ones are, exactly.
    expect(x.flat().some((v) => Math.abs(((v * 37) % 1) - 0.5) > 1e-6)).toBe(true)
    const c = rows(latinHypercube(stream(1), 8, 3, { centred: true }))
    for (let j = 0; j < 3; j++)
      expect(column(c, j).sort((a, b) => a - b)).toEqual(Array.from({ length: 8 }, (_, k) => (k + 0.5) / 8))
  })

  it('draws from its stream: the same stream gives the same design, and the stream advances', () => {
    const s = stream(2)
    expect(rows(latinHypercube(fork(s), 10, 2))).toEqual(rows(latinHypercube(fork(s), 10, 2)))
    const first = rows(latinHypercube(s, 10, 2))
    expect(rows(latinHypercube(s, 10, 2))).not.toEqual(first)
    expect(latinHypercube(stream(0), 0, 3).shape).toEqual([0, 3])
    expect(() => latinHypercube(stream(0), 2.5, 2)).toThrow(DomainError)
    expect(() => latinHypercube(stream(0), 4, 0)).toThrow(DomainError)
  })
})

describe('randomShift', () => {
  it('adds one vector modulo 1 to every point', () => {
    const p = rows(sobol(16, 3))
    const x = rows(randomShift(stream(3), p))
    const shift = x[0].map((v, j) => (v - p[0][j] + 1) % 1)
    for (let i = 0; i < 16; i++)
      for (let j = 0; j < 3; j++) {
        expect(x[i][j]).toBeGreaterThanOrEqual(0)
        expect(x[i][j]).toBeLessThan(1)
        expect((x[i][j] - p[i][j] - shift[j] + 2) % 1).toBeCloseTo(0, 14)
      }
    // A shifted stratified design stays stratified on the torus: gaps are kept.
    for (let j = 0; j < 3; j++) {
      const sorted = column(x, j).sort((a, b) => a - b)
      const gaps = sorted.map((v, i) => (i === 0 ? v + 1 - sorted[15] : v - sorted[i - 1]))
      for (const g of gaps) expect(g).toBeCloseTo(1 / 16, 14)
    }
  })
})

describe('scrambledSobol', () => {
  const m = 7
  const plain = rows(sobol(2 ** m, 8))
  /** The t of the unscrambled Sobol points on coordinates a, b. */
  const tOf = (a: number, b: number) => {
    let t = 0
    while (!isNet(plain, m, t, [a, b])) t++
    return t
  }

  it.each(['lms', 'owen'] as const)('%s keeps the net property of every pair of coordinates', (scramble) => {
    const x = rows(scrambledSobol(stream(4), 2 ** m, 8, { scramble }))
    expect(x.flat().every((v) => v >= 0 && v < 1)).toBe(true)
    expect(isNet(x, m, 0, [0, 1])).toBe(true)
    for (let a = 0; a < 8; a++) {
      expect(stratified(column(x, a), 2 ** m)).toBe(true)
      for (let b = a + 1; b < 8; b++) expect(isNet(x, m, tOf(a, b), [a, b])).toBe(true)
    }
    // Randomised: not the plain points, and a different stream gives different points.
    expect(x[0]).not.toEqual(plain[0])
    expect(rows(scrambledSobol(stream(5), 4, 8, { scramble }))[0]).not.toEqual(x[0])
  })

  it.each(['lms', 'owen'] as const)('%s: the first points of a longer design are the shorter design', (scramble) => {
    const s = stream(6)
    const long = rows(scrambledSobol(fork(s), 32, 5, { scramble }))
    expect(rows(scrambledSobol(fork(s), 8, 5, { scramble }))).toEqual(long.slice(0, 8))
    expect(rows(scrambledSobol(fork(s), 8, 5, { scramble, skip: 8 }))).toEqual(long.slice(8, 16))
  })

  it('each Owen-scrambled point is uniform: the first point over many streams', () => {
    const R = 4000
    const first = Array.from(
      { length: R },
      (_, r) => rows(scrambledSobol(stream(100 + r), 1, 2, { scramble: 'owen' }))[0],
    )
    for (let j = 0; j < 2; j++) {
      const mean = first.reduce((a, p) => a + p[j], 0) / R
      expect(Math.abs(mean - 0.5)).toBeLessThan(4 * Math.sqrt(1 / 12 / R))
      // The quarter it lands in is uniform too.
      const quarters = [0, 0, 0, 0]
      for (const p of first) quarters[Math.floor(4 * p[j])]++
      for (const q of quarters) expect(Math.abs(q / R - 0.25)).toBeLessThan(4 * Math.sqrt(0.1875 / R))
    }
  })

  it('checks its arguments', () => {
    expect(() => scrambledSobol(stream(0), 4, 22)).toThrow(ShapeError)
    expect(() => scrambledSobol(stream(0), 4, 2, { scramble: 'nested' as 'owen' })).toThrow(DomainError)
    expect(() => scrambledSobol(stream(0), 4, 2, { skip: -1 })).toThrow(DomainError)
  })
})

describe('scrambledHalton', () => {
  it("keeps each coordinate's stratification in its base", () => {
    const x = rows(scrambledHalton(stream(7), 125, 4))
    expect(stratified(column(x.slice(0, 64), 0), 64)).toBe(true)
    expect(stratified(column(x.slice(0, 81), 1), 81)).toBe(true)
    expect(stratified(column(x, 2), 125)).toBe(true)
    expect(stratified(column(x.slice(0, 49), 3), 49)).toBe(true)
    expect(x.flat().every((v) => v >= 0 && v < 1)).toBe(true)
    const plain = rows(halton(125, 4))
    expect(x[1]).not.toEqual(plain[1])
  })

  it('is a prefix of a longer design, and skip continues it', () => {
    const s = stream(8)
    const long = rows(scrambledHalton(fork(s), 30, 3))
    expect(rows(scrambledHalton(fork(s), 10, 3))).toEqual(long.slice(0, 10))
    expect(rows(scrambledHalton(fork(s), 10, 3, { skip: 20 }))).toEqual(long.slice(20))
    expect(() => scrambledHalton(stream(0), 4, 26)).toThrow(ShapeError)
  })
})

describe('starDiscrepancy', () => {
  it.each(Object.entries(F.starDiscrepancy))('%s matches the brute force of the generator', (_, c) => {
    expect(starDiscrepancy(c.points)).toBeCloseTo(c.value, 14)
  })

  it('agrees with counting every corner, and has the closed forms of one dimension', () => {
    for (const x of [rows(scrambledSobol(stream(9), 24, 2)), rows(latinHypercube(stream(10), 17, 2))])
      expect(starDiscrepancy(x)).toBeCloseTo(bruteForce(x), 14)
    expect(starDiscrepancy([[0.125], [0.375], [0.625], [0.875]])).toBeCloseTo(0.125, 15)
    expect(starDiscrepancy([[0], [0.5]])).toBeCloseTo(0.5, 15)
    expect(starDiscrepancy([[1, 1]])).toBe(1)
    expect(() => starDiscrepancy([[0.5, 1.2]])).toThrow(DomainError)
    expect(() => starDiscrepancy(rows(sobol(64, 5)))).toThrow(DomainError)
  })
})

describe('maxGap', () => {
  it('takes the ends of the interval into account', () => {
    expect(maxGap(Array.from({ length: 16 }, (_, i) => i / 16))).toBe(1 / 16)
    expect(maxGap([0.9, 0.1, 0.5])).toBeCloseTo(0.4, 15)
    expect(maxGap([0.2], { lo: -1, hi: 1 })).toBeCloseTo(1.2, 15)
    expect(maxGap([])).toBe(1)
    expect(() => maxGap([1.5])).toThrow(DomainError)
    expect(() => maxGap([0.5], { lo: 1, hi: 0 })).toThrow(DomainError)
  })
})
