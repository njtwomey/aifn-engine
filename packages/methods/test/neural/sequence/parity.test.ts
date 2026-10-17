/**
 * aifn-methods/neural/sequence finite-precision parity (the expressiveness note's checks): truncation toward zero, the
 * string 1011 under a flip, a flip exact and no decay at chance at every length, halving at 8 bits against the closed
 * form, and the threshold readout against brute force over all strings.
 */
import { describe, expect, it } from 'vitest'
import {
  onesUntilZero,
  parityAccuracyByLength,
  parityStates,
  roundToGrid,
  stateValues,
} from 'aifn-methods/neural/sequence'

const binom = (T: number, c: number) => {
  let r = 1
  for (let i = 0; i < c; i++) r = (r * (T - i)) / (i + 1)
  return r / 2 ** T
}

describe('finite-precision parity', () => {
  it('truncates toward zero and reads 1011 under lambda = -1 as -1, -1, 1, -1', () => {
    const rounded = [roundToGrid(0.9, 0), roundToGrid(1.7, 0), roundToGrid(-0.7, 0), roundToGrid(-1.5, 1)]
    expect(rounded).toEqual([0, 1, 0, -1.5])
    expect(Object.is(roundToGrid(-0.1, 2), 0)).toBe(true)
    expect(parityStates([1, 0, 1, 1], -1, 8)).toEqual([-1, -1, 1, -1])
  })

  it('tracks parity exactly with a flip, and stays at chance without decay, to T = 200', () => {
    for (const readout of ['threshold', 'any'] as const) {
      const flip = parityAccuracyByLength({ lambda: -1, precision: 8, maxLength: 200, readout }).accuracy
      flip.forEach((a) => expect(Math.abs(a - 1)).toBeLessThan(1e-12))
      const keep = parityAccuracyByLength({ lambda: 1, precision: 8, maxLength: 200, readout }).accuracy
      keep.forEach((a) => expect(Math.abs(a - 0.5)).toBeLessThan(1e-12))
    }
  })

  it('halves to 2^-8 then 0 at 8 bits, exact to T = 8, and matches the closed form at T = 40', () => {
    const { accuracy, values } = parityAccuracyByLength({ lambda: 0.5, precision: 8, maxLength: 64, readout: 'any' })
    expect(values[8]).toBe(2 ** -8)
    expect(values[9]).toBe(0)
    expect(stateValues(0.5, 8, 9)).toEqual(values.slice(0, 10))
    expect(onesUntilZero(0.5, 8)).toBe(9)
    expect(onesUntilZero(-1, 8, 100)).toBe(Infinity)
    accuracy.slice(0, 8).forEach((a) => expect(Math.abs(a - 1)).toBeLessThan(1e-12))
    const T = 40
    let low = 0
    for (let c = 0; c <= 8; c++) low += binom(T, c)
    let even = 0
    let odd = 0
    for (let c = 9; c <= T; c++) {
      if (c % 2) odd += binom(T, c)
      else even += binom(T, c)
    }
    expect(Math.abs(accuracy[T - 1] - (low + Math.max(even, odd)))).toBeLessThan(1e-12)
  })

  it('gives the threshold readout found by brute force over all strings, T <= 10', () => {
    for (const lambda of [0.5, -0.5, 0.9, -0.7]) {
      const acc = parityAccuracyByLength({ lambda, precision: 4, maxLength: 10, readout: 'threshold' }).accuracy
      for (let T = 1; T <= 10; T++) {
        const states: [number, number][] = []
        for (let m = 0; m < 2 ** T; m++) {
          const bits = Array.from({ length: T }, (_, i) => (m >> i) & 1)
          const h = parityStates(bits, lambda, 4).at(-1)!
          states.push([h === 0 ? 0 : h, bits.reduce((a, b) => a + b, 0) % 2])
        }
        const thresholds = [-Infinity, ...new Set(states.map((s) => s[0]))].sort((a, b) => a - b)
        let best = 0
        for (const t of thresholds)
          for (const dir of [0, 1]) {
            const right = states.filter(([h, y]) => (h > t ? 1 - dir : dir) === y).length
            best = Math.max(best, right / states.length)
          }
        expect(Math.abs(best - acc[T - 1])).toBeLessThan(1e-12)
      }
    }
  })
})
