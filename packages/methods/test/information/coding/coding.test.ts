/** Prefix codes, arithmetic coding intervals, Hamming distances and code bounds, hand-checked. */
import { describe, expect, it } from 'vitest'
import {
  arithmeticInterval,
  hammingBound,
  hammingDistance,
  hammingWeight,
  huffmanCode,
  kraftSum,
  minimumDistance,
  plotkinBound,
  shannonCode,
  shannonFanoCode,
  singletonBound,
} from 'aifn-methods/information/coding'

const close = (a: number, b: number, tol = 1e-12) =>
  expect(Math.abs(a - b) / Math.max(1, Math.abs(b))).toBeLessThan(tol)

describe('info: coding', () => {
  it('Huffman, Shannon–Fano, Shannon and Kraft', () => {
    const p = [0.25, 0.25, 0.2, 0.15, 0.15]
    const h = huffmanCode(p)
    expect(h.codewords).toEqual(['01', '10', '00', '110', '111'])
    close(h.expectedLength, 2.3)
    expect(h.expectedLength).toBeGreaterThanOrEqual(h.entropy)
    close(kraftSum(h.lengths), 1)
    expect(shannonFanoCode(p).codewords).toEqual(['00', '01', '10', '110', '111'])
    expect(shannonCode(p).codewords).toEqual(['00', '01', '100', '101', '110'])
    expect(huffmanCode([1]).codewords).toEqual(['0'])
  })

  it('arithmetic interval, Hamming distances and bounds', () => {
    const a = arithmeticInterval([0, 1, 0], [0.8, 0.2])
    close(a.low, 0.64)
    close(a.width, 0.8 * 0.2 * 0.8)
    expect(a.bits).toBe(4)
    expect(hammingDistance('10110', '11100')).toBe(2)
    expect(hammingWeight([1, 0, 1, 1])).toBe(3)
    expect(minimumDistance(['0000000', '1101001', '0101010', '1000011'])).toBe(3)
    // The [7, 4, 3] Hamming code is perfect: it attains the sphere-packing bound.
    expect(hammingBound(7, 3)).toBe(16)
    expect(singletonBound(7, 3)).toBe(32)
    expect(plotkinBound(7, 3)).toBe(16)
    expect(plotkinBound(10, 6)).toBe(6)
    expect(plotkinBound(20, 3)).toBe(Infinity)
  })
})
