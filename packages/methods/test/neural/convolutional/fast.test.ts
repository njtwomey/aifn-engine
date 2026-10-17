/**
 * aifn-methods/neural/convolutional fast convolution: Toom-Cook Winograd tiles equal the direct correlation, Lavin and
 * Gray's printed matrices pass, the float32 error grows with the tile, and the cost model gives the counts of the
 * fast-convolution-algorithms note (the site's self-check).
 */
import { describe, expect, it } from 'vitest'
import { AifnError } from 'aifn-compute/foundation/errors'
import { stream, units } from 'aifn-compute/foundation/random'
import {
  convolutionAlgorithmCosts,
  directTile,
  winogradPoints,
  winogradRelativeError,
  winogradTile,
  winogradTransforms,
  type Rows,
  type WinogradTransforms,
} from 'aifn-methods/neural/convolutional'

const s = stream('fast-convolution-check')
const rand = (n: number) => Array.from(units(s, n), (u) => 2 * u - 1)
const tileCheck = (t: WinogradTransforms) => {
  const d: Rows = Array.from({ length: t.n }, () => rand(t.n))
  const g: Rows = Array.from({ length: t.r }, () => rand(t.r))
  const a = winogradTile(d, g, t).flat()
  const b = directTile(d, g, t.m).flat()
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(1e-8 * Math.max(1, Math.abs(b[i]))))
}

describe('Winograd minimal filtering', () => {
  it.each([
    [2, 3],
    [4, 3],
    [6, 3],
    [3, 2],
    [2, 5],
    [4, 5],
    [1, 3],
    [3, 1],
  ])('F(%i x %i) by Toom-Cook equals the direct correlation', (m, r) => tileCheck(winogradTransforms(m, r)))

  it('gives F(2, 3) an integer data transform', () => {
    const t = winogradTransforms(2, 3)
    expect(t.BT.flat().every((v) => v === 0 || v === 1 || v === -1)).toBe(true)
    expect(t.points).toEqual([0, 1, -1])
    expect(winogradPoints(7)).toEqual([0, 1, -1, 2, -2, 0.5, -0.5])
  })

  it("passes with Lavin and Gray's printed F(2 x 2, 3 x 3) and F(4 x 4, 3 x 3)", () => {
    tileCheck({
      m: 2,
      r: 3,
      n: 4,
      points: [0, 1, -1],
      BT: [
        [1, 0, -1, 0],
        [0, 1, 1, 0],
        [0, -1, 1, 0],
        [0, 1, 0, -1],
      ],
      G: [
        [1, 0, 0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0, 0, 1],
      ],
      AT: [
        [1, 1, 1, 0],
        [0, 1, -1, -1],
      ],
    })
    tileCheck({
      m: 4,
      r: 3,
      n: 6,
      points: [0, 1, -1, 2, -2],
      BT: [
        [4, 0, -5, 0, 1, 0],
        [0, -4, -4, 1, 1, 0],
        [0, 4, -4, -1, 1, 0],
        [0, -2, -1, 2, 1, 0],
        [0, 2, -1, -2, 1, 0],
        [0, 4, 0, -5, 0, 1],
      ],
      G: [
        [1 / 4, 0, 0],
        [-1 / 6, -1 / 6, -1 / 6],
        [-1 / 6, 1 / 6, -1 / 6],
        [1 / 24, 1 / 12, 1 / 6],
        [1 / 24, -1 / 12, 1 / 6],
        [0, 0, 1],
      ],
      AT: [
        [1, 1, 1, 1, 1, 0],
        [0, 1, -1, 2, -2, 0],
        [0, 1, 1, 4, 4, 0],
        [0, 1, -1, 8, -8, 1],
      ],
    })
  })

  it('gives the worked example: outputs 0, 2, -1, -1 of a vertical-edge filter', () => {
    const d = [
      [1, 2, 0, 1],
      [0, 1, 3, 1],
      [2, 1, 0, 0],
      [1, 0, 1, 2],
    ]
    const g = [
      [1, 0, -1],
      [1, 0, -1],
      [1, 0, -1],
    ]
    expect(directTile(d, g, 2).flat()).toEqual([0, 2, -1, -1])
    winogradTile(d, g, winogradTransforms(2, 3))
      .flat()
      .forEach((v, i) => expect(v).toBeCloseTo([0, 2, -1, -1][i], 12))
  })

  it('loses float32 accuracy as the tile grows: 1.4e-7, 1.2e-6, 2.2e-6, 2.0e-4 against 8e-8 direct', () => {
    // The values the self-check prints, to four figures (1.4e-7, 1.2e-6, 2.2e-6, 2.0e-4 and about 8e-8 rounded).
    const want: Record<number, [number, number]> = {
      2: [1.3943e-7, 7.9642e-8],
      4: [1.1626e-6, 7.3637e-8],
      6: [2.2268e-6, 8.8011e-8],
      8: [1.9976e-4, 9.2304e-8],
    }
    for (const m of [2, 4, 6, 8]) {
      const e = winogradRelativeError(m, 3, stream(`err-${m}`))
      expect(Math.abs(e.winograd - want[m][0]) / want[m][0], `m = ${m}`).toBeLessThan(1e-3)
      expect(Math.abs(e.direct - want[m][1]) / want[m][1], `m = ${m}`).toBeLessThan(1e-3)
    }
  })

  it('throws for too few or repeated points', () => {
    expect(() => winogradTransforms(2, 3, [0, 1])).toThrow(AifnError)
    expect(() => winogradTransforms(2, 3, [0, 1, 1])).toThrow(AifnError)
    expect(() => winogradTransforms(0, 3)).toThrow(AifnError)
  })
})

describe('convolutionAlgorithmCosts', () => {
  it('counts a 3 x 3, 64 to 64 layer on 56 x 56', () => {
    const c = convolutionAlgorithmCosts({ height: 56, width: 56, inChannels: 64, outChannels: 64, kernel: 3 })
    expect(c.direct.multiplies).toBe(115_605_504)
    expect(c.im2col.multiplies).toBe(115_605_504)
    expect(c.direct.multiplies / c.winograd.elementwise).toBeCloseTo(2.25, 12)
    expect(c.fft.size).toEqual([64, 64])
    const c4 = convolutionAlgorithmCosts({ height: 56, width: 56, inChannels: 64, outChannels: 64, kernel: 3, tile: 4 })
    expect(c4.direct.multiplies / c4.winograd.elementwise).toBeCloseTo(4, 12)
    expect(c4.winograd.tiles).toBe(14 * 14)
  })
})
