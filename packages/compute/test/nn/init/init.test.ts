/** aifn-compute/nn/init: the initialisers' variances (Glorot and Bengio, 2010; He et al., 2015). */
import { describe, expect, it } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  heNormal,
  heUniform,
  lecunUniform,
  normalInit,
  xavierNormal,
  xavierUniform,
  zerosInit,
} from 'aifn-compute/nn/init'

const variance = (v: ArrayLike<number>) => Array.from(v).reduce((a, b) => a + b * b, 0) / v.length

describe('initialisers', () => {
  it('initialisers have their variances', () => {
    const w = xavierUniform()(stream('x'), [200, 300], { fanIn: 200, fanOut: 300 })
    const v = toFlat(w).reduce((a, b) => a + b * b, 0) / (200 * 300)
    expect(v).toBeCloseTo(2 / 500, 4)
    const h = heUniform()(stream('h'), [400, 50], { fanIn: 400, fanOut: 50 })
    const vh = toFlat(h).reduce((a, b) => a + b * b, 0) / (400 * 50)
    expect(Math.abs(vh / (2 / 400) - 1)).toBeLessThan(0.05)
  })

  it('normal and LeCun initialisers, zeros, and reproducibility', () => {
    const fans = { fanIn: 400, fanOut: 100 }
    const rel = (v: number, e: number) => Math.abs(v / e - 1)
    expect(rel(variance(toFlat(xavierNormal()(stream('xn'), [400, 100], fans))), 2 / 500)).toBeLessThan(0.05)
    expect(rel(variance(toFlat(heNormal()(stream('hn'), [400, 100], fans))), 2 / 400)).toBeLessThan(0.05)
    expect(rel(variance(toFlat(lecunUniform()(stream('lu'), [400, 100], fans))), 1 / (3 * 400))).toBeLessThan(0.05)
    expect(rel(variance(toFlat(normalInit(0.1)(stream('n'), [400, 100], fans))), 0.01)).toBeLessThan(0.05)
    expect(toFlat(zerosInit()(stream('z'), [3, 2], { fanIn: 3, fanOut: 2 }))).toEqual([0, 0, 0, 0, 0, 0])
    expect(toFlat(xavierUniform()(stream('r'), [4, 4], { fanIn: 4, fanOut: 4 }))).toEqual(
      toFlat(xavierUniform()(stream('r'), [4, 4], { fanIn: 4, fanOut: 4 })),
    )
  })
})
