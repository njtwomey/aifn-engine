import { describe, expect, it } from 'vitest'
import { decimateSignal, dftFilterBank, polyphase, resamplePoly, upfirdn } from 'aifn-compute/signal/multirate'
import { signal } from 'aifn-compute/signal'
import { imagPart, realPart, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Fixture = {
  x: number[]
  h: number[]
  upfirdn: { up: number; down: number; y: number[] }[]
  resamplePoly: {
    cases: { up: number; down: number; y: number[] }[]
    variants: Record<
      string,
      { up: number; down: number; window?: 'hamming'; taps?: number[]; padtype?: 'mean'; y: number[] }
    >
  }
  long: number[]
  decimateSignal: { q: number; ftype: 'iir' | 'fir'; zeroPhase: boolean; n?: number; y: number[] }[]
  dftFilterBank: { x: number[]; prototype: number[]; channels: number; re: number[][]; im: number[][] }
  polyphase: { h: number[]; branches: number; components: number[][] }
}
const F = fixture<Fixture>('signal/multirate')

const close = (got: ArrayLike<number>, want: number[], tol = 1e-12) => {
  expect(got.length).toBe(want.length)
  want.forEach((w, i) => expect(Math.abs(got[i] - w), `[${i}] ${got[i]} vs ${w}`).toBeLessThanOrEqual(tol))
}

describe('multirate matches scipy.signal', () => {
  it.each(F.upfirdn.map((c) => [`${c.up}/${c.down}`, c] as const))('upfirdn %s', (_, c) =>
    close(toFlat(upfirdn(tensor(F.h), tensor(F.x), { up: c.up, down: c.down }) as Tensor), c.y),
  )
  it.each(F.resamplePoly.cases.map((c) => [`${c.up}/${c.down}`, c] as const))('resamplePoly %s', (_, c) => {
    const y = resamplePoly(signal(F.x, { fs: 12 }), c.up, c.down)
    close(toFlat(y.data), c.y)
    expect(y.fs).toBeCloseTo((12 * c.up) / c.down, 12)
  })
  for (const [name, v] of Object.entries(F.resamplePoly.variants))
    it(`resamplePoly with ${name}`, () => {
      const window = v.taps ?? v.window
      close(toFlat(resamplePoly(F.x, v.up, v.down, { window, padtype: v.padtype }).data), v.y)
    })
  it.each(F.decimateSignal.map((c) => [`q=${c.q} ${c.ftype} zeroPhase=${c.zeroPhase} n=${c.n ?? '-'}`, c] as const))(
    'decimateSignal %s',
    (_, c) => {
      const y = decimateSignal(signal(F.long, { fs: 100 }), c.q, { ftype: c.ftype, zeroPhase: c.zeroPhase, n: c.n })
      close(toFlat(y.data), c.y, 1e-10)
      expect(y.fs).toBeCloseTo(100 / c.q, 12)
    },
  )
})

describe('polyphase filter banks', () => {
  it('polyphase components', () => {
    const p = F.polyphase
    expect(toRows(polyphase(p.h, p.branches))).toEqual(p.components)
  })
  it('the DFT analysis bank by polyphase equals each modulated channel filtered and decimated', () => {
    const b = F.dftFilterBank
    const y = dftFilterBank(b.x, b.prototype, b.channels)
    expect(y.shape).toEqual([b.re.length, b.re[0].length])
    close(toFlat(realPart(y)), b.re.flat(), 1e-12)
    close(toFlat(imagPart(y)), b.im.flat(), 1e-12)
  })
})
