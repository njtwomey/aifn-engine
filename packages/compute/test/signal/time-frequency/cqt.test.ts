import { describe, expect, it } from 'vitest'
import { cqt } from 'aifn-compute/signal/time-frequency'
import { magnitude, signal } from 'aifn-compute/signal'
import { imagPart, realPart, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Case = {
  binsPerOctave: number
  bins: number
  hop: number
  window: 'hamming' | 'hann'
  f: number[]
  re: number[][]
  im: number[][]
}
const F = fixture<{ x: number[]; fs: number; fmin: number; cqt: Record<string, Case> }>('signal/time-frequency')

const close = (got: ArrayLike<number>, want: number[], tol: number) => {
  expect(got.length).toBe(want.length)
  want.forEach((w, i) => expect(Math.abs(got[i] - w), `[${i}] ${got[i]} vs ${w}`).toBeLessThanOrEqual(tol))
}

describe('constant-Q transform', () => {
  for (const [name, c] of Object.entries(F.cqt))
    it(`${name}: the FFT-kernel computation equals the direct sum`, () => {
      const tf = cqt(signal(F.x, { fs: F.fs }), {
        fmin: F.fmin,
        binsPerOctave: c.binsPerOctave,
        bins: c.bins,
        hop: c.hop,
        window: c.window,
      })
      expect(tf.kind).toBe('time-frequency')
      expect(tf.method).toBe('cqt')
      expect(tf.frequencyScale).toBe('log')
      expect(tf.values.shape).toEqual([c.bins, c.re[0].length])
      close(toFlat(tf.f), c.f, 1e-9)
      close(toFlat(realPart(tf.values)), c.re.flat(), 1e-12)
      close(toFlat(imagPart(tf.values)), c.im.flat(), 1e-12)
      close(toFlat(tf.t).slice(0, 3), [0, c.hop / F.fs, (2 * c.hop) / F.fs], 1e-15)
    })
  it('the tones peak in their semitone bins', () => {
    const tf = cqt(signal(F.x, { fs: F.fs }), { fmin: F.fmin, bins: 48, hop: 160 })
    const rows = toRows(magnitude(tf.values))
    const middle = Math.floor(rows[0].length / 2)
    const column = rows.map((r) => r[middle])
    const order = column.map((v, k) => [v, k] as const).sort((a, b) => b[0] - a[0])
    expect(
      order
        .slice(0, 2)
        .map(([, k]) => k)
        .sort((a, b) => a - b),
    ).toEqual([12, 31])
    // A unit sinusoid on a bin centre has magnitude about ½·mean(w) (Hamming: 0.27).
    expect(column[12]).toBeGreaterThan(0.24)
    expect(column[12]).toBeLessThan(0.3)
  })
  it('sparse spectral kernels approximate the dense result', () => {
    const opts = { fmin: F.fmin, bins: 48, hop: 160 }
    const dense = toFlat(magnitude(cqt(signal(F.x, { fs: F.fs }), opts).values))
    const sparse = toFlat(magnitude(cqt(signal(F.x, { fs: F.fs }), { ...opts, sparsity: 0.0054 }).values))
    const peak = Math.max(...dense)
    dense.forEach((v, i) => expect(Math.abs(v - sparse[i])).toBeLessThan(0.02 * peak))
  })
})
