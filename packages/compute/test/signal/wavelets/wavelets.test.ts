import { describe, expect, it } from 'vitest'
import { cwt, dwt, idwt, morlet, wavedec, waveletFilters, wavefun, waverec } from 'aifn-compute/signal/wavelets'
import { signal } from 'aifn-compute/signal'
import { complexAbs, toFlat } from 'aifn-compute/foundation/tensor'
import { close } from '../helpers'

describe('wavelets', () => {
  it('filters are orthonormal with the stated vanishing moments', () => {
    for (const name of ['haar', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10'] as const) {
      const f = waveletFilters(name)
      const h = toFlat(f.recLo)
      const g = toFlat(f.recHi)
      expect(h.reduce((a, b) => a + b, 0)).toBeCloseTo(Math.SQRT2, 10)
      for (let k = 0; 2 * k < h.length; k++) {
        let s = 0
        for (let n = 2 * k; n < h.length; n++) s += h[n] * h[n - 2 * k]
        expect(s).toBeCloseTo(k === 0 ? 1 : 0, 10)
      }
      for (let p = 0; p < f.vanishingMoments; p++)
        expect(Math.abs(g.reduce((a, gv, n) => a + gv * n ** p, 0))).toBeLessThan(1e-7 * 10 ** p)
    }
  })

  it('dwt preserves energy and idwt / waverec invert exactly', () => {
    const x = Array.from({ length: 64 }, (_, i) => Math.sin(i / 3) + (i % 7) / 5)
    const one = dwt(x, 'db3')
    const energy = (v: number[]) => v.reduce((a, b) => a + b * b, 0)
    expect(energy(toFlat(one.approx)) + energy(toFlat(one.detail))).toBeCloseTo(energy(x), 8)
    close(idwt(one.approx, one.detail, 'db3'), x, 1e-11)
    const d = wavedec(signal(x, { fs: 8, t0: 1 }), 'db4', 3)
    expect(d.details.map((t) => t.shape[0])).toEqual([32, 16, 8])
    const back = waverec(d)
    expect(back.kind).toBe('signal')
    expect(back.fs).toBe(8)
    expect(back.t0).toBe(1)
    close(back.data, x, 1e-12)
  })

  it('db2 kills linear trends in the detail coefficients', () => {
    const x = Array.from({ length: 32 }, (_, i) => 3 + 0.5 * i)
    const d = toFlat(dwt(x, 'db2').detail)
    // Periodic wrap-around breaks the trend only in the last coefficient.
    for (let k = 0; k < d.length - 1; k++) expect(Math.abs(d[k])).toBeLessThan(1e-10)
  })

  it('the cascade gives a scaling function with unit integral', () => {
    const w = wavefun('db2', 8)
    const dt = toFlat(w.t)[1]
    expect(toFlat(w.phi).reduce((a, b) => a + b, 0) * dt).toBeCloseTo(1, 6)
  })

  it('Morlet is complex128 with a Gaussian envelope', () => {
    const m = morlet([0, 1, 2])
    expect(m.dtype).toBe('complex128')
    const a = toFlat(complexAbs(m))
    expect(a[1] / a[0]).toBeCloseTo(Math.exp(-0.5), 10)
    expect(a[2] / a[0]).toBeCloseTo(Math.exp(-2), 10)
  })

  it('the Morlet CWT of a tone peaks at its frequency', () => {
    const fs = 200
    const x = Array.from({ length: 512 }, (_, i) => Math.sin((2 * Math.PI * 20 * i) / fs))
    const freqs = [5, 10, 20, 40, 60]
    const c = cwt(x, freqs, { fs })
    expect(c.kind).toBe('time-frequency')
    expect(c.values.dtype).toBe('complex128')
    expect(c.values.shape).toEqual([5, 512])
    const mag = toFlat(c.magnitude)
    const mid = freqs.map((_, r) => mag[r * 512 + 256])
    expect(mid.indexOf(Math.max(...mid))).toBe(2)
  })
})
