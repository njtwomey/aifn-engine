/**
 * aifn-compute/signal/wavelets against PyWavelets (fixtures/gen/signal/wavelets.py): the filter banks, single- and multi-level
 * periodic DWTs (pywt's 'periodization' mode mapped to aifn's alignment, see the generator) and the cascade functions.
 */
import { describe, expect, it } from 'vitest'
import { dwt, wavedec, waveletFilters, wavefun, type WaveletName } from 'aifn-compute/signal/wavelets'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'

type Fx = {
  filters: Record<WaveletName, { decLo: number[]; decHi: number[]; recLo: number[]; recHi: number[] }>
  dwt: { wavelet: WaveletName; x: number[]; approx: number[]; detail: number[] }[]
  wavedec: { wavelet: WaveletName; x: number[]; levels: number; approx: number[]; details: number[][] }[]
  wavefun: { wavelet: WaveletName; iterations: number; t: number[]; phi: number[]; psi: number[] }[]
}
const F = fixture<Fx>('signal/wavelets')
const flat = (t: Tensor) => Array.from(toFlat(t))
const close = (a: number[], b: number[], tol = 1e-12) => {
  expect(a.length).toBe(b.length)
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i]))))
}

describe('wavelets against PyWavelets', () => {
  it.each(Object.entries(F.filters))('the %s filter bank', (name, f) => {
    const w = waveletFilters(name as WaveletName)
    close(flat(w.decLo), f.decLo, 1e-14)
    close(flat(w.decHi), f.decHi, 1e-14)
    close(flat(w.recLo), f.recLo, 1e-14)
    close(flat(w.recHi), f.recHi, 1e-14)
  })
  it.each(F.dwt.map((c) => [`${c.wavelet}, n = ${c.x.length}`, c] as const))('one level, %s', (_, c) => {
    const r = dwt(c.x, c.wavelet)
    close(flat(r.approx), c.approx)
    close(flat(r.detail), c.detail)
  })
  it.each(F.wavedec.map((c) => [`${c.wavelet}, ${c.levels} levels`, c] as const))('wavedec, %s', (_, c) => {
    const r = wavedec(c.x, c.wavelet, c.levels)
    close(flat(r.approx), c.approx, 1e-11)
    r.details.forEach((d, j) => close(flat(d), c.details[j], 1e-11))
  })
  it.each(F.wavefun.map((c) => [`${c.wavelet}, ${c.iterations} iterations`, c] as const))('wavefun, %s', (_, c) => {
    const r = wavefun(c.wavelet, c.iterations)
    close(flat(r.t), c.t, 1e-14)
    close(flat(r.phi), c.phi, 1e-12)
    close(flat(r.psi), c.psi, 1e-12)
  })
})
