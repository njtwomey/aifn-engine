import { describe, expect, it } from 'vitest'
import { envelope, hilbert, hilbertSpectrum, instantaneous } from 'aifn-compute/signal/time-frequency'
import { sampleTimes, signal } from 'aifn-compute/signal'
import { realPart, toFlat } from 'aifn-compute/foundation/tensor'
import { close, closeC, F } from '../helpers'

describe('analytic signal', () => {
  it('hilbert matches scipy for even and odd lengths, as complex128', () => {
    closeC(hilbert(F.hilbert.even.x), F.hilbert.even.z, 1e-11)
    closeC(hilbert(F.hilbert.odd.x), F.hilbert.odd.z, 1e-11)
    close(realPart(hilbert(F.hilbert.odd.x)), F.hilbert.odd.x, 1e-12)
  })

  it('an AM tone has its modulation as envelope and its carrier as frequency', () => {
    const t = toFlat(sampleTimes(signal(new Float64Array(1024), { fs: 1024 })))
    const x = t.map((s) => (1 + 0.5 * Math.cos(2 * Math.PI * 4 * s)) * Math.cos(2 * Math.PI * 100 * s))
    const env = toFlat(envelope(x))
    const inst = toFlat(instantaneous(x, { fs: 1024 }).frequency)
    for (let i = 200; i < 800; i += 50) {
      expect(env[i]).toBeCloseTo(1 + 0.5 * Math.cos(2 * Math.PI * 4 * t[i]), 2)
      expect(inst[i]).toBeCloseTo(100, 0)
    }
  })

  it('the Hilbert spectrum puts a tone at its frequency', () => {
    const fs = 256
    const x = Array.from({ length: 512 }, (_, i) => Math.sin((2 * Math.PI * 32 * i) / fs))
    const h = hilbertSpectrum([x], { fs, freqBins: 64, timeBins: 16, fMax: 64 })
    expect(h.kind).toBe('time-frequency')
    expect(h.method).toBe('hht')
    const f = toFlat(h.f)
    const marginal = toFlat(h.marginal)
    expect(f[marginal.indexOf(Math.max(...marginal))]).toBeCloseTo(32, -1)
  })
})
