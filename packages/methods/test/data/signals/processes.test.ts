import { describe, expect, test } from 'vitest'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { coherence, logSpectralError, welch } from 'aifn-compute/signal/spectral'
import { arProcess, armaProcess, coupledProcesses, sinusoidsInNoise, unevenSinusoids } from 'aifn-methods/data/signals'
import type { SpectralTruth } from 'aifn-methods/data'

const truthOf = (d: { meta: { truth?: unknown } }) => d.meta.truth as SpectralTruth

describe('series with a known spectrum', () => {
  test('a long Welch estimate of an AR(2) and an ARMA(2, 2) sits on the true PSD', () => {
    for (const d of [arProcess(stream(1), { n: 1 << 15, radius: 0.9 }), armaProcess(stream(2), { n: 1 << 15 })]) {
      const w = welch(d.signal, { nperseg: 256 })
      const e = logSpectralError(w, toFlat(truthOf(d).psd(w.f)), { band: [0.01, 0.49] })
      // Many segments: little variance; the Hann window's smoothing bias stays under a dB on average.
      expect(Math.abs(e.bias)).toBeLessThan(0.5)
      expect(e.sd).toBeLessThan(1)
    }
  })
  test('the noise variance is the integral of the PSD and matches the sample variance', () => {
    const d = arProcess(stream(3), { n: 1 << 16, radius: 0.8, frequency: 0.1 })
    const t = truthOf(d)
    const x = toFlat(d.signal.data)
    const v = x.reduce((a, b) => a + b * b, 0) / x.length
    expect(v / t.noiseVariance).toBeGreaterThan(0.95)
    expect(v / t.noiseVariance).toBeLessThan(1.05)
  })
  test('sinusoids carry their lines and the white floor 2σ²/fs', () => {
    const d = sinusoidsInNoise(stream(4), { noise: 0.5, fs: 10 })
    const t = truthOf(d)
    expect(t.lines.map((l) => l.frequency)).toEqual([0.1, 0.13])
    expect(toFlat(t.psd([1]))[0]).toBeCloseTo((2 * 0.25) / 10, 12)
  })
  test('the coupled pair has the coherence of its truth', () => {
    const d = coupledProcesses(stream(5), { n: 1 << 15 })
    const c = coherence(d.signal, d.partner!, { nperseg: 128 })
    const truth = toFlat(truthOf(d).coupled!.coherence(c.f))
    const est = toFlat(c.values)
    const err = est.reduce((a, v, i) => a + Math.abs(v - truth[i]), 0) / est.length
    expect(err).toBeLessThan(0.03)
  })
  test('seasonal sampling keeps to the observing season, one sample per night', () => {
    const d = unevenSinusoids(stream(6), { sampling: 'seasonal', span: 730, n: 1000 })
    const t = toFlat(d.x)
    for (const v of t) expect((v % 365.25) / 365.25 <= 0.61).toBe(true)
    expect(new Set(t.map((v) => Math.floor(v))).size).toBe(t.length)
  })
})
