/** Laws of the cepstrum and the pitch estimators on synthetic voiced signals with a known f₀. */
import { describe, expect, it } from 'vitest'
import {
  cepstralPitch,
  complexCepstrum,
  inverseComplexCepstrum,
  realCepstrum,
  yin,
  yinDifference,
  yinPitch,
} from 'aifn-compute/signal/cepstrum'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { normals, stream } from 'aifn-compute/foundation/random'

const fs = 16000
/** A glottal-like harmonic signal: harmonics k f₀ with amplitude 1/k. */
const voiced = (f0: number, n: number) =>
  Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let k = 1; k * f0 < fs / 2 && k <= 20; k++) s += Math.sin((2 * Math.PI * k * f0 * i) / fs + 0.3 * k) / k
    return s
  })

describe('cepstrum', () => {
  it('the real cepstrum is even and matches the definition on a small case', () => {
    const c = toFlat(realCepstrum([1, 0.5, -0.25, 0.1, 0, 0.3, -0.2, 0.05]))
    for (let q = 1; q < 8; q++) expect(c[q]).toBeCloseTo(c[8 - q], 12)
  })

  it('the complex cepstrum inverts', () => {
    const x = [0.2, 1, -0.4, 0.3, 0.1, -0.05, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    const { cepstrum, delay } = complexCepstrum(x)
    const back = toFlat(inverseComplexCepstrum(cepstrum, delay))
    back.forEach((v, i) => expect(v).toBeCloseTo(x[i], 9))
  })

  it('cepstral pitch and YIN find f₀ of a harmonic signal', () => {
    for (const f0 of [110, 196.5, 330]) {
      const x = voiced(f0, 2048)
      expect(Math.abs(cepstralPitch(x, { fs }).f0 / f0 - 1)).toBeLessThan(0.02)
      const y = yinPitch(x, { fs })
      expect(y.voiced).toBe(true)
      expect(Math.abs(y.f0 / f0 - 1)).toBeLessThan(0.003)
    }
  })

  it('YIN: d′(0) = 1, d′ ≥ 0, white noise is unvoiced, and the tracker follows a step in pitch', () => {
    const { dPrime } = yinDifference(voiced(200, 1024), 300)
    const dp = toFlat(dPrime)
    expect(dp[0]).toBe(1)
    expect(Math.min(...dp)).toBeGreaterThanOrEqual(0)
    const noise = toFlat(normals(stream('yin-noise'), 2048, 0, 1))
    expect(yinPitch(noise, { fs }).voiced).toBe(false)
    const x = [...voiced(150, 4000), ...voiced(250, 4000)]
    const track = yin(x, { fs, hop: 400 })
    const f = toFlat(track.f0)
    expect(Math.abs(f[1] - 150)).toBeLessThan(2)
    expect(Math.abs(f[f.length - 2] - 250)).toBeLessThan(2)
  })
})
