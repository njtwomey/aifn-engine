/** Laws of the Wigner–Ville distributions and the sharpened spectrograms (no Python reference implements them). */
import { describe, expect, it } from 'vitest'
import {
  hilbert,
  pseudoWignerVille,
  reassignedSpectrogram,
  smoothedPseudoWignerVille,
  synchrosqueeze,
  wignerVille,
} from 'aifn-compute/signal/time-frequency'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

const fs = 256
const n = 256
const chirp = Array.from({ length: n }, (_, i) => Math.cos(2 * Math.PI * (20 * (i / fs) + 40 * (i / fs) ** 2)))
const twoTones = Array.from(
  { length: n },
  (_, i) => Math.cos((2 * Math.PI * 30 * i) / fs) + Math.cos((2 * Math.PI * 90 * i) / fs),
)
/** Column t of a [f, t] raster. */
const column = (v: Tensor, t: number) => {
  const [F, T] = v.shape
  const d = toFlat(v)
  return Array.from({ length: F }, (_, k) => d[k * T + t])
}
const argmax = (a: number[]) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0)

describe('Wigner–Ville', () => {
  it('the time marginal is |z(t)|² and a chirp concentrates on its instantaneous frequency', () => {
    const w = wignerVille(chirp, { fs, nfft: 256 })
    const z = toFlat(hilbert(chirp))
    for (const t of [40, 128, 200]) {
      const col = column(w.values, t)
      const marginal = col.reduce((s, v) => s + v, 0) / 256
      expect(marginal).toBeCloseTo(z[2 * t] ** 2 + z[2 * t + 1] ** 2, 8)
      // f(t) = 20 + 80 t.
      expect(toFlat(w.f)[argmax(col)]).toBeCloseTo(20 + (80 * t) / fs, -0.5)
    }
  })

  it('two tones interfere half way between them; smoothing removes most of it', () => {
    const at = (s: { values: Tensor; f: Tensor }, f: number) => {
      const fi = argmax(toFlat(s.f).map((v) => -Math.abs(v - f)))
      return Math.max(...column(s.values, 128).map((_, k, c) => (k === fi ? Math.abs(c[k]) : 0)))
    }
    const plain = wignerVille(twoTones, { fs, nfft: 256 })
    const pseudo = pseudoWignerVille(twoTones, { fs, nfft: 256, lagLength: 63 })
    const smooth = smoothedPseudoWignerVille(twoTones, { fs, nfft: 256, lagLength: 63, timeLength: 31 })
    expect(at(plain, 60)).toBeGreaterThan(0.5 * at(plain, 30))
    expect(at(smooth, 60)).toBeLessThan(0.05 * at(smooth, 30))
    expect(pseudo.smoothing).toBe('pseudo')
    expect(Math.min(...toFlat(plain.values))).toBeLessThan(0)
  })
})

describe('reassignment and synchrosqueezing', () => {
  it('a tone between bins is moved onto its frequency', () => {
    const f0 = 50.3
    const x = Array.from({ length: 1024 }, (_, i) => Math.cos((2 * Math.PI * f0 * i) / 1024))
    const opts = { fs: 1024, nperseg: 128, hop: 16 }
    const r = reassignedSpectrogram(x, opts)
    const s = synchrosqueeze(x, opts)
    const f = toFlat(r.f)
    for (const raster of [r.values, s.values]) {
      const col = column(raster, 32)
      const top = col.reduce((acc, v) => acc + v, 0)
      // Nearly all the column's weight sits in the bin nearest f0 (bins are 8 Hz apart).
      expect(col[argmax(col)] / top).toBeGreaterThan(0.7)
      expect(Math.abs(f[argmax(col)] - f0)).toBeLessThanOrEqual(4)
    }
    // Reassignment conserves energy within the record.
    const total = (t: Tensor) => toFlat(t).reduce((a, v) => a + v, 0)
    expect(total(r.values) / total(r.original)).toBeGreaterThan(0.97)
  })
})
