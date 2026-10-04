import { describe, expect, it } from 'vitest'
import { checkCola, checkNola, istft, stft } from 'aifn-compute/signal/spectral'
import { signal } from 'aifn-compute/signal'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { close, closeC, F } from '../helpers'

const I = F.istft
const complex = (z: { re: number[][]; im: number[][] }) => {
  const [r, c] = [z.re.length, z.re[0].length]
  const d = new Float64Array(2 * r * c)
  z.re.flat().forEach((v, k) => (d[2 * k] = v))
  z.im.flat().forEach((v, k) => (d[2 * k + 1] = v))
  return fromData(d, [r, c], 'complex128')
}

describe('istft against scipy.signal', () => {
  it.each(Object.keys(I.cases))('inverts stft exactly under NOLA (%s)', (key) => {
    const c = I.cases[key]
    const Z = stft(I.x, { fs: 100, window: c.window, nperseg: c.nperseg, noverlap: c.noverlap })
    const x = istft(Z, { window: c.window })
    expect(x.kind).toBe('signal')
    expect(x.fs).toBeCloseTo(100, 12)
    close(x.data, c.x, 1e-12)
    // Perfect reconstruction: the first n samples are the original signal (the rest is stft's end padding).
    close(toFlat(x.data).slice(0, I.x.length), I.x, 1e-12)
  })

  it('inverts a two-sided stft (complex output) and gives the least-squares signal for a modified STFT', () => {
    const x2 = istft(complex(I.twoSided.Z), { fs: 100, nperseg: 64, onesided: false })
    closeC(x2.data, I.twoSided.x, 1e-12)
    const xm = istft(complex(I.masked.Z), { fs: 100, nperseg: 64 })
    close(xm.data, I.masked.x, 1e-12)
  })

  it('keeps the time origin of the signal', () => {
    const s = signal(I.x, { fs: 100, t0: 2.5 })
    expect(istft(stft(s, { nperseg: 64 })).t0).toBeCloseTo(2.5, 12)
    const nb = istft(stft(s, { nperseg: 64, boundary: false }), { boundary: false })
    expect(nb.t0).toBeCloseTo(2.5, 12)
    // Without boundary padding the periodic Hann window is 0 at sample 0, which no segment then sees.
    close(toFlat(nb.data).slice(1, 400), I.x.slice(1, 400), 1e-12)
  })

  it.each(Object.keys(I.cola))('checkCola and checkNola match scipy (%s)', (key) => {
    const c = I.cola[key]
    expect(checkCola(c.window, c.nperseg, c.noverlap)).toBe(c.cola)
    expect(checkNola(c.window, c.nperseg, c.noverlap)).toBe(c.nola)
  })

  it('refuses a window and overlap that fail NOLA', () => {
    const Z = stft(I.x, { window: 'hann', nperseg: 4, noverlap: 0 })
    expect(() => istft(Z, { window: 'hann' })).toThrow(/NOLA/)
  })
})
