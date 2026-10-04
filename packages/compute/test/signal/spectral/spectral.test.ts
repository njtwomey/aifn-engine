import { describe, expect, it } from 'vitest'
import { dpss, multitaper, periodogram, spectrogram, stft, welch } from 'aifn-compute/signal/spectral'
import { signal } from 'aifn-compute/signal'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { close, closeC, F } from '../helpers'

describe('spectral estimation against scipy.signal', () => {
  const S = F.spectral
  it('welch (mean and median, detrending) and periodogram, as Spectrum', () => {
    const w = welch(S.x, { fs: 100, nperseg: 128 })
    expect(w.kind).toBe('spectrum')
    expect(w.quantity).toBe('psd')
    expect(w.axis).toBe('hz')
    close(w.f, S.welch.f)
    close(w.values, S.welch.psd, 1e-10)
    const m = welch(S.x, { fs: 100, nperseg: 100, noverlap: 25, average: 'median', detrend: 'linear' })
    close(m.values, S.welchMedian.psd, 1e-10)
    const p = periodogram((S.x as number[]).slice(0, 300), { fs: 100, window: 'hann', nfft: 512 })
    close(p.f, S.periodogram.f)
    close(p.values, S.periodogram.psd, 1e-10)
  })

  it('a Signal supplies its sample rate', () => {
    const a = welch(signal(S.x, { fs: 100 }), { nperseg: 128 })
    close(a.f, S.welch.f)
    close(a.values, S.welch.psd, 1e-10)
  })

  it('spectrogram and stft, as TimeFrequency', () => {
    const sg = spectrogram(S.x, { fs: 100, nperseg: 64 })
    expect(sg.kind).toBe('time-frequency')
    expect(sg.quantity).toBe('power')
    close(sg.t, S.spectrogram.t)
    close(sg.values, S.spectrogram.power, 1e-10)
    const st = stft((S.x as number[]).slice(0, 500), { fs: 100, nperseg: 64 })
    expect(st.quantity).toBe('complex')
    close(st.t, S.stft.t)
    close(st.f, S.stft.f)
    expect(st.values.shape).toEqual([S.stft.f.length, S.stft.t.length])
    closeC(st.values, S.stft.Z, 1e-10)
  })

  it('dpss tapers and concentrations', () => {
    const d = dpss(64, 3, 5)
    close(d.tapers, F.dpss.tapers, 1e-8)
    close(d.concentrations, F.dpss.ratios, 1e-8)
  })

  it('multitaper integrates to the variance', () => {
    const x = Array.from({ length: 512 }, (_, i) => Math.sin(i * 1.3) + Math.cos(i * 0.37 + 1))
    const m = multitaper(x, { nw: 3 })
    const df = toFlat(m.f)[1]
    const total = toFlat(m.values).reduce((a, b) => a + b, 0) * df
    const mean = x.reduce((a, b) => a + b, 0) / x.length
    const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / x.length
    expect(total).toBeCloseTo(variance, 1)
  })
})
