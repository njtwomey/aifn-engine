/** Wavelet shrinkage: thresholding rules and the universal threshold improving the SNR of a noisy smooth signal. */
import { describe, expect, it } from 'vitest'
import { waveletDenoise, waveletThreshold } from 'aifn-compute/signal/wavelets'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

describe('wavelet denoising', () => {
  it('soft and hard thresholding, as pywt.threshold', () => {
    expect(toFlat(waveletThreshold([-3, -1, 0.5, 2, 4], 1.5, 'soft'))).toEqual([-1.5, -0, 0, 0.5, 2.5])
    expect(toFlat(waveletThreshold([-3, -1, 0.5, 2, 4], 1.5, 'hard'))).toEqual([-3, 0, 0, 2, 4])
  })

  it('the universal threshold raises the SNR of a noisy piecewise-smooth signal', () => {
    const n = 1024
    const clean = Array.from(
      { length: n },
      (_, i) => (i < 400 ? Math.sin((2 * Math.PI * 3 * i) / n) : 0.8) + (i > 700 ? -1 : 0),
    )
    const noise = toFlat(normals(stream('denoise'), n, 0, 0.3))
    const x = clean.map((v, i) => v + noise[i])
    const out = waveletDenoise(x, { wavelet: 'db4' })
    expect(out.sigma).toBeGreaterThan(0.25)
    expect(out.sigma).toBeLessThan(0.35)
    expect(out.threshold).toBeCloseTo(out.sigma * Math.sqrt(2 * Math.log(n)), 12)
    const err = (y: ArrayLike<number>) => clean.reduce((s, v, i) => s + (v - y[i]) ** 2, 0)
    expect(err(toFlat(out.signal.data))).toBeLessThan(0.25 * err(x))
  })
})
