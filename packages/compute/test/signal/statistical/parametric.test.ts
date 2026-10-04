import { describe, expect, test } from 'vitest'
import {
  armaSpectrum,
  arPsd,
  burg,
  esprit,
  leastSquaresAr,
  music,
  sinusoidFit,
  yuleWalker,
} from 'aifn-compute/signal/statistical'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'
import { close } from '../helpers'

// Golden values from statsmodels, scipy.signal.freqz and a numpy least-squares fit (fixtures/gen/signal/spectral.py).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('signal/spectral')
const A = F.ar

/** Two tones in white noise, with known frequencies and amplitudes. */
function tones(n: number, f1: number, f2: number, sd = 0.1): number[] {
  const e = toFlat(normals(stream('parametric-tones'), n, 0, sd))
  return e.map((v, t) => v + Math.sin(2 * Math.PI * f1 * t + 0.3) + 0.5 * Math.sin(2 * Math.PI * f2 * t + 1.1))
}

describe('AR fits', () => {
  test('Burg and Yule–Walker match statsmodels; modified covariance matches least squares', () => {
    // Burg's σ² is the final mean error power here; statsmodels normalises it differently, so only φ is compared.
    close(toFlat(burg(F.x, 4).ar), A.burg.ar, 1e-9)
    close(toFlat(yuleWalker(F.x, 4).ar), A.yuleWalker.ar, 1e-9)
    expect(yuleWalker(F.x, 4).sigma2).toBeCloseTo(A.yuleWalker.sigma2, 9)
    const ls = leastSquaresAr(F.x, 4)
    close(toFlat(ls.ar), A.leastSquares.ar, 1e-9)
    expect(ls.sigma2).toBeCloseTo(A.leastSquares.sigma2, 9)
  })
  test('the step-down reflection coefficients of a stationary fit lie inside (−1, 1) and end with φ_p', () => {
    const k = toFlat(leastSquaresAr(F.x, 4).reflection)
    expect(k.every((v) => Math.abs(v) < 1)).toBe(true)
    expect(k[3]).toBeCloseTo(toFlat(leastSquaresAr(F.x, 4).ar)[3], 12)
  })
})

describe('ARMA spectrum', () => {
  test('matches |B/A|² from freqz, one-sided', () => {
    const s = armaSpectrum({ ar: A.arma.ar, ma: A.arma.ma, sigma2: A.arma.sigma2 }, { fs: F.fs, frequencies: A.arma.f })
    close(s.values, A.arma.psd, 1e-10)
  })
  test('integrates over [0, fs/2] to the process variance', () => {
    // AR(1) with φ = 0.6: variance σ²/(1 − φ²).
    const nfft = 1 << 14
    const fs = 3
    const s = toFlat(armaSpectrum({ ar: [0.6], sigma2: 2 }, { fs, nfft }).values)
    const df = fs / nfft
    const integral = s.reduce((a, v) => a + v, 0) * df
    expect(integral).toBeCloseTo(2 / (1 - 0.36), 3)
  })
  test('an AR estimate of an AR(2) series peaks near the true peak', () => {
    const est = arPsd(F.x, 2, { method: 'burg', nfft: 1024 })
    const v = toFlat(est.values)
    const f = toFlat(est.f)
    expect(f[v.indexOf(Math.max(...v))]).toBeGreaterThan(0.06)
    expect(f[v.indexOf(Math.max(...v))]).toBeLessThan(0.14)
  })
})

describe('line spectra', () => {
  const x = tones(128, 0.2, 0.23)
  test('MUSIC and ESPRIT resolve two tones closer than the Fourier resolution 1/n', () => {
    // 0.03 cycles per sample apart with n = 128 (1/n ≈ 0.008), at high SNR.
    const m = music(x, { sinusoids: 2, order: 30 })
    close(toFlat(m.frequencies), [0.2, 0.23], 2e-3)
    const e = esprit(x, { sinusoids: 2, order: 30 })
    close(toFlat(e.frequencies), [0.2, 0.23], 1e-3)
    close(toFlat(e.powers), [0.5, 0.125], 0.02)
    // Two pairs of eigenvalues stand above the noise floor.
    const ev = toFlat(e.eigenvalues)
    expect(ev[3] / ev[4]).toBeGreaterThan(20)
  })
  test('sinusoidFit recovers amplitudes and phases at known frequencies', () => {
    const fit = sinusoidFit(x, [0.2, 0.23])
    close(toFlat(fit.amplitudes), [1, 0.5], 0.02)
    close(toFlat(fit.phases), [0.3, 1.1], 0.05)
  })
})
