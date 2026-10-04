import { describe, expect, test } from 'vitest'
import {
  bartlett,
  blackmanTukey,
  chiSquareQuantile,
  coherence,
  coherenceThreshold,
  csd,
  falseAlarmLevel,
  falseAlarmProbability,
  logSpectralError,
  lombScargle,
  lombScargleFrequencies,
  multitaper,
  periodogram,
  spectralConfidence,
  spectralWindow,
  welch,
  welchDof,
} from 'aifn-compute/signal/spectral'
import { imagPart, realPart, toFlat } from 'aifn-compute/foundation/tensor'
import { fixture } from '../../fixtures'
import { close } from '../helpers'

// Golden values from scipy.signal, astropy and numpy ports (fixtures/gen/signal/spectral.py).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('signal/spectral')
const fs = F.fs as number

describe('segment averages', () => {
  test('welch, bartlett and the periodogram match scipy', () => {
    close(welch(F.x, { fs, nperseg: 128, noverlap: 64 }).values, F.welch.psd, 1e-10)
    close(bartlett(F.x, { fs, nperseg: 100 }).values, F.bartlett, 1e-10)
    close(periodogram(F.x, { fs }).values, F.periodogram, 1e-10)
  })
  test('equivalent degrees of freedom: 2K without overlap, about 36K/19 for Hann at 50%', () => {
    expect(welchDof('boxcar', 100, 0, 6)).toBe(12)
    expect(bartlett(F.x, { nperseg: 100 }).dof).toBe(12)
    expect(periodogram(F.x).dof).toBe(2)
    // Percival and Walden (1993), Table 294: Hann with 50% overlap gives ν ≈ 36K/19 for large K.
    expect(welchDof('hann', 256, 128, 400) / 400).toBeCloseTo(36 / 19, 2)
  })
  test('the cross-spectral density and coherence match scipy', () => {
    const p = csd(F.x, F.y, { fs, nperseg: 128 })
    close(realPart(p.values), F.csd.re, 1e-10)
    close(imagPart(p.values), F.csd.im, 1e-10)
    close(coherence(F.x, F.y, { fs, nperseg: 128 }).values, F.coherence, 1e-9)
    // The csd of x with itself is welch's PSD.
    close(realPart(csd(F.x, F.x, { fs, nperseg: 128 }).values), F.welch.psd, 1e-10)
  })
  test('coherence of independent noise stays under its null threshold most of the time', () => {
    const K = 9
    const c = coherenceThreshold(K, { level: 0.95 })
    expect(Math.pow(1 - c, K - 1)).toBeCloseTo(0.05, 12)
  })
})

describe('Blackman–Tukey', () => {
  test('matches the direct lag-window sum', () => {
    const bt = blackmanTukey(F.x, { fs, maxLag: 30 })
    close(bt.values, F.blackmanTukey.psd, 1e-10)
    expect(bt.dof).toBeCloseTo(
      (2 * 600) / (1 + 2 * Array.from({ length: 30 }, (_, k) => (1 - (k + 1) / 30) ** 2).reduce((a, b) => a + b)),
      10,
    )
  })
  test('with a rectangular window over every lag it is the periodogram (Wiener–Khinchin)', () => {
    const n = F.x.length
    const bt = blackmanTukey(F.x, { fs, maxLag: n - 1, lagWindow: 'boxcar', nfft: 2 * n })
    const p = periodogram(F.x, { fs, nfft: 2 * n })
    close(bt.values, toFlat(p.values) as number[], 1e-9)
  })
})

describe('multitaper', () => {
  test('plain and adaptive estimates match the reference weights', () => {
    const plain = multitaper(F.x, { fs, nw: 4, k: 7 })
    close(plain.concentrations, F.multitaper.ratios, 1e-6)
    close(plain.values, F.multitaper.plain, 1e-6)
    expect(toFlat(plain.dof).every((v) => v === 14)).toBe(true)
    const adaptive = multitaper(F.x, { fs, nw: 4, k: 7, adaptive: true })
    close(adaptive.values, F.multitaper.adaptive, 1e-6)
    close(adaptive.dof, F.multitaper.dof, 1e-5)
  })
})

describe('confidence intervals', () => {
  test('χ² quantiles match scipy and the interval brackets the estimate', () => {
    F.chi2.nu.forEach((nu: number, i: number) => {
      expect(chiSquareQuantile(nu, 0.025)).toBeCloseTo(F.chi2.lo[i], 9)
      expect(chiSquareQuantile(nu, 0.975)).toBeCloseTo(F.chi2.hi[i], 9)
    })
    const w = welch(F.x, { fs, nperseg: 128 })
    const ci = spectralConfidence(w, w.dof)
    const [lo, hi, v] = [toFlat(ci.lower), toFlat(ci.upper), toFlat(w.values)]
    for (let i = 0; i < v.length; i++) expect(lo[i] < v[i] && v[i] < hi[i]).toBe(true)
    // Constant width in dB.
    const width = (i: number) => 10 * Math.log10(hi[i] / lo[i])
    expect(width(3)).toBeCloseTo(width(40), 10)
  })
  test('log-spectral error splits into bias and spread', () => {
    const e = logSpectralError([2, 2, 2, 2], [1, 1, 1, 1])
    expect(e.bias).toBeCloseTo(10 * Math.log10(2), 12)
    expect(e.sd).toBeCloseTo(0, 12)
    const g = logSpectralError([1, 4, 1, 4], [2, 2, 2, 2])
    expect(g.distance ** 2).toBeCloseTo(g.bias ** 2 + g.sd ** 2, 12)
  })
})

describe('Lomb–Scargle', () => {
  const U = F.uneven
  test('classic, floating-mean and generalised powers match astropy', () => {
    close(lombScargle(U.t, U.y, U.f, { method: 'classic' }).values, U.classic, 1e-9)
    close(lombScargle(U.t, U.y, U.f).values, U.floating, 1e-9)
    close(lombScargle(U.t, U.y, U.f, { method: 'generalised', dy: U.dy }).values, U.generalised, 1e-9)
    close(lombScargle(U.t, U.y, U.f, { normalization: 'psd' }).values, U.psd, 1e-9)
  })
  test("Baluev's false-alarm probability and level match astropy", () => {
    const ls = lombScargle(U.t, U.y, U.f)
    // astropy reads f_max off its autofrequency grid, which stops at or below the requested maximum.
    const maximum = Math.max(...toFlat(lombScargleFrequencies(U.t, { maximum: 1.5 })))
    U.levels.forEach((z: number, i: number) => {
      const p = falseAlarmProbability(z, ls, { maximum })
      expect(Math.abs(p - U.fap[i])).toBeLessThan(1e-9 * Math.max(1, U.fap[i]) + 1e-15)
    })
    U.fapTargets.forEach((p: number, i: number) => {
      expect(falseAlarmLevel(p, ls, { maximum })).toBeCloseTo(U.fapLevel[i], 8)
    })
  })
  test('the spectral window is 1 at zero frequency and matches the direct sum', () => {
    close(spectralWindow(U.t, U.f).values, U.window, 1e-10)
    expect(toFlat(spectralWindow(U.t, [0]).values)[0]).toBeCloseTo(1, 12)
  })
  test('the frequency grid spaces peaks by 1/(samplesPerPeak · T)', () => {
    const f = toFlat(lombScargleFrequencies([0, 10, 20], { samplesPerPeak: 4, maximum: 1 }))
    expect(f[1] - f[0]).toBeCloseTo(1 / 80, 12)
    expect(f[0]).toBeCloseTo(1 / 160, 12)
  })
})
