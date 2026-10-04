/**
 * Audio features: the mel scale, triangular mel filter banks, the orthonormal DCT-II, and mel-frequency cepstral
 * coefficients (Davis and Mermelstein, 1980, IEEE Trans. ASSP 28(4)).
 */

import { dct, readSignal, type Signal } from 'aifn-compute/foundation/fourier'
import { astype, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { stft } from 'aifn-compute/signal/spectral'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'

/** The mel scale: `htk` is 2595 log₁₀(1 + f/700); `slaney` is linear below 1 kHz and logarithmic above (librosa's default). */
export type MelScale = 'htk' | 'slaney'

const SLANEY = { fSp: 200 / 3, minLogHz: 1000, logStep: Math.log(6.4) / 27 }
const SLANEY_MIN_LOG_MEL = SLANEY.minLogHz / SLANEY.fSp

/** Hertz to mels (`librosa.hz_to_mel`). */
export function hzToMel(f: number, scale: MelScale = 'htk'): number {
  if (scale === 'htk') return 2595 * Math.log10(1 + f / 700)
  return f < SLANEY.minLogHz ? f / SLANEY.fSp : SLANEY_MIN_LOG_MEL + Math.log(f / SLANEY.minLogHz) / SLANEY.logStep
}

/** Mels to hertz (`librosa.mel_to_hz`). */
export function melToHz(m: number, scale: MelScale = 'htk'): number {
  if (scale === 'htk') return 700 * (10 ** (m / 2595) - 1)
  return m < SLANEY_MIN_LOG_MEL ? SLANEY.fSp * m : SLANEY.minLogHz * Math.exp(SLANEY.logStep * (m - SLANEY_MIN_LOG_MEL))
}

/** A mel filter bank: filter weights on the rFFT bins, the band edges and the bin frequencies. */
export interface MelFilterbank {
  /** Weights, [nMels, nfft/2 + 1]. */
  weights: Tensor
  /** nMels + 2 band edges in Hz, equally spaced in mel. */
  edges: Tensor
  /** Frequencies of the rFFT bins. */
  bins: Tensor
}

/**
 * A bank of `nMels` triangular filters with edges equally spaced in mel between fMin and fMax, evaluated at the rFFT
 * bin frequencies of an nfft-point transform. `norm: 'slaney'` scales each filter to unit area in Hz (2 / bandwidth),
 * so that wide high-frequency filters do not dominate; `null` leaves peaks at 1.
 */
export function melFilterbank(
  nMels: number,
  nfft: number,
  fs: number,
  options: { fMin?: number; fMax?: number; scale?: MelScale; norm?: 'slaney' | null } = {},
): MelFilterbank {
  const { fMin = 0, fMax = fs / 2, scale = 'htk', norm = 'slaney' } = options
  const lo = hzToMel(fMin, scale)
  const hi = hzToMel(fMax, scale)
  const edges = Float64Array.from({ length: nMels + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (nMels + 1), scale))
  const nbin = Math.floor(nfft / 2) + 1
  const bins = Float64Array.from({ length: nbin }, (_, k) => (k * fs) / nfft)
  const w = new Float64Array(nMels * nbin)
  for (let m = 0; m < nMels; m++) {
    const [a, b, c] = [edges[m], edges[m + 1], edges[m + 2]]
    const gain = norm === 'slaney' ? 2 / (c - a) : 1
    for (let k = 0; k < nbin; k++)
      w[m * nbin + k] = gain * Math.max(0, Math.min((bins[k] - a) / (b - a), (c - bins[k]) / (c - b)))
  }
  return { weights: fromData(w, [nMels, nbin]), edges: fromData(edges), bins: fromData(bins) }
}

/** Mel-frequency cepstral coefficients and the intermediate representations. */
export interface Mfcc {
  /** Cepstral coefficients, [frames, nMfcc]. */
  mfcc: Tensor
  /** Log mel energies, [frames, nMels]. */
  logMel: Tensor
  /** Power spectrogram |STFT|², [frames, bins]. */
  power: Tensor
  /** Frame centre times. */
  t: Tensor
  filterbank: MelFilterbank
}

/** Options for `mfcc`. */
export interface MfccOptions {
  nfft?: number
  hop?: number
  nMels?: number
  nMfcc?: number
  window?: WindowInput
  fMin?: number
  fMax?: number
  scale?: MelScale
  /** Added to mel energies before the log, so silent frames stay finite. Default 1e-10. */
  logOffset?: number
}

/**
 * MFCCs: the power spectrogram (frames of `nfft` samples every `hop`), mel energies by the filter bank, their log
 * (natural, plus `logOffset`), and the orthonormal DCT-II of each frame, keeping the first `nMfcc` coefficients.
 */
export function mfcc(x: Signal, fs: number, options: MfccOptions = {}): Mfcc {
  const { nfft = 512, hop = nfft / 4, nMels = 26, nMfcc = 13, window = 'hann', logOffset = 1e-10 } = options
  const v = readSignal(x, 'mfcc')
  const s = stft(v, { fs, nperseg: nfft, noverlap: nfft - hop, window, boundary: false, padded: false })
  // A complex time–frequency map: complex128 values [bins, frames], stored interleaved (re, im).
  const [nbin, frames] = s.values.shape
  const z = astype(s.values, 'complex128').data
  // stft divides by the window's sum; undo it so the power is |Σ w x e^{−iωn}|², as in the usual MFCC recipe.
  const scale = windowValues(window, nfft, true).reduce((acc, w) => acc + w, 0)
  const power = new Float64Array(frames * nbin)
  for (let t = 0; t < frames; t++)
    for (let k = 0; k < nbin; k++) {
      const a = z[2 * (k * frames + t)] * scale
      const b = z[2 * (k * frames + t) + 1] * scale
      power[t * nbin + k] = a * a + b * b
    }
  const bank = melFilterbank(nMels, nfft, fs, options)
  const w = bank.weights.data
  const logMel = new Float64Array(frames * nMels)
  for (let t = 0; t < frames; t++)
    for (let m = 0; m < nMels; m++) {
      let e = 0
      for (let k = 0; k < nbin; k++) e += w[m * nbin + k] * power[t * nbin + k]
      logMel[t * nMels + m] = Math.log(e + logOffset)
    }
  const cep = dct(fromData(logMel, [frames, nMels])).data
  const out = new Float64Array(frames * nMfcc)
  for (let t = 0; t < frames; t++) for (let c = 0; c < nMfcc; c++) out[t * nMfcc + c] = cep[t * nMels + c]
  return {
    mfcc: fromData(out, [frames, nMfcc]),
    logMel: fromData(logMel, [frames, nMels]),
    power: fromData(power, [frames, nbin]),
    t: s.t,
    filterbank: bank,
  }
}
