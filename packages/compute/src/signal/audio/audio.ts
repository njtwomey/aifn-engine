/**
 * Audio features: the mel scale, triangular mel filter banks, and mel-frequency cepstral coefficients (Davis and
 * Mermelstein, 1980, IEEE Trans. ASSP 28(4)), which apply the orthonormal DCT-II of `aifn-compute/foundation/fourier`
 * to log mel energies.
 *
 * Two mel scales are offered. HTK's is $m = 2595 \log_{10}(1 + f / 700)$. Slaney's (1998, Auditory Toolbox), librosa's
 * default, is linear below 1 kHz, $m = 3f / 200$, and logarithmic above, $m = 15 + 27 \ln(f / 1000) / \ln 6.4$.
 * Frequencies are in hertz throughout, and the functions follow `librosa.hz_to_mel`, `librosa.filters.mel` and
 * `librosa.feature.mfcc`, except that the default scale here is HTK's.
 */

import { dct, readSignal, type Signal } from 'aifn-compute/foundation/fourier'
import { astype, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { stft } from 'aifn-compute/signal/spectral'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'

/**
 * The mel scale: `'htk'` is $2595 \log_{10}(1 + f / 700)$; `'slaney'` is linear below 1 kHz and logarithmic above
 * (librosa's default).
 */
export type MelScale = 'htk' | 'slaney'

const SLANEY = { fSp: 200 / 3, minLogHz: 1000, logStep: Math.log(6.4) / 27 }
const SLANEY_MIN_LOG_MEL = SLANEY.minLogHz / SLANEY.fSp

/**
 * Hertz to mels (`librosa.hz_to_mel`). HTK's scale is $m = 2595 \log_{10}(1 + f / 700)$; Slaney's is $m = 3f / 200$
 * below 1000 Hz and $m = 15 + 27 \ln(f / 1000) / \ln 6.4$ above, so 1000 Hz is 15 mels. The inverse is `melToHz`.
 *
 * @param f The frequency $f$ in hertz.
 * @param scale Which mel scale: `'htk'` (default) or `'slaney'`.
 * @returns The frequency in mels.
 *
 * @example 1000 Hz on both scales
 * print('HTK:', hzToMel(1000))
 * print('Slaney:', hzToMel(1000, 'slaney'))
 *
 * @example Slaney: linear to 1 kHz, then each octave adds the same number of mels
 * print('mels of 500, 1000, 2000, 4000 Hz:', [500, 1000, 2000, 4000].map((f) => hzToMel(f, 'slaney')))
 */
export function hzToMel(f: number, scale: MelScale = 'htk'): number {
  if (scale === 'htk') return 2595 * Math.log10(1 + f / 700)
  return f < SLANEY.minLogHz ? f / SLANEY.fSp : SLANEY_MIN_LOG_MEL + Math.log(f / SLANEY.minLogHz) / SLANEY.logStep
}

/**
 * Mels to hertz (`librosa.mel_to_hz`), the inverse of `hzToMel` on the same scale: $f = 700 (10^{m / 2595} - 1)$ for
 * HTK's, and $f = 200m / 3$ below 15 mels, $f = 1000 \cdot 6.4^{(m - 15) / 27}$ above, for Slaney's.
 *
 * @param m The frequency $m$ in mels.
 * @param scale Which mel scale: `'htk'` (default) or `'slaney'`.
 * @returns The frequency in hertz.
 *
 * @example Round trip through the mel scale
 * const m = hzToMel(440)
 * print('440 Hz in mels:', m)
 * print('back to hertz:', melToHz(m))
 * print('15 Slaney mels in hertz:', melToHz(15, 'slaney'))
 */
export function melToHz(m: number, scale: MelScale = 'htk'): number {
  if (scale === 'htk') return 700 * (10 ** (m / 2595) - 1)
  return m < SLANEY_MIN_LOG_MEL ? SLANEY.fSp * m : SLANEY.minLogHz * Math.exp(SLANEY.logStep * (m - SLANEY_MIN_LOG_MEL))
}

/** A mel filter bank: filter weights on the rFFT bins, the band edges and the bin frequencies. */
export interface MelFilterbank {
  /** Weights, $n_\text{mels} \times (\lfloor n_\text{fft} / 2 \rfloor + 1)$: row $m$ is filter $m$ on the bins. */
  weights: Tensor
  /** The $n_\text{mels} + 2$ band edges in Hz, equally spaced in mel; filter $m$ spans edges $m$ to $m + 2$. */
  edges: Tensor
  /** Frequencies of the rFFT bins in Hz, $k f_s / n_\text{fft}$ for $0 \le k \le \lfloor n_\text{fft} / 2 \rfloor$. */
  bins: Tensor
}

/**
 * A bank of `nMels` triangular filters with edges equally spaced in mel between `fMin` and `fMax`, evaluated at the
 * rFFT bin frequencies of an `nfft`-point transform (`librosa.filters.mel`). Filter $m$ rises from 0 at edge $e_m$ to
 * its peak at $e_{m+1}$ and falls to 0 at $e_{m+2}$. `norm: 'slaney'` scales each filter to unit area in Hz (by
 * $2 / (e_{m+2} - e_m)$), so that wide high-frequency filters do not dominate; `null` leaves peaks at 1.
 *
 * @param nMels The number of filters $n_\text{mels}$.
 * @param nfft The length $n_\text{fft}$ of the transform whose $\lfloor n_\text{fft} / 2 \rfloor + 1$ one-sided bins
 *   the filters weight.
 * @param fs The sampling rate $f_s$ in Hz.
 * @param options The band, the mel scale and the normalisation.
 * @param options.fMin The lower edge of the first filter in Hz (default 0).
 * @param options.fMax The upper edge of the last filter in Hz (default the Nyquist frequency $f_s / 2$).
 * @param options.scale The mel scale the edges are spaced on: `'htk'` (default) or `'slaney'`.
 * @param options.norm `'slaney'` (default) for unit area in Hz, `null` for unit peaks.
 * @returns The weights, the band edges and the bin frequencies.
 *
 * @example Three unit-peak filters on the 9 bins of a 16-point transform
 * const bank = melFilterbank(3, 16, 16000, { norm: null })
 * print('edges (Hz):', bank.edges)
 * print('bins (Hz):', bank.bins)
 * print('weights:', bank.weights)
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
  /** Cepstral coefficients, frames $\times$ `nMfcc`. */
  mfcc: Tensor
  /** Natural-log mel energies, frames $\times$ `nMels`. */
  logMel: Tensor
  /** Power spectrogram $\abs{\sum_n w_n x_n e^{-i \omega n}}^2$ of each frame, frames $\times$ bins. */
  power: Tensor
  /** Frame centre times in seconds. */
  t: Tensor
  /** The mel filter bank applied to `power`. */
  filterbank: MelFilterbank
}

/** Options for `mfcc`. */
export interface MfccOptions {
  /** Frame length in samples, which is also the transform length (default 512). */
  nfft?: number
  /** Samples between the starts of successive frames (default `nfft / 4`). */
  hop?: number
  /** Number of mel filters (default 26). */
  nMels?: number
  /** Number of cepstral coefficients kept, from the first (default 13); at most `nMels`. */
  nMfcc?: number
  /** The analysis window (default `'hann'`). */
  window?: WindowInput
  /** Lower edge of the filter bank in Hz (default 0). */
  fMin?: number
  /** Upper edge of the filter bank in Hz (default the Nyquist frequency). */
  fMax?: number
  /** The mel scale of the filter bank (default `'htk'`). */
  scale?: MelScale
  /** Added to mel energies before the log, so silent frames stay finite. Default 1e-10. */
  logOffset?: number
}

/**
 * MFCCs (Davis and Mermelstein, 1980): the power spectrogram (frames of `nfft` samples every `hop`, with no padding at
 * the ends), mel energies by a Slaney-normalised filter bank, their natural log (plus `logOffset`), and the
 * orthonormal DCT-II of each frame, keeping the first `nMfcc` coefficients. The power is that of the unscaled windowed
 * transform, $\abs{\sum_n w_n x_n e^{-i \omega n}}^2$, as in librosa.
 *
 * @param x The signal: a rank-1 tensor or an array of samples. Other ranks throw `ShapeError`.
 * @param fs The sampling rate $f_s$ in Hz.
 * @param options Frame length and hop, window, filter bank and the number of coefficients kept.
 * @returns The coefficients with the power spectrogram, log mel energies, frame times and filter bank they came from.
 *
 * @example A steady tone gives the same coefficients in every frame
 * const fs = 8000
 * const x = Array.from({ length: 1024 }, (_, n) => Math.sin((2 * Math.PI * 1000 * n) / fs))
 * const { mfcc: c, logMel, t, filterbank } = mfcc(x, fs, { nfft: 256, hop: 128, nMels: 20, nMfcc: 4 })
 * print('frames x coefficients:', c.shape)
 * print('frame times (s):', t)
 * print('first two frames:', c.data.slice(0, 8))
 * const bands = Array.from(logMel.data.slice(0, 20))
 * const loudest = bands.indexOf(Math.max(...bands))
 * print('loudest mel band of frame 0:', loudest, 'peaking at', filterbank.edges.data[loudest + 1], 'Hz')
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
