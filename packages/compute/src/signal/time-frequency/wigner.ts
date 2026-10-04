/**
 * Quadratic time–frequency distributions of Cohen's class (Cohen, 1989, Proc. IEEE 77(7)): the Wigner–Ville
 * distribution W(t, f) = Σ_τ z[t + τ] z*[t − τ] e^{−i4πfτ} of the analytic signal z (Ville, 1948), its pseudo form (a
 * lag window h(τ), which smooths along frequency) and its smoothed pseudo form (also a time window g, which smooths
 * along time), as the Time–Frequency Toolbox's `tfrwv`, `tfrpwv` and `tfrspwv` (Auger, Flandrin, Gonçalvès and Lemoine,
 * 1996). The distribution is real and can be negative: two components interfere half way between them (cross-terms),
 * oscillating at a rate set by their separation, and the smoothing windows trade those terms against resolution.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { fft, nextPowerOfTwo } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size, TimeFrequency } from 'aifn-compute/foundation/contracts'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'
import { readSamples, timeFrequency, type SignalInput } from '../signal'
import { hilbert } from './hilbert'
import { DomainError } from 'aifn-compute/foundation/errors'

/** Options of `wignerVille`. */
export type WignerOptions = {
  /** Sample rate (default the signal's, else 1). */
  fs?: Scalar
  /** Frequency bins over [0, fs/2) (default the next power of two ≥ the length, at most 1024). */
  nfft?: Size
  /** Compute every `hop`-th time sample (default 1). */
  hop?: Size
  /** Analyse the analytic signal (default true), which removes the interference between positive and negative frequencies. */
  analytic?: boolean
  /** A lag window h(τ) of odd length (pseudo-WVD): smooths along frequency. A spec with `lagLength`, or values. */
  lagWindow?: WindowInput
  lagLength?: Size
  /** A time window g of odd length (smoothed pseudo-WVD): smooths along time. A spec with `timeLength`, or values. */
  timeWindow?: WindowInput
  timeLength?: Size
}

/** A Wigner–Ville raster: real values [f, t] (`quantity: 'distribution'`), possibly negative. */
export type WignerVille = TimeFrequency & {
  /** The smoothing applied: none, lag (pseudo) or lag and time (smoothed pseudo). */
  smoothing: 'none' | 'pseudo' | 'smoothed-pseudo'
}

function oddWindow(input: WindowInput | undefined, length: Size | undefined, where: string): Float64Array | null {
  if (input === undefined) return null
  const L =
    typeof input === 'string' || (typeof input === 'object' && 'name' in input)
      ? (length ?? 63)
      : (input as ArrayLike<number>).length
  if (L % 2 !== 1) throw new DomainError(where, `${where}: smoothing windows must have odd length`)
  return windowValues(input, L, false)
}

/**
 * The (pseudo, smoothed pseudo) Wigner–Ville distribution of a signal: for each analysed time t the local
 * autocorrelation K(t, τ) = h(τ) Σ_s g(s) z[t + s + τ] z*[t + s − τ] over the lags that fit, Fourier transformed over τ.
 * The lag runs in steps of one sample on each side, so bin k is the frequency k fs/(2 nfft) and the raster covers
 * [0, fs/2) with the analytic signal (a real input analysed directly folds negative frequencies in). The marginal
 * (1/nfft) Σ_f W(t, f) is |z(t)|² for the plain distribution.
 */
export function wignerVille(x: SignalInput, options: WignerOptions = {}): WignerVille {
  const input = readSamples(x, 'wignerVille', options.fs)
  const { fs, t0 } = input
  const n = input.values.length
  const nfft = options.nfft ?? Math.min(1024, nextPowerOfTwo(n))
  const hop = Math.max(1, Math.floor(options.hop ?? 1))
  const h = oddWindow(options.lagWindow, options.lagLength, 'wignerVille')
  const g = oddWindow(options.timeWindow, options.timeLength, 'wignerVille')
  // Analytic signal (or the real samples as complex).
  let re: Float64Array
  let im: Float64Array
  if (options.analytic ?? true) {
    const z = hilbert(input.values).data as Float64Array
    re = Float64Array.from({ length: n }, (_, i) => z[2 * i])
    im = Float64Array.from({ length: n }, (_, i) => z[2 * i + 1])
  } else {
    re = Float64Array.from(input.values)
    im = new Float64Array(n)
  }
  const hHalf = h ? (h.length - 1) / 2 : Infinity
  const hMid = h ? h[(h.length - 1) / 2] : 1
  const gHalf = g ? (g.length - 1) / 2 : 0
  const times: number[] = []
  for (let t = 0; t < n; t += hop) times.push(t)
  const T = times.length
  const K = new Float64Array(2 * T * nfft)
  times.forEach((t, row) => {
    const tauMax = Math.min(t, n - 1 - t, hHalf, Math.floor(nfft / 2) - 1)
    for (let tau = -tauMax; tau <= tauMax; tau++) {
      let sr = 0
      let si = 0
      if (g) {
        // Σ_s g(s) z[t + s + τ] z*[t + s − τ] / Σ g over the s that keep both samples inside the record.
        let norm = 0
        const lo = Math.max(-gHalf, -t + Math.abs(tau))
        const hi = Math.min(gHalf, n - 1 - t - Math.abs(tau))
        for (let s = lo; s <= hi; s++) {
          const a = t + s + tau
          const b = t + s - tau
          const w = g[s + gHalf]
          sr += w * (re[a] * re[b] + im[a] * im[b])
          si += w * (im[a] * re[b] - re[a] * im[b])
          norm += w
        }
        if (norm > 0) {
          sr /= norm
          si /= norm
        }
      } else {
        const a = t + tau
        const b = t - tau
        sr = re[a] * re[b] + im[a] * im[b]
        si = im[a] * re[b] - re[a] * im[b]
      }
      const w = h ? h[tau + hHalf] / hMid : 1
      const k = ((tau % nfft) + nfft) % nfft
      K[2 * (row * nfft + k)] = w * sr
      K[2 * (row * nfft + k) + 1] = w * si
    }
  })
  const F = fft(fromData(K, [T, nfft], 'complex128')).data as Float64Array
  // K(t, −τ) = K*(t, τ), so the transform is real: keep the real part, transposed to [f, t].
  const values = new Float64Array(nfft * T)
  for (let row = 0; row < T; row++) for (let k = 0; k < nfft; k++) values[k * T + row] = F[2 * (row * nfft + k)]
  return {
    ...timeFrequency({
      t: fromData(
        Float64Array.from(times, (t) => t0 + t / fs),
        [T],
      ),
      f: fromData(
        Float64Array.from({ length: nfft }, (_, k) => (k * fs) / (2 * nfft)),
        [nfft],
      ),
      values: fromData(values, [nfft, T]) as Tensor,
      quantity: 'distribution',
      method: 'wvd',
      frequencyScale: 'linear',
    }),
    smoothing: g ? 'smoothed-pseudo' : h ? 'pseudo' : 'none',
  }
}

/** The pseudo Wigner–Ville distribution: `wignerVille` with a lag window (default Hann of `lagLength`, 63). */
export function pseudoWignerVille(
  x: SignalInput,
  options: Omit<WignerOptions, 'timeWindow' | 'timeLength'> = {},
): WignerVille {
  return wignerVille(x, { lagWindow: 'hann', ...options })
}

/**
 * The smoothed pseudo Wigner–Ville distribution: lag and time windows (default Hann of 63 and 15 samples), a separable
 * Cohen-class kernel that removes most cross-terms at some cost in resolution.
 */
export function smoothedPseudoWignerVille(x: SignalInput, options: WignerOptions = {}): WignerVille {
  return wignerVille(x, { lagWindow: 'hann', timeWindow: 'hann', timeLength: 15, ...options })
}
