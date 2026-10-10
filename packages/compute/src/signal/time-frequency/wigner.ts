/**
 * Quadratic time–frequency distributions of Cohen's class (Cohen, 1989, Proc. IEEE 77(7)): the Wigner–Ville
 * distribution $W(t, f) = \sum_\tau z[t + \tau]\, z^*[t - \tau]\, e^{-i 4\pi f \tau}$ of the analytic signal $z$
 * (Ville, 1948), its pseudo form (a lag window $h(\tau)$, which smooths along frequency) and its smoothed pseudo form
 * (also a time window $g$, which smooths along time), as the Time–Frequency Toolbox's `tfrwv`, `tfrpwv` and `tfrspwv`
 * (Auger, Flandrin, Gonçalvès and Lemoine, 1996). The distribution is real and can be negative: two components
 * interfere half way between them (cross-terms), oscillating at a rate set by their separation, and the smoothing
 * windows trade those terms against resolution.
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
  /** Frequency bins over $[0, f_s/2)$ (default the next power of two $\ge$ the length, at most 1024). */
  nfft?: Size
  /** Compute every `hop`-th time sample (default 1; rounded down, at least 1). */
  hop?: Size
  /**
   * Analyse the analytic signal (default true), which removes the interference between positive and negative
   * frequencies. When false the real samples are analysed as they are.
   */
  analytic?: boolean
  /**
   * A lag window $h(\tau)$ of odd length (pseudo-WVD): smooths along frequency. A spec with `lagLength`, or values.
   * It is scaled so that $h(0) = 1$, and limits the lags to its half-length.
   */
  lagWindow?: WindowInput
  /** The odd length of a `lagWindow` given as a spec (default 63); ignored for values. */
  lagLength?: Size
  /**
   * A time window $g$ of odd length (smoothed pseudo-WVD): smooths along time. A spec with `timeLength`, or values.
   * The local autocorrelation is averaged over it, divided by the sum of its weights inside the record.
   */
  timeWindow?: WindowInput
  /** The odd length of a `timeWindow` given as a spec (default 63); ignored for values. */
  timeLength?: Size
}

/** A Wigner–Ville raster: real values $[f, t]$ (`quantity: 'distribution'`, `method: 'wvd'`), possibly negative. */
export type WignerVille = TimeFrequency & {
  /** The smoothing applied: none, lag (pseudo) or lag and time (smoothed pseudo). */
  smoothing: 'none' | 'pseudo' | 'smoothed-pseudo'
}

/**
 * The values of an optional smoothing window, which must have odd length. Throws `DomainError` for an even length.
 *
 * @param input The window: a spec (name or parameterised), its values, or undefined for none.
 * @param length The length for a spec (default 63); ignored for values.
 * @param where The caller's name for error messages.
 * @returns The symmetric window's values, or null when `input` is undefined.
 */
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
 * The (pseudo, smoothed pseudo) Wigner–Ville distribution of a signal: for each analysed time $t$ the local
 * autocorrelation $K(t, \tau) = h(\tau) \sum_s g(s)\, z[t + s + \tau]\, z^*[t + s - \tau]$ over the lags that fit,
 * Fourier transformed over $\tau$. The lag runs in steps of one sample on each side, so bin $k$ is the frequency
 * $k f_s / (2 n_\text{fft})$ and the raster covers $[0, f_s/2)$ with the analytic signal (a real input analysed
 * directly folds negative frequencies in). The marginal $\frac{1}{n_\text{fft}} \sum_f W(t, f)$ is $\abs{z(t)}^2$ for
 * the plain distribution. The lags used at $t$ are limited by the record's ends, by the lag window's half-length and
 * by $n_\text{fft}/2 - 1$. Throws `DomainError` for a smoothing window of even length.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The sample rate, frequency bins, hop, analytic flag and smoothing windows; see `WignerOptions`.
 *   Without windows this is the plain distribution.
 * @returns The real raster $[n_\text{fft}, T]$ for the $T$ analysed times, with the smoothing applied.
 *
 * @example A tone sits on its frequency
 * // cos at 0.125 cycles per sample: the middle column peaks at 0.125, and its mean is |z|^2 = 1.
 * const x = Array.from({ length: 64 }, (_, i) => Math.cos(2 * Math.PI * 0.125 * i))
 * const W = wignerVille(x)
 * const column = Array.from({ length: 64 }, (_, k) => W.values.data[k * 64 + 32])
 * print('peak at f =', W.f.data[argmax(tensor(column))], ' mean of the column =', mean(tensor(column)))

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

/**
 * The pseudo Wigner–Ville distribution: `wignerVille` with a lag window (default Hann of `lagLength`, 63), which
 * smooths along frequency and limits the lags to $\pm 31$ (by default).
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The `WignerOptions` without a time window; `lagWindow` defaults to `'hann'`.
 * @returns The real raster, `smoothing: 'pseudo'`.
 *
 * @example The lag window spreads a tone
 * // A shorter lag window gives a wider ridge: count the bins above half the peak in the middle column.
 * const x = Array.from({ length: 128 }, (_, i) => Math.cos(2 * Math.PI * 0.2 * i))
 * for (const lagLength of [15, 63]) {
 *   const W = pseudoWignerVille(x, { lagLength })
 *   const column = Array.from({ length: 128 }, (_, k) => W.values.data[k * 128 + 64])
 *   const peak = Math.max(...column)
 *   print(`lagLength ${lagLength}: bins above half the peak =`, column.filter((v) => v > peak / 2).length)
 * }
 */
export function pseudoWignerVille(
  x: SignalInput,
  options: Omit<WignerOptions, 'timeWindow' | 'timeLength'> = {},
): WignerVille {
  return wignerVille(x, { lagWindow: 'hann', ...options })
}

/**
 * The smoothed pseudo Wigner–Ville distribution: lag and time windows (default Hann of 63 and 15 samples), a separable
 * Cohen-class kernel that removes most cross-terms at some cost in resolution.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The `WignerOptions`; `lagWindow` and `timeWindow` default to `'hann'` and `timeLength` to 15.
 * @returns The real raster, `smoothing: 'smoothed-pseudo'`.
 *
 * @example Cross-terms between two tones
 * // Halfway between tones at 0.1 and 0.3 the pseudo form keeps an oscillating cross-term; time smoothing removes
 * // it.
 * const tone = (f, i) => Math.cos(2 * Math.PI * f * i)
 * const x = Array.from({ length: 128 }, (_, i) => tone(0.1, i) + tone(0.3, i))
 * const at = (W, f) => W.values.data[Math.round(f * 256) * 128 + 64]
 * const P = pseudoWignerVille(x, { lagLength: 31 })
 * const S = smoothedPseudoWignerVille(x, { lagLength: 31 })
 * print('pseudo at 0.1, 0.2, 0.3 =', at(P, 0.1), at(P, 0.2), at(P, 0.3))
 * print('smoothed pseudo at 0.1, 0.2, 0.3 =', at(S, 0.1), at(S, 0.2), at(S, 0.3))
 */
export function smoothedPseudoWignerVille(x: SignalInput, options: WignerOptions = {}): WignerVille {
  return wignerVille(x, { lagWindow: 'hann', timeWindow: 'hann', timeLength: 15, ...options })
}
