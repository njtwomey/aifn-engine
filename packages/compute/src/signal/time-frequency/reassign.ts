/**
 * Sharpened spectrograms. The reassignment method (Kodera, Gendrin and de Villedary, 1978; Auger and Flandrin, 1995,
 * IEEE Trans. Signal Process. 43(5)) moves each spectrogram value from the centre of its cell to the local centre of
 * gravity of the energy: the instantaneous frequency $\hat{f} = f - \operatorname{Im}(V_{g'}/V_g)/2\pi$ and the group
 * delay $\hat{t} = t + \operatorname{Re}(V_{tg}/V_g)$, computed from two extra STFTs with the windows $g'$ (derivative)
 * and $t g$. Synchrosqueezing (Daubechies, Lu and Wu, 2011, Appl. Comput. Harmon. Anal. 30(2); Thakur and Wu, 2011,
 * for the STFT) moves only along frequency and sums the complex coefficients, so each column keeps the sum that
 * inverts the STFT. Both concentrate a chirp or a tone onto a line.
 *
 * The STFTs are one-sided, on frames centred at samples $0, \text{hop}, 2\,\text{hop}, \dots$ (zeros beyond the ends),
 * with bins $f_s / n_\text{fft}$ apart; a value moved off the grid is dropped.
 */

import { fromData } from 'aifn-compute/foundation/tensor'
import { fft } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size, TimeFrequency } from 'aifn-compute/foundation/contracts'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'
import { readSamples, timeFrequency, type SignalInput } from '../signal'

/** Options of `reassignedSpectrogram` and `synchrosqueeze`. */
export type ReassignOptions = {
  /** The sample rate in Hz (default the signal's, or 1 for bare samples). */
  fs?: Scalar
  /** Window length in samples (default 128). */
  nperseg?: Size
  /** Samples between frames (default `nperseg`/8, rounded down; at least 1). */
  hop?: Size
  /** FFT length, raised to `nperseg` if smaller (default `nperseg`). */
  nfft?: Size
  /**
   * The window (default a symmetric Hann). A smooth window is needed: the derivative window is its central
   * difference.
   */
  window?: WindowInput
  /**
   * Cells whose magnitude is at or below this fraction of the peak magnitude are dropped from the sharpened raster,
   * neither moved nor kept (default 1e-6).
   */
  threshold?: Scalar
}

/**
 * The three short-time transforms with $g$, $g'$ and $t g$ on centred frames, each $[T, n_\text{fft}]$ complex,
 * interleaved (the caller reads the first `bins` of each row).
 *
 * @param v The samples; not modified.
 * @param options The window length, hop, FFT length and window; `fs` and `threshold` are not read here.
 * @returns The transforms `Vg`, `Vd` ($g'$) and `Vt` ($t g$), the frame centres in samples, their number `T`, `nfft`,
 *   the one-sided bin count, the hop, and the window's description for the raster.
 */
function transforms(v: Float64Array, options: ReassignOptions) {
  const L = options.nperseg ?? 128
  const nfft = Math.max(L, options.nfft ?? L)
  const hop = Math.max(1, Math.floor(options.hop ?? L / 8))
  const g = windowValues(options.window ?? 'hann', L, false)
  const half = Math.floor(L / 2)
  // Offsets m = j − half; the derivative in per-sample units by central differences (zero outside the window).
  const dg = Float64Array.from(g, (_, j) => ((j + 1 < L ? g[j + 1] : 0) - (j > 0 ? g[j - 1] : 0)) / 2)
  const tg = Float64Array.from(g, (w, j) => w * (j - half))
  const n = v.length
  const frames: number[] = []
  for (let c = 0; c < n; c += hop) frames.push(c)
  const T = frames.length
  const bins = Math.floor(nfft / 2) + 1
  const buf = (w: Float64Array) => {
    const d = new Float64Array(2 * T * nfft)
    frames.forEach((c, row) => {
      for (let j = 0; j < L; j++) {
        const m = j - half
        const i = c + m
        if (i < 0 || i >= n) continue
        const k = ((m % nfft) + nfft) % nfft
        d[2 * (row * nfft + k)] += v[i] * w[j]
      }
    })
    return fft(fromData(d, [T, nfft], 'complex128')).data as Float64Array
  }
  const w = options.window ?? 'hann'
  const name = typeof w === 'string' ? w : 'name' in w && typeof w.name === 'string' ? w.name : 'custom'
  return { Vg: buf(g), Vd: buf(dg), Vt: buf(tg), frames, T, nfft, bins, hop, window: { name, length: L, hop } }
}

/** A reassigned or synchrosqueezed raster, with the plain spectrogram or STFT magnitude it sharpens. */
export type Sharpened = TimeFrequency & {
  /**
   * The ordinary spectrogram $\abs{V_g}^2$ (reassignment) or $\abs{V_g}$ (synchrosqueezing), $[f, t]$ on the same
   * grid.
   */
  original: TimeFrequency['values']
}

/**
 * The reassigned spectrogram: each value $\abs{V_g(t, f)}^2$ moved to the nearest cell to $(\hat{t}, \hat{f})$ on the
 * same grid (frames `hop` apart, bins $f_s / n_\text{fft}$ apart, centred frames with zeros beyond the ends). Energy
 * is conserved, except for the cells below `threshold` and those moved off the grid, which are dropped; a linear
 * chirp collapses to its instantaneous-frequency line and an impulse to its time. The power is not scaled by the
 * window or $f_s$.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The window, its length, hop, FFT length, sample rate and threshold; see `ReassignOptions`.
 * @returns The reassigned power, `method: 'reassigned'`, and the plain spectrogram as `original`.
 *
 * @example A tone between bins
 * // 123 Hz lies between the 109 and 125 Hz bins: the spectrogram spreads it, reassignment puts it all in one bin.
 * const fs = 1000
 * const x = Array.from({ length: 512 }, (_, i) => Math.sin((2 * Math.PI * 123 * i) / fs))
 * const R = reassignedSpectrogram(x, { fs, nperseg: 64 })
 * const [bins, frames] = R.values.shape
 * const column = (v) => Array.from({ length: bins }, (_, k) => v.data[k * frames + frames / 2])
 * const share = (c) => Math.max(...c) / c.reduce((a, b) => a + b, 0)
 * print('share of the middle frame in its top bin: spectrogram', share(column(R.original)))
 * print('reassigned', share(column(R.values)), 'at', R.f.data[argmax(tensor(column(R.values)))], 'Hz')
 * print('total power: spectrogram', sum(R.original), ' reassigned', sum(R.values))
 */
export function reassignedSpectrogram(x: SignalInput, options: ReassignOptions = {}): Sharpened {
  const input = readSamples(x, 'reassignedSpectrogram', options.fs)
  const { fs, t0 } = input
  const { Vg, Vd, Vt, frames, T, nfft, bins, hop, window } = transforms(input.values, options)
  const out = new Float64Array(bins * T)
  const plain = new Float64Array(bins * T)
  let peak = 0
  for (let r = 0; r < T; r++)
    for (let k = 0; k < bins; k++) peak = Math.max(peak, Vg[2 * (r * nfft + k)] ** 2 + Vg[2 * (r * nfft + k) + 1] ** 2)
  const floor = (options.threshold ?? 1e-6) ** 2 * peak
  for (let r = 0; r < T; r++)
    for (let k = 0; k < bins; k++) {
      const p = 2 * (r * nfft + k)
      const [a, b] = [Vg[p], Vg[p + 1]]
      const power = a * a + b * b
      plain[k * T + r] = power
      if (power <= floor) continue
      // V_d / V_g and V_t / V_g.
      const dIm = (Vd[p + 1] * a - Vd[p] * b) / power
      const tRe = (Vt[p] * a + Vt[p + 1] * b) / power
      const kHat = Math.round(k - (dIm * nfft) / (2 * Math.PI))
      const rHat = Math.round((frames[r] + tRe) / hop)
      if (kHat < 0 || kHat >= bins || rHat < 0 || rHat >= T) continue
      out[kHat * T + rHat] += power
    }
  return {
    ...grid(t0, fs, frames, nfft, bins, out, 'reassigned', 'power', window),
    original: fromData(plain, [bins, T]),
  }
}

/**
 * The synchrosqueezed STFT: each coefficient $V_g(t, f)$ added at $(t, \hat{f})$, its instantaneous-frequency estimate
 * rounded to a bin, so the columns stay aligned in time and $\sum_f$ of the squeezed column equals that of the
 * one-sided STFT column, from which the signal is recovered (up to the window's value at zero). Coefficients below
 * `threshold` or moved off the grid are dropped. Returns the magnitude $\abs{T(t, f)}$ (`quantity: 'amplitude'`), not
 * the complex values.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The window, its length, hop, FFT length, sample rate and threshold; see `ReassignOptions`.
 * @returns The squeezed magnitude, `method: 'synchrosqueezed'`, and the STFT magnitude as `original`.
 *
 * @example A tone between bins
 * const fs = 1000
 * const x = Array.from({ length: 512 }, (_, i) => Math.sin((2 * Math.PI * 123 * i) / fs))
 * const S = synchrosqueeze(x, { fs, nperseg: 64 })
 * const [bins, frames] = S.values.shape
 * const column = (v) => Array.from({ length: bins }, (_, k) => v.data[k * frames + frames / 2])
 * const share = (c) => Math.max(...c) / c.reduce((a, b) => a + b, 0)
 * print('share of the middle frame in its top bin: STFT', share(column(S.original)))
 * print('squeezed', share(column(S.values)), 'at', S.f.data[argmax(tensor(column(S.values)))], 'Hz')
 */
export function synchrosqueeze(x: SignalInput, options: ReassignOptions = {}): Sharpened {
  const input = readSamples(x, 'synchrosqueeze', options.fs)
  const { fs, t0 } = input
  const { Vg, Vd, frames, T, nfft, bins, window } = transforms(input.values, options)
  const re = new Float64Array(bins * T)
  const im = new Float64Array(bins * T)
  const plain = new Float64Array(bins * T)
  let peak = 0
  for (let r = 0; r < T; r++)
    for (let k = 0; k < bins; k++) peak = Math.max(peak, Math.hypot(Vg[2 * (r * nfft + k)], Vg[2 * (r * nfft + k) + 1]))
  const floor = (options.threshold ?? 1e-6) * peak
  for (let r = 0; r < T; r++)
    for (let k = 0; k < bins; k++) {
      const p = 2 * (r * nfft + k)
      const [a, b] = [Vg[p], Vg[p + 1]]
      const mag = Math.hypot(a, b)
      plain[k * T + r] = mag
      if (mag <= floor) continue
      const dIm = (Vd[p + 1] * a - Vd[p] * b) / (mag * mag)
      const kHat = Math.round(k - (dIm * nfft) / (2 * Math.PI))
      if (kHat < 0 || kHat >= bins) continue
      re[kHat * T + r] += a
      im[kHat * T + r] += b
    }
  const amp = Float64Array.from(re, (v, i) => Math.hypot(v, im[i]))
  return {
    ...grid(t0, fs, frames, nfft, bins, amp, 'synchrosqueezed', 'amplitude', window),
    original: fromData(plain, [bins, T]),
  }
}

/**
 * The `TimeFrequency` raster of a sharpened transform on the STFT grid.
 *
 * @param t0 The time of sample 0, in seconds.
 * @param fs The sample rate in Hz.
 * @param frames The frame centres, in samples.
 * @param nfft The FFT length: bin $k$ is at $k f_s / n_\text{fft}$.
 * @param bins The number of one-sided bins.
 * @param values The raster, $[\text{bins}, T]$ row-major; kept, not copied.
 * @param method Which sharpening made it.
 * @param quantity `'power'` (reassignment) or `'amplitude'` (synchrosqueezing).
 * @param window The window's name, length and hop.
 * @returns The raster with its time and frequency axes.
 */
function grid(
  t0: Scalar,
  fs: Scalar,
  frames: number[],
  nfft: Size,
  bins: Size,
  values: Float64Array,
  method: 'reassigned' | 'synchrosqueezed',
  quantity: 'power' | 'amplitude',
  window: { name: string; length: Size; hop: Size },
): TimeFrequency {
  return timeFrequency({
    t: fromData(
      Float64Array.from(frames, (c) => t0 + c / fs),
      [frames.length],
    ),
    f: fromData(
      Float64Array.from({ length: bins }, (_, k) => (k * fs) / nfft),
      [bins],
    ),
    values: fromData(values, [bins, frames.length]),
    quantity,
    method,
    frequencyScale: 'linear',
    window,
  })
}
