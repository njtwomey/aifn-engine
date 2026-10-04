/**
 * Sharpened spectrograms. The reassignment method (Kodera, Gendrin and de Villedary, 1978; Auger and Flandrin, 1995,
 * IEEE Trans. Signal Process. 43(5)) moves each spectrogram value from the centre of its cell to the local centre of
 * gravity of the energy: the instantaneous frequency f̂ = f − Im(V_{g′}/V_g)/2π and the group delay
 * t̂ = t + Re(V_{tg}/V_g), computed from two extra STFTs with the windows g′ (derivative) and t·g. Synchrosqueezing
 * (Daubechies, Lu and Wu, 2011, Appl. Comput. Harmon. Anal. 30(2); Thakur and Wu, 2011, for the STFT) moves only along
 * frequency and keeps the complex coefficients, so the result stays invertible. Both concentrate a chirp or a tone onto
 * a line.
 */

import { fromData } from 'aifn-compute/foundation/tensor'
import { fft } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size, TimeFrequency } from 'aifn-compute/foundation/contracts'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'
import { readSamples, timeFrequency, type SignalInput } from '../signal'

/** Options of `reassignedSpectrogram` and `synchrosqueeze`. */
export type ReassignOptions = {
  fs?: Scalar
  /** Window length in samples (default 128). */
  nperseg?: Size
  /** Samples between frames (default nperseg/8). */
  hop?: Size
  /** FFT length ≥ nperseg (default nperseg). */
  nfft?: Size
  /** Default Hann. A smooth window is needed: the derivative window is its central difference. */
  window?: WindowInput
  /** Cells below this fraction of the peak magnitude are not moved (default 1e-6). */
  threshold?: Scalar
}

/** The three short-time transforms with g, g′ and t·g on centred frames, one-sided: [frames, bins] interleaved. */
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
  /** The ordinary spectrogram |V_g|² (reassignment) or |V_g| (synchrosqueezing), [f, t] on the same grid. */
  original: TimeFrequency['values']
}

/**
 * The reassigned spectrogram: each value |V_g(t, f)|² moved to (t̂, f̂) on the same grid (frames `hop` apart, bins
 * fs/nfft apart, centred frames with zeros beyond the ends). Energy is conserved; a linear chirp collapses to its
 * instantaneous-frequency line and an impulse to its time.
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
 * The synchrosqueezed STFT: each coefficient V_g(t, f) added at (t, f̂), its instantaneous-frequency estimate, so the
 * columns stay aligned in time and Σ_f of the squeezed column still reconstructs the signal (up to the window's value at
 * zero). Returns the magnitude |T(t, f)| (`quantity: 'amplitude'`).
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
