/**
 * The analytic signal and Hilbert spectral analysis: `hilbert` (as `scipy.signal.hilbert`), instantaneous amplitude,
 * phase and frequency, and the Hilbert spectrum of a set of intrinsic mode functions (Huang et al., 1998, Proc. R. Soc.
 * Lond. A 454).
 *
 * The analytic signal $z = x + i\mathcal{H}\{x\}$ is computed by DFT, so it treats the record as periodic: values near
 * the ends are distorted unless the signal wraps smoothly. Frequencies are in Hz at the signal's sample rate (or the
 * `fs` option), and in cycles per sample for bare samples.
 */

import { copy, fromData, imagPart, mul, realPart, type Tensor } from 'aifn-compute/foundation/tensor'
import { fft, ifft } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size, TimeFrequency } from 'aifn-compute/foundation/contracts'
import { readSamples, timeFrequency, unwrapPhase, type SignalInput } from '../signal'

/**
 * The analytic signal $z = x + i\mathcal{H}\{x\}$, as `scipy.signal.hilbert`: the FFT with negative frequencies zeroed
 * and positive ones doubled (DC and Nyquist kept once), inverted. Any length.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples); its sample rate is not used.
 * @param options Options.
 * @param options.n The DFT length $n$: the samples are zero-padded or truncated to it first (default their length).
 * @returns $z$, a complex128 tensor of shape $[n]$ whose real part is $x$ (padded or truncated).
 *
 * @example A cosine becomes a complex exponential
 * // cos is the real part; the Hilbert transform, the imaginary part, is sin.
 * print(hilbert([1, 0, -1, 0]))
 */
export function hilbert(x: SignalInput, { n }: { n?: Size } = {}): Tensor {
  const v = readSamples(x, 'hilbert').values
  const size = n ?? v.length
  const h = Float64Array.from({ length: size }, (_, k) =>
    k === 0 || (size % 2 === 0 && k === size / 2) ? 1 : k < size / 2 ? 2 : 0,
  )
  return ifft(mul(fft(fromData(v), { n: size }), fromData(h)))
}

/** Instantaneous amplitude, phase and frequency of a real signal. */
export interface Instantaneous {
  /** $\abs{z}$, the envelope. Length $n$. */
  amplitude: Tensor
  /** Unwrapped $\arg z$, radians. Length $n$. */
  phase: Tensor
  /**
   * Frequency from the phase increment $\arg(z[t+1]\, \bar{z}[t]) \cdot f_s / 2\pi$ (needs no unwrapping), for
   * $t = 0, \dots, n - 2$; the last value repeats so the length is $n$. Lies in $(-f_s/2, f_s/2]$.
   */
  frequency: Tensor
}

/**
 * Instantaneous amplitude, phase and frequency from `hilbert`, the frequency in Hz ($f_s$ from the signal or the `fs`
 * option; cycles per sample for bare samples).
 *
 * @param x The single-channel real signal (a `Signal`, or bare samples).
 * @param options `fs`, the sample rate in Hz, overriding the signal's (default the signal's, or 1 for bare samples).
 * @returns The amplitude $\abs{z}$, the unwrapped phase, and the frequency, each of length $n$.
 *
 * @example An amplitude-modulated tone
 * // A 10 Hz carrier whose amplitude swings between 0.5 and 1.5 at 2 Hz.
 * const fs = 100
 * const wave = (f, i) => Math.cos((2 * Math.PI * f * i) / fs)
 * const x = Array.from({ length: 200 }, (_, i) => (1 + 0.5 * wave(2, i)) * wave(10, i))
 * const r = instantaneous(x, { fs })
 * print('amplitude at 0, 0.25, 0.5 s =', r.amplitude.data[0], r.amplitude.data[25], r.amplitude.data[50])
 * print('frequency at 0.5 s =', r.frequency.data[50], 'Hz')
 */
export function instantaneous(x: SignalInput, options: { fs?: Scalar } = {}): Instantaneous {
  const { fs } = readSamples(x, 'instantaneous', options.fs)
  const z = hilbert(x)
  // Contiguous copies of the parts (realPart/imagPart are strided views of the interleaved storage).
  const re = copy(realPart(z)).data as Float64Array
  const im = copy(imagPart(z)).data as Float64Array
  const n = re.length
  const amp = new Float64Array(n)
  const ph = new Float64Array(n)
  const freq = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    amp[i] = Math.hypot(re[i], im[i])
    if (i + 1 < n) {
      const dr = re[i + 1] * re[i] + im[i + 1] * im[i]
      const di = im[i + 1] * re[i] - re[i + 1] * im[i]
      freq[i] = (Math.atan2(di, dr) * fs) / (2 * Math.PI)
    }
    ph[i] = Math.atan2(im[i], re[i])
  }
  if (n > 1) freq[n - 1] = freq[n - 2]
  return { amplitude: fromData(amp), phase: unwrapPhase(ph), frequency: fromData(freq) }
}

/**
 * The envelope $\abs{x + i\mathcal{H}\{x\}}$ of a real signal: the `amplitude` of `instantaneous`.
 *
 * @param x The single-channel real signal (a `Signal`, or bare samples).
 * @returns The $n$ values of the envelope.
 *
 * @example The envelope of a beat
 * // cos(9 w t) + cos(11 w t) = 2 cos(w t) cos(10 w t): the envelope is |2 cos(w t)|.
 * const wave = (f, i) => Math.cos((2 * Math.PI * f * i) / 40)
 * const x = Array.from({ length: 40 }, (_, i) => wave(9, i) + wave(11, i))
 * print('envelope =', envelope(x).data.slice(0, 11))
 * print('|2 cos| =', Array.from({ length: 11 }, (_, i) => Math.abs(2 * wave(1, i))))
 */
export function envelope(x: SignalInput): Tensor {
  return instantaneous(x).amplitude
}

/**
 * A Hilbert spectrum: a `TimeFrequency` raster (`method: 'hht'`, `quantity: 'amplitude'`) of amplitude summed per
 * (frequency bin, time bin), divided by the samples per time bin, and its marginal over time.
 */
export type HilbertSpectrum = TimeFrequency & {
  /** $h(f) = \sum_t H(t, f)$, summed over the time bins. */
  marginal: Tensor
}

/**
 * The Hilbert spectrum $H(t, f)$ of intrinsic mode functions: each mode's instantaneous amplitude is placed at its
 * instantaneous frequency on a grid of `timeBins` $\times$ `freqBins` cells over $[0, f_{\max})$ (Hz), and summed.
 * Each cell holds the summed amplitude divided by the samples per time bin, so a steady mode of amplitude $a$ puts
 * about $a$ in its cell. Samples at a negative frequency or at or above $f_{\max}$ are left out. The time axis gives
 * the bin centres from 0, ignoring the signal's `t0`.
 *
 * @param imfs The modes, e.g. from `emd`: single-channel signals (or bare samples) of one length $n$ (the first's is
 *   used, and not checked against the others).
 * @param options Options.
 * @param options.timeBins The number of time bins, each $n / \text{timeBins}$ samples (default 64).
 * @param options.freqBins The number of frequency bins, each $f_{\max} / \text{freqBins}$ wide (default 64).
 * @param options.fMax The top of the frequency axis $f_{\max}$, in Hz (default $f_s / 2$).
 * @param options.fs The sample rate in Hz, overriding the first mode's (default its rate, or 1 for bare samples).
 * @returns The raster $[\text{freqBins}, \text{timeBins}]$ with bin-centre axes, and its marginal over time.
 *
 * @example Two steady modes
 * // 12 Hz at amplitude 1 and 32 Hz at amplitude 0.5, binned 10 Hz wide.
 * const fs = 100
 * const a = Array.from({ length: 200 }, (_, i) => Math.cos((2 * Math.PI * 12 * i) / fs))
 * const b = Array.from({ length: 200 }, (_, i) => 0.5 * Math.cos((2 * Math.PI * 32 * i) / fs))
 * const H = hilbertSpectrum([a, b], { fs, timeBins: 4, freqBins: 5 })
 * print('f =', H.f, ' t =', H.t)
 * print('H =', H.values)
 * print('marginal =', H.marginal)
 */
export function hilbertSpectrum(
  imfs: readonly SignalInput[],
  options: { timeBins?: Size; freqBins?: Size; fMax?: Scalar; fs?: Scalar } = {},
): HilbertSpectrum {
  const fs = imfs.length ? readSamples(imfs[0], 'hilbertSpectrum', options.fs).fs : (options.fs ?? 1)
  const { timeBins = 64, freqBins = 64, fMax = fs / 2 } = options
  const tracks = imfs.map((m) => instantaneous(m, { fs }))
  const n = tracks[0]?.amplitude.shape[0] ?? 0
  const per = n / timeBins
  const power = new Float64Array(freqBins * timeBins)
  const marginal = new Float64Array(freqBins)
  for (const { amplitude, frequency } of tracks) {
    const a = amplitude.data
    const f = frequency.data
    for (let i = 0; i < n; i++) {
      const b = Math.floor((f[i] / fMax) * freqBins)
      if (b < 0 || b >= freqBins) continue
      power[b * timeBins + Math.min(timeBins - 1, Math.floor(i / per))] += a[i] / per
      marginal[b] += a[i] / per
    }
  }
  return {
    ...timeFrequency({
      t: fromData(
        Float64Array.from({ length: timeBins }, (_, j) => ((j + 0.5) * per) / fs),
        [timeBins],
      ),
      f: fromData(
        Float64Array.from({ length: freqBins }, (_, b) => ((b + 0.5) * fMax) / freqBins),
        [freqBins],
      ),
      values: fromData(power, [freqBins, timeBins]),
      quantity: 'amplitude',
      method: 'hht',
      frequencyScale: 'linear',
    }),
    marginal: fromData(marginal, [freqBins]),
  }
}
