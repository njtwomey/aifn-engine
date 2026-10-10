/**
 * The cepstrum (Bogert, Healy and Tukey, 1963; Childers, Skinner and Kemerait, 1977, Proc. IEEE 65(10)): the inverse
 * Fourier transform of the log spectrum. Convolution becomes addition in the log spectrum, so an excitation and the
 * filter it drives separate along quefrency: a periodic pulse train of period $P$ puts peaks at $P, 2P, \dots$, and a
 * smooth vocal-tract envelope stays near quefrency 0. The real cepstrum keeps $\log \abs{X}$ only; the complex cepstrum
 * also keeps the unwrapped phase (linear phase removed), and is invertible. Cepstral pitch: the strongest peak in a
 * quefrency range (Noll, 1967, J. Acoust. Soc. Am. 41(2)).
 *
 * The transforms are circular DFTs of length $n$ (the input zero-padded or truncated), so quefrencies are in samples,
 * $0$ to $n - 1$, and wrap around: quefrency $n - q$ is $-q$.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { fft, ifft } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'
import { readSamples, unwrapPhase, type SignalInput } from '../signal'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The DFT of real samples (zero-padded or truncated to $n$) as interleaved (re, im).
 *
 * @param v The real samples; not modified.
 * @param n The DFT length.
 * @returns $2n$ values: the real and imaginary parts of bin $k$ at entries $2k$ and $2k + 1$.
 */
function spectrumOf(v: Float64Array, n: Size): Float64Array {
  return fft(fromData(Float64Array.from(v)), { n }).data as Float64Array
}

/**
 * The real part of the inverse DFT of interleaved (re, im).
 *
 * @param d $2n$ values: the real and imaginary parts of bin $k$ at entries $2k$ and $2k + 1$.
 * @param n The DFT length.
 * @returns The $n$ real parts; the imaginary parts are dropped.
 */
function inverseReal(d: Float64Array, n: Size): Float64Array {
  const y = ifft(fromData(d, [n], 'complex128')).data as Float64Array
  return Float64Array.from({ length: n }, (_, k) => y[2 * k])
}

/**
 * The real cepstrum $c[q] = \operatorname{Re} \mathcal{F}^{-1}\{\log \abs{\mathcal{F}\{x\}}\}[q]$ over quefrencies
 * $q = 0, \dots, n - 1$ samples (as scipy's `signal.ifft(log(abs(fft(x))))`, real part). Bins with $\abs{X} = 0$ are
 * floored at $10^{-300}$ of the largest so the log stays finite. Even in $q$: $c[q] = c[n - q]$.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples); its sample rate is not used.
 * @param options Options.
 * @param options.n The DFT length $n$: the samples are zero-padded or truncated to it (default their length).
 * @returns The $n$ cepstral values, indexed by quefrency in samples.
 *
 * @example An echo
 * // x = pulse + 0.5 pulse delayed by 20 samples: log|1 + 0.5 e^{-20 i w}| puts 0.5/2 at quefrency 20 (and 108 = -20),
 * // -0.5^2/4 at 40, 0.5^3/6 at 60.
 * const x = Array.from({ length: 128 }, (_, i) => (i === 0 ? 1 : i === 20 ? 0.5 : 0))
 * const c = realCepstrum(x).data
 * print('c[20], c[40], c[60] =', c[20], c[40], c[60])
 * print('c[108] =', c[108])
 */
export function realCepstrum(x: SignalInput, { n }: { n?: Size } = {}): Tensor {
  const v = readSamples(x, 'realCepstrum').values
  const size = n ?? v.length
  const X = spectrumOf(v, size)
  let peak = 0
  for (let k = 0; k < size; k++) peak = Math.max(peak, Math.hypot(X[2 * k], X[2 * k + 1]))
  const floor = 1e-300 * peak || 1e-300
  const L = new Float64Array(2 * size)
  for (let k = 0; k < size; k++) L[2 * k] = Math.log(Math.max(Math.hypot(X[2 * k], X[2 * k + 1]), floor))
  return fromData(inverseReal(L, size), [size])
}

/**
 * The complex cepstrum and the linear-phase delay removed before it (needed to invert): `cepstrum`, $n$ values
 * indexed by quefrency in samples, and `delay`, in samples.
 */
export type ComplexCepstrum = { cepstrum: Tensor; delay: Size }

/**
 * The complex cepstrum $\hat{c} = \mathcal{F}^{-1}\{\log \abs{X} + i \arg X\}$ with the phase unwrapped and its linear
 * part removed (an integer circular delay of `delay` samples, the rounded phase at the centre bin over $\pi$), as
 * `python-acoustics`' and MATLAB's `cceps`. The real part of the inverse DFT is returned, which for a real input is
 * the whole of it. `inverseComplexCepstrum` undoes it.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples); its sample rate is not used.
 * @param options Options.
 * @param options.n The DFT length $n$: the samples are zero-padded or truncated to it (default their length).
 * @returns The $n$ cepstral values and the `delay` removed: $\operatorname{round}(\phi_c / \pi)$, with $\phi_c$ the
 *   unwrapped phase at bin $\lfloor (n + 1)/2 \rfloor$, so negative for a signal delayed in time.
 *
 * @example A minimum-phase echo
 * // For x = pulse + 0.5 pulse delayed by 20, log(1 + 0.5 z^-20) = 0.5 z^-20 - 0.5^2/2 z^-40 + 0.5^3/3 z^-60 - ...
 * const x = Array.from({ length: 128 }, (_, i) => (i === 0 ? 1 : i === 20 ? 0.5 : 0))
 * const { cepstrum, delay } = complexCepstrum(x)
 * print('delay =', delay)
 * print('c[20], c[40], c[60] =', cepstrum.data[20], cepstrum.data[40], cepstrum.data[60])
 */
export function complexCepstrum(x: SignalInput, { n }: { n?: Size } = {}): ComplexCepstrum {
  const v = readSamples(x, 'complexCepstrum').values
  const size = n ?? v.length
  const X = spectrumOf(v, size)
  const phase = new Float64Array(size)
  for (let k = 0; k < size; k++) phase[k] = Math.atan2(X[2 * k + 1], X[2 * k])
  const unwrapped = unwrapPhase(phase).data as Float64Array
  const centre = Math.floor((size + 1) / 2)
  const delay = Math.round(unwrapped[centre] / Math.PI)
  const L = new Float64Array(2 * size)
  for (let k = 0; k < size; k++) {
    L[2 * k] = Math.log(Math.max(Math.hypot(X[2 * k], X[2 * k + 1]), 1e-300))
    L[2 * k + 1] = unwrapped[k] - (Math.PI * delay * k) / centre
  }
  return { cepstrum: fromData(inverseReal(L, size), [size]), delay }
}

/**
 * The signal whose complex cepstrum is `cepstrum` with the linear-phase `delay` restored (the inverse of
 * `complexCepstrum`): the DFT of the cepstrum is exponentiated, the delay's linear phase added back, and the real part
 * of the inverse DFT returned.
 *
 * @param cepstrum The complex cepstrum, as `complexCepstrum` returns it ($n$ values).
 * @param delay The linear-phase delay in samples that `complexCepstrum` removed.
 * @returns The $n$ samples of the signal.
 *
 * @example A delayed echo, there and back
 * const x = Array.from({ length: 64 }, (_, i) => (i === 10 ? 1 : i === 15 ? 0.5 : 0))
 * const { cepstrum, delay } = complexCepstrum(x)
 * print('delay =', delay)
 * const y = inverseComplexCepstrum(cepstrum, delay)
 * print('y[10], y[15] =', y.data[10], y.data[15], ' largest error =', max(abs(sub(y, tensor(x)))))
 */
export function inverseComplexCepstrum(cepstrum: SignalInput, delay: Size): Tensor {
  const c = readSamples(cepstrum, 'inverseComplexCepstrum').values
  const size = c.length
  const C = spectrumOf(c, size)
  const centre = Math.floor((size + 1) / 2)
  const out = new Float64Array(2 * size)
  for (let k = 0; k < size; k++) {
    const mag = Math.exp(C[2 * k])
    const ph = C[2 * k + 1] + (Math.PI * delay * k) / centre
    out[2 * k] = mag * Math.cos(ph)
    out[2 * k + 1] = mag * Math.sin(ph)
  }
  return fromData(inverseReal(out, size), [size])
}

/**
 * A log-magnitude envelope over frequency: `f`, the frequencies of the bins in Hz, and `logMagnitude`, the smoothed
 * natural log of $\abs{X}$ at each.
 */
export type CepstralEnvelope = { f: Tensor; logMagnitude: Tensor }

/**
 * The spectral envelope by liftering: keep the real cepstrum below quefrency `lifter` samples (both ends, as it is
 * even), transform back, and read the real part as $\log \abs{X}$ smoothed (natural log). The fine harmonic structure
 * of a voiced sound sits at quefrencies $\ge$ one period, so a lifter shorter than the period keeps only the formants.
 * Returns the first $n_\text{fft}/2 + 1$ bins and their frequencies in Hz. No window is applied.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options Options.
 * @param options.lifter The number of low quefrencies kept, in samples: $q < \text{lifter}$ and the mirror
 *   $n_\text{fft} - q < \text{lifter}$.
 * @param options.fs The sample rate in Hz (default the signal's, or 1 for bare samples); sets the frequency axis only.
 * @param options.nfft The DFT length $n_\text{fft}$ (default the next power of two $\ge$ the length).
 * @returns The envelope over the $\lfloor n_\text{fft}/2 \rfloor + 1$ bins from 0 Hz to $f_s/2$.
 *
 * @example A formant under harmonics
 * // Harmonics of 100 Hz with amplitudes peaking at 1 kHz: the envelope peaks near 1 kHz, not at a harmonic.
 * const fs = 8000
 * const x = Array.from({ length: 512 }, (_, i) => {
 *   let s = 0
 *   for (let k = 1; k <= 30; k++) {
 *     s += Math.cos((2 * Math.PI * 100 * k * i) / fs) * Math.exp(-(((100 * k - 1000) / 300) ** 2))
 *   }
 *   return s
 * })
 * const e = cepstralEnvelope(x, { fs, lifter: 20 })
 * print('bins =', e.f.shape[0], ' envelope peak at', e.f.data[argmax(e.logMagnitude)], 'Hz')
 */
export function cepstralEnvelope(
  x: SignalInput,
  { lifter = 30, fs, nfft }: { lifter?: Size; fs?: Scalar; nfft?: Size } = {},
): CepstralEnvelope {
  const input = readSamples(x, 'cepstralEnvelope', fs)
  const size = nfft ?? 2 ** Math.ceil(Math.log2(Math.max(2, input.values.length)))
  const c = realCepstrum(input.values, { n: size }).data as Float64Array
  const kept = new Float64Array(2 * size)
  for (let q = 0; q < size; q++) if (q < lifter || size - q < lifter) kept[2 * q] = c[q]
  const L = fft(fromData(kept, [size], 'complex128')).data as Float64Array
  const half = Math.floor(size / 2) + 1
  return {
    f: fromData(
      Float64Array.from({ length: half }, (_, k) => (k * input.fs) / size),
      [half],
    ),
    logMagnitude: fromData(
      Float64Array.from({ length: half }, (_, k) => L[2 * k]),
      [half],
    ),
  }
}

/** Options of `cepstralPitch`. */
export type CepstralPitchOptions = {
  /** The sample rate in Hz (default the signal's, or 1 for bare samples). */
  fs?: Scalar
  /** The lowest pitch searched, in Hz (default 60; with `fmax`, the range of a voice). */
  fmin?: Scalar
  /** The highest pitch searched, in Hz (default 500). */
  fmax?: Scalar
  /** The analysis window (default a symmetric Hann over the whole input). */
  window?: WindowInput
  /** FFT length (default the next power of two $\ge$ the length). */
  nfft?: Size
}

/** The cepstral pitch estimate and the cepstrum it was read from. */
export type CepstralPitch = {
  /** The pitch $f_s / q^\star$, in Hz. */
  f0: Scalar
  /**
   * The quefrency of the peak, in seconds ($q^\star / f_s$), refined by a parabola through the peak and its
   * neighbours.
   */
  quefrency: Scalar
  /** The cepstrum's value at the peak (at the integer quefrency, before refinement). */
  peak: Scalar
  /** The real cepstrum of the windowed frame, its first half ($q < n_\text{fft}/2$). */
  cepstrum: Tensor
  /** The quefrency axis of `cepstrum`, in seconds. */
  q: Tensor
}

/**
 * Pitch from the real cepstrum of a windowed frame: the largest cepstral value at quefrencies between $1/f_{\max}$ and
 * $1/f_{\min}$ seconds (in samples, $\lfloor f_s / f_{\max} \rfloor$ to $\lceil f_s / f_{\min} \rceil$, at least 1 and
 * below $n_\text{fft}/2 - 1$); a harmonic spectrum with spacing $f_0$ has a log spectrum periodic in $f$ with period
 * $f_0$, whose Fourier series peaks at quefrency $1/f_0$. Throws `DomainError` when that range is empty (the frame is
 * too short for the pitch range).
 *
 * @param x The frame: a single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The sample rate, pitch range, window and FFT length; see `CepstralPitchOptions`.
 * @returns The pitch, the refined quefrency, the peak value, and the first half of the cepstrum with its axis.
 *
 * @example A 200 Hz harmonic tone
 * const fs = 8000
 * const x = Array.from({ length: 1024 }, (_, i) => {
 *   let s = 0
 *   for (let k = 1; k <= 10; k++) s += Math.sin((2 * Math.PI * 200 * k * i) / fs) / k
 *   return s
 * })
 * const p = cepstralPitch(x, { fs })
 * print('f0 =', p.f0, 'Hz  quefrency =', p.quefrency, 's')
 */
export function cepstralPitch(x: SignalInput, options: CepstralPitchOptions = {}): CepstralPitch {
  const input = readSamples(x, 'cepstralPitch', options.fs)
  const { fs } = input
  const { fmin = 60, fmax = 500 } = options
  const v = input.values
  const w = windowValues(options.window ?? 'hann', v.length, false)
  const frame = Float64Array.from(v, (u, i) => u * w[i])
  const nfft = options.nfft ?? 2 ** Math.ceil(Math.log2(Math.max(2, v.length)))
  const c = realCepstrum(frame, { n: nfft }).data as Float64Array
  const half = Math.floor(nfft / 2)
  const lo = Math.max(1, Math.floor(fs / fmax))
  const hi = Math.min(half - 1, Math.ceil(fs / fmin))
  if (!(hi > lo)) throw new DomainError('cepstralPitch', 'cepstralPitch: the frame is too short for the pitch range')
  let best = lo
  for (let q = lo; q <= hi; q++) if (c[q] > c[best]) best = q
  const shift = parabola(c[best - 1], c[best], c[best + 1])
  const qStar = best + shift
  return {
    f0: fs / qStar,
    quefrency: qStar / fs,
    peak: c[best],
    cepstrum: fromData(c.slice(0, half), [half]),
    q: fromData(
      Float64Array.from({ length: half }, (_, k) => k / fs),
      [half],
    ),
  }
}

/**
 * The vertex offset of the parabola through $(-1, a)$, $(0, b)$, $(1, c)$, clamped to $[-\tfrac12, \tfrac12]$; 0 when
 * the three points are collinear or not finite.
 *
 * @param a The value one sample before the extremum.
 * @param b The value at the extremum.
 * @param c The value one sample after the extremum.
 * @returns The offset of the vertex from the middle point, in samples.
 */
export function parabola(a: number, b: number, c: number): number {
  const den = a - 2 * b + c
  if (!(Math.abs(den) > 0) || !Number.isFinite(den)) return 0
  const d = (0.5 * (a - c)) / den
  return Math.max(-0.5, Math.min(0.5, d))
}
