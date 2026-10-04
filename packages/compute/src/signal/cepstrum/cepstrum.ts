/**
 * The cepstrum (Bogert, Healy and Tukey, 1963; Childers, Skinner and Kemerait, 1977, Proc. IEEE 65(10)): the inverse
 * Fourier transform of the log spectrum. Convolution becomes addition in the log spectrum, so an excitation and the
 * filter it drives separate along quefrency: a periodic pulse train of period P puts peaks at P, 2P, …, and a smooth
 * vocal-tract envelope stays near quefrency 0. The real cepstrum keeps log |X| only; the complex cepstrum also keeps
 * the unwrapped phase (linear phase removed), and is invertible. Cepstral pitch: the strongest peak in a quefrency
 * range (Noll, 1967, J. Acoust. Soc. Am. 41(2)).
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { fft, ifft } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'
import { readSamples, unwrapPhase, type SignalInput } from '../signal'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The DFT of real samples (zero-padded or truncated to n) as interleaved (re, im). */
function spectrumOf(v: Float64Array, n: Size): Float64Array {
  return fft(fromData(Float64Array.from(v)), { n }).data as Float64Array
}

/** The real part of the inverse DFT of interleaved (re, im). */
function inverseReal(d: Float64Array, n: Size): Float64Array {
  const y = ifft(fromData(d, [n], 'complex128')).data as Float64Array
  return Float64Array.from({ length: n }, (_, k) => y[2 * k])
}

/**
 * The real cepstrum c[q] = Re F⁻¹{log |F{x}|}[q] over quefrencies q = 0 … n − 1 samples (`n` pads or truncates,
 * default the length). Bins with |X| = 0 are floored at 1e-300 of the largest so the log stays finite. Even in q.
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

/** The complex cepstrum and the linear-phase delay removed before it (needed to invert). */
export type ComplexCepstrum = { cepstrum: Tensor; delay: Size }

/**
 * The complex cepstrum ĉ = F⁻¹{log |X| + i arg X} with the phase unwrapped and its linear part removed (an integer
 * circular delay of `delay` samples, the rounded phase at the centre bin over π), as `python-acoustics`' and MATLAB's
 * `cceps`. Real for a real input. `inverseComplexCepstrum` undoes it.
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

/** The signal whose complex cepstrum is `cepstrum` with the linear-phase `delay` restored (the inverse of `complexCepstrum`). */
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

/** A log-magnitude envelope over frequency. */
export type CepstralEnvelope = { f: Tensor; logMagnitude: Tensor }

/**
 * The spectral envelope by liftering: keep the real cepstrum below quefrency `lifter` samples (both ends, as it is
 * even), transform back, and read the real part as log |X| smoothed (natural log). The fine harmonic structure of a
 * voiced sound sits at quefrencies ≥ one period, so a lifter shorter than the period keeps only the formants. Returns
 * the first nfft/2 + 1 bins and their frequencies (fs units).
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
  fs?: Scalar
  /** Search range in Hz (default 60–500, a voice). */
  fmin?: Scalar
  fmax?: Scalar
  /** The analysis window (default Hann over the whole input). */
  window?: WindowInput
  /** FFT length (default the next power of two ≥ the length). */
  nfft?: Size
}

/** The cepstral pitch estimate and the cepstrum it was read from. */
export type CepstralPitch = {
  /** fs / q⋆, Hz. */
  f0: Scalar
  /** The quefrency of the peak, in seconds (q⋆/fs), refined by a parabola through the peak and its neighbours. */
  quefrency: Scalar
  /** The cepstrum's value at the peak. */
  peak: Scalar
  /** The real cepstrum, and its quefrency axis in seconds (the first half, q < nfft/2). */
  cepstrum: Tensor
  q: Tensor
}

/**
 * Pitch from the real cepstrum of a windowed frame: the largest cepstral value at quefrencies between 1/fmax and 1/fmin
 * seconds; a harmonic spectrum with spacing f₀ has a log spectrum periodic in f with period f₀, whose Fourier series
 * peaks at quefrency 1/f₀.
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

/** The vertex offset in (−½, ½) of the parabola through (−1, a), (0, b), (1, c). */
export function parabola(a: number, b: number, c: number): number {
  const den = a - 2 * b + c
  if (!(Math.abs(den) > 0) || !Number.isFinite(den)) return 0
  const d = (0.5 * (a - c)) / den
  return Math.max(-0.5, Math.min(0.5, d))
}
