/**
 * Multirate signal processing, as scipy.signal: rational resampling by a polyphase FIR (`resamplePoly`), decimation
 * with an anti-aliasing filter (`decimateSignal`, scipy's `decimate`), the polyphase decomposition of a filter
 * (`polyphase`), the uniform DFT analysis filter bank computed through it (`dftFilterBank`), and band-limited
 * (Whittaker–Shannon) interpolation at arbitrary times (`sincInterpolate`). Everything but the last is a composition
 * over `aifn-compute/foundation/convolution`'s `upfirdn` and `conv`, `aifn-compute/foundation/fourier`'s `ifft` and the
 * filter design and filtering of `aifn-compute/signal/filters` (Crochiere and Rabiner, 1981, "Interpolation and
 * decimation of digital signals: a tutorial review", Proc. IEEE 69(3); Vaidyanathan, 1993, "Multirate Systems and
 * Filter Banks").
 *
 * Inputs are single-channel: a `Signal`, whose rate $f_s$, start time $t_0$ and unit are carried to the result, or
 * bare samples, taken at $f_s = 1$ from $t_0 = 0$. Rates and factors that are not positive integers throw
 * `ShapeError`.
 */

import { ShapeError } from 'aifn-compute/foundation/errors'
import { conv, upfirdn } from 'aifn-compute/foundation/convolution'
import { ifft } from 'aifn-compute/foundation/fourier'
import {
  add,
  dense,
  fromData,
  mul,
  reshape,
  slice,
  tensor,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Scalar, Size, VectorLike } from 'aifn-compute/foundation/contracts'
import { cheby1, filtfilt, firwin, sosfilt } from 'aifn-compute/signal/filters'
import type { WindowSpec } from 'aifn-compute/signal/windows'
import { readSamples, signal, type Signal, type SignalInput } from '../signal'

/**
 * The greatest common divisor of two non-negative integers, by Euclid's algorithm.
 *
 * @param a The first integer.
 * @param b The second integer; 0 returns `a`.
 * @returns $\gcd(a, b)$.
 */
const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))

/**
 * Throws `ShapeError` unless `v` is an integer of at least 1.
 *
 * @param v The value to check.
 * @param what The value's name in the error message (`'up'`, `'q'`, ...).
 * @param where The caller's name, for error messages.
 */
function positiveInteger(v: number, what: string, where: string) {
  if (!(Number.isInteger(v) && v >= 1)) throw new ShapeError(where, `${where}: ${what} must be a positive integer`)
}

/**
 * The output of a resampler: a `Signal` at the new rate, with the input's start time and unit.
 *
 * @param values The resampled samples, a rank-1 tensor.
 * @param fs The new sample rate, in samples per second.
 * @param t0 The time of sample 0, in seconds, kept from the input.
 * @param unit The unit of the values, kept from the input; left out of the result when undefined.
 * @returns The `Signal`.
 */
const resampled = (values: Value, fs: Scalar, t0: Scalar, unit: string | undefined): Signal =>
  signal(values as Tensor, { fs, t0, ...(unit === undefined ? {} : { unit }) })

/** Options for `resamplePoly`. */
export type ResamplePolyOptions = {
  /**
   * The anti-aliasing FIR: a window for `firwin` (default Kaiser with $\beta = 5$, scipy's default) applied to a
   * lowpass with cutoff $1/\max(u, d)$ of Nyquist and $20\max(u, d) + 1$ taps, where $u/d$ is `up`/`down` in lowest
   * terms; or the filter's taps themselves, designed at the upsampled rate (they are scaled by $u$, and their delay is
   * taken as $\lfloor (L - 1)/2 \rfloor$ for $L$ taps, so they should be linear phase and of odd length).
   */
  window?: WindowSpec | VectorLike
  /**
   * How the signal is extended beyond its ends: `constant` (zeros, default) or a statistic of the signal (`mean`,
   * `median`, `minimum`, `maximum`), which is subtracted before filtering and added back.
   */
  padtype?: 'constant' | 'mean' | 'median' | 'minimum' | 'maximum'
}

/**
 * True when the `window` option is a window spec (a name or an object with `name`) rather than the filter's taps.
 *
 * @param w The `window` option of `ResamplePolyOptions`.
 * @returns Whether `w` is a `WindowSpec`.
 */
const isWindowSpec = (w: unknown): w is WindowSpec =>
  typeof w === 'string' || (typeof w === 'object' && w !== null && 'name' in w)

/**
 * Resample by the rational factor $u/d$ with a polyphase FIR, as `scipy.signal.resample_poly`: reduce `up`/`down` by
 * their gcd to $u/d$, design (or take) a lowpass $h$ scaled by $u$, pad it so the output is centred on the input, and
 * keep the $\lceil n u / d \rceil$ samples of `upfirdn(h, x, u, d)` that align with the input's span ($n$ input
 * samples). The input is a single-channel `SignalInput`; the result is a `Signal` at $f_s u / d$. When $u = d$ the
 * samples are returned unfiltered. Throws `ShapeError` when `up` or `down` is not a positive integer.
 *
 * @param x The signal: a single-channel `Signal`, or bare samples at rate 1.
 * @param up The upsampling factor (a positive integer).
 * @param down The downsampling factor (a positive integer).
 * @param options The anti-aliasing filter (`window`) and how the signal is extended beyond its ends (`padtype`).
 * @returns The resampled `Signal`, of $\lceil n u / d \rceil$ samples at rate $f_s u / d$, with the input's $t_0$ and
 *   unit.
 *
 * @example Upsample a sine by 2
 * // Two periods of 8 samples; away from the ends the new samples land on the sine, at half-sample times.
 * const x = Array.from({ length: 16 }, (_, k) => Math.sin((2 * Math.PI * k) / 8))
 * const y = resamplePoly(x, 2, 1)
 * print('fs =', y.fs)
 * print('y[12..19] =', slice(y.data, [12, 20]))
 * print('sine at t = 6, 6.5, ..., 9.5 =', [12, 13, 14, 15, 16, 17, 18, 19].map((j) => Math.sin((Math.PI * j) / 8)))
 *
 * @example Padding with the mean removes the edge ripple of a constant
 * // As scipy.signal.resample_poly: zero padding rings at the ends, padtype 'mean' gives exact ones.
 * print('constant =', resamplePoly([1, 1, 1, 1, 1, 1], 2, 1).data)
 * print('mean =', resamplePoly([1, 1, 1, 1, 1, 1], 2, 1, { padtype: 'mean' }).data)
 *
 * @example 48 kHz to 44.1 kHz, the factor 147/160
 * // 10 ms of a 1 kHz tone, as a Signal so that the rate is carried through.
 * const tone = Array.from({ length: 480 }, (_, k) => Math.sin((2 * Math.PI * k) / 48))
 * const x = { kind: 'signal', data: tensor(tone), fs: 48000, t0: 0 }
 * const y = resamplePoly(x, 147, 160)
 * print('samples', x.data.shape[0], '->', y.data.shape[0])
 * print('fs =', y.fs)
 */
export function resamplePoly(x: SignalInput, up: Size, down: Size, options: ResamplePolyOptions = {}): Signal {
  const where = 'resamplePoly'
  positiveInteger(up, 'up', where)
  positiveInteger(down, 'down', where)
  const s = readSamples(x, where)
  const g = gcd(up, down)
  const [u, d] = [up / g, down / g]
  if (u === 1 && d === 1) return resampled(tensor(s.values), s.fs, s.t0, s.unit)
  const n = s.values.length
  const nOut = Math.ceil((n * u) / d)
  const w = options.window ?? { name: 'kaiser', beta: 5 }
  let taps: Float64Array
  let halfLen: number
  if (isWindowSpec(w)) {
    const maxRate = Math.max(u, d)
    halfLen = 10 * maxRate
    taps = Float64Array.from(toFlat(firwin(2 * halfLen + 1, 1 / maxRate, { window: w }).repr.b))
  } else {
    taps = Float64Array.from(w as ArrayLike<number>)
    halfLen = Math.floor((taps.length - 1) / 2)
  }
  for (let k = 0; k < taps.length; k++) taps[k] *= u
  // Pad h in front so the first output sample lands on the first input sample, and behind so enough samples exist.
  const prePad = d - (halfLen % d)
  const preRemove = Math.floor((halfLen + prePad) / d)
  const outLength = (taps: number) => Math.floor(((n - 1) * u + taps - 1) / d) + 1
  let postPad = 0
  while (outLength(taps.length + prePad + postPad) < nOut + preRemove) postPad++
  const h = new Float64Array(prePad + taps.length + postPad)
  h.set(taps, prePad)
  const padtype = options.padtype ?? 'constant'
  const background = padtype === 'constant' ? 0 : statistic(s.values, padtype)
  const centred = background === 0 ? s.values : s.values.map((v) => v - background)
  const y = upfirdn(tensor(h), tensor(centred), { up: u, down: d }) as Tensor
  const kept = slice(y, [preRemove, preRemove + nOut]) as Tensor
  const out = background === 0 ? kept : add(kept, background)
  return resampled(out, (s.fs * u) / d, s.t0, s.unit)
}

/**
 * A summary statistic of the samples, the background `resamplePoly` subtracts before filtering for a non-constant
 * `padtype`.
 *
 * @param v The samples; not modified.
 * @param kind Which statistic: the mean, the median (the mean of the two middle values for an even count), the
 *   minimum or the maximum.
 * @returns The statistic (NaN for the mean of no samples).
 */
function statistic(v: Float64Array, kind: 'mean' | 'median' | 'minimum' | 'maximum'): number {
  if (kind === 'mean') return v.reduce((a, b) => a + b, 0) / v.length
  if (kind === 'minimum') return Math.min(...v)
  if (kind === 'maximum') return Math.max(...v)
  const s = Float64Array.from(v).sort()
  const m = s.length >> 1
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m])
}

/** Options for `decimateSignal`. */
export type DecimateOptions = {
  /** The filter order: default 8 for `iir` (Chebyshev I), $20q$ for `fir` (Hamming window). */
  n?: Size
  /**
   * `iir` (default): Chebyshev type I, 0.05 dB ripple, edge $0.8/q$ of Nyquist; `fir`: a Hamming-windowed-sinc lowpass
   * at $1/q$ of Nyquist.
   */
  ftype?: 'iir' | 'fir'
  /**
   * Filter without phase shift (default true). The FIR case then resamples by $1/q$ with the filter (`resamplePoly`,
   * which compensates its delay), the IIR case filters forwards and backwards (`filtfilt` on second-order sections).
   * `false`: filter forwards only, so the output is delayed by the filter.
   */
  zeroPhase?: boolean
}

/**
 * Downsample by the integer factor $q$ after an anti-aliasing lowpass, as `scipy.signal.decimate` (named apart from
 * `aifn-compute/foundation/trace`'s `decimate`, which thins a trace). The result is a `Signal` of $\lceil n/q \rceil$
 * samples at $f_s/q$: samples $0, q, 2q, \dots$ of the filtered input. Throws `ShapeError` when `q` is not a positive
 * integer.
 *
 * @param x The signal: a single-channel `Signal`, or bare samples at rate 1.
 * @param q The downsampling factor (a positive integer).
 * @param options The filter (`ftype` and its order `n`) and whether it is applied without phase shift (`zeroPhase`).
 * @returns The decimated `Signal` at rate $f_s/q$, with the input's $t_0$ (and its unit, except for a zero-phase FIR,
 *   whose result has none).
 *
 * @example Decimate a slow sine by 2
 * // Three periods of 16 samples, decimated to 8 per period: the kept samples stay on the sine.
 * const x = Array.from({ length: 48 }, (_, k) => Math.sin((2 * Math.PI * k) / 16))
 * const y = decimateSignal(x, 2)
 * print('fs =', y.fs, ', samples =', y.data.shape[0])
 * print('y[8..11] =', slice(y.data, [8, 12]))
 * print('sine at t = 16, 18, 20, 22 =', [8, 9, 10, 11].map((m) => Math.sin((2 * Math.PI * m) / 8)))
 *
 * @example The anti-aliasing filter removes what would alias
 * // A tone at 0.4 of the sample rate is above the new Nyquist (0.25): plain thinning keeps it, decimation does not.
 * const x = Array.from({ length: 64 }, (_, k) => Math.cos(2 * Math.PI * 0.4 * k))
 * const thinned = x.filter((_, k) => k % 2 === 0)
 * const y = decimateSignal(x, 2)
 * print('thinned, largest |value| =', Math.max(...thinned.map(Math.abs)))
 * print('decimated, largest |value| in the middle =', max(abs(slice(y.data, [8, 24]))))
 */
export function decimateSignal(x: SignalInput, q: Size, options: DecimateOptions = {}): Signal {
  const where = 'decimateSignal'
  positiveInteger(q, 'q', where)
  const s = readSamples(x, where)
  const ftype = options.ftype ?? 'iir'
  const zeroPhase = options.zeroPhase ?? true
  const values = tensor(s.values)
  if (ftype === 'fir') {
    const order = options.n ?? 20 * q
    const b = firwin(order + 1, 1 / q, { window: 'hamming' }).repr.b as Tensor
    if (zeroPhase) return resamplePoly(signal(values, { fs: s.fs, t0: s.t0 }), 1, q, { window: toFlat(b) })
    const nOut = Math.ceil(s.values.length / q)
    const y = slice(upfirdn(b, values, { down: q }), [0, nOut]) as Tensor
    return resampled(y, s.fs / q, s.t0, s.unit)
  }
  const sos = cheby1(options.n ?? 8, 0.05, 0.8 / q, { output: 'sos' })
  const filtered = zeroPhase ? filtfilt(sos, values) : sosfilt(sos, values).y
  return resampled(slice(filtered, [null, null, q]), s.fs / q, s.t0, s.unit)
}

/**
 * The polyphase components of a filter $h$ for $M$ branches (type 1): $E_k[p] = h[pM + k]$, $k = 0, \dots, M - 1$, so
 * that $H(z) = \sum_k z^{-k} E_k(z^M)$. Returns $M \times \lceil L/M \rceil$ for $L$ taps, zero-padded at the end
 * (Vaidyanathan, 1993, §4.3). Throws `ShapeError` when `branches` is not a positive integer.
 *
 * @param h The filter's taps $h[0], \dots, h[L - 1]$.
 * @param branches The number of branches $M$ (a positive integer).
 * @returns The components as rows: row $k$ is $E_k$, the taps $h[k], h[k + M], h[k + 2M], \dots$
 *
 * @example Three branches of a seven-tap filter
 * print('E =', polyphase([0, 1, 2, 3, 4, 5, 6], 3))
 */
export function polyphase(h: VectorLike, branches: Size): Tensor {
  positiveInteger(branches, 'branches', 'polyphase')
  const taps = readSamples(h, 'polyphase').values
  const P = Math.ceil(taps.length / branches)
  const padded = new Float64Array(P * branches)
  padded.set(taps)
  return transpose(reshape(tensor(padded), [P, branches])) as Tensor
}

/**
 * The uniform DFT analysis filter bank with $M$ channels decimated by $M$ (critically sampled): channel $k$ filters
 * $x$ with the modulated prototype $h_k[n] = h[n] e^{i 2\pi k n/M}$ and keeps every $M$-th sample,
 * $y_k[m] = (x * h_k)[mM]$. Computed by the polyphase identity $y_k[m] = \sum_r e^{i 2\pi k r/M} u_r[m]$, with
 * $u_r = E_r * x_r$ and $x_r[m] = x[mM - r]$: one grouped convolution of the $M$ polyphase branches and an inverse DFT
 * across them per output sample. Returns complex128 $M \times \lceil (N + L - 1)/M \rceil$ for $N$ samples and $L$
 * taps, so the full convolution is covered (Crochiere and Rabiner, 1983, ch. 7; Vaidyanathan, 1993, §4.5). The sample
 * rate of a `Signal` input is not used. Throws `ShapeError` when `channels` is not a positive integer.
 *
 * @param x The signal: a single-channel `Signal`, or bare samples.
 * @param prototype The prototype lowpass $h$, as taps.
 * @param channels The number of channels $M$, which is also the decimation factor (a positive integer).
 * @returns The channel outputs as rows: entry $(k, m)$ is $y_k[m]$, complex.
 *
 * @example Two channels with a two-tap prototype
 * // h = [1, 1]: channel 0 filters by [1, 1] (sums), channel 1 by [1, -1] (differences); every second sample is kept.
 * print('y =', dftFilterBank([1, 2, 3, 4], [1, 1], 2))
 */
export function dftFilterBank(x: SignalInput, prototype: VectorLike, channels: Size): Tensor {
  const where = 'dftFilterBank'
  positiveInteger(channels, 'channels', where)
  const M = channels
  const xs = readSamples(x, where).values
  const L = readSamples(prototype, where).values.length
  const frames = Math.ceil((xs.length + L - 1) / M)
  const E = polyphase(prototype, M)
  const P = E.shape[1]
  // Branch r reads xᵣ[m] = x[mM − r] for m = 0 … frames − 1 (zero outside x); its convolution with Eᵣ is truncated.
  const branches = new Float64Array(M * frames)
  for (let r = 0; r < M; r++)
    for (let m = 0; m < frames; m++) {
      const i = m * M - r
      branches[r * frames + m] = i >= 0 && i < xs.length ? xs[i] : 0
    }
  const input = reshape(tensor(branches), [1, M, frames])
  const kernels = reshape(E, [M, 1, P])
  const u = slice(conv(input, kernels, { groups: M, padding: [[P - 1, 0]] }), 0) as Tensor // [M, frames], causal
  // y[k, m] = Σᵣ uᵣ[m] e^{i2πkr/M} = M · ifft over r.
  const y = mul(M, ifft(transpose(u), { axis: -1 })) as Tensor // [frames, M]
  return transpose(y) as Tensor
}

/**
 * Whittaker–Shannon reconstruction $x(t) = \sum_n x[n] \operatorname{sinc}\big(f_s (t - t_0) - n\big)$, with
 * $\operatorname{sinc} u = \sin(\pi u)/(\pi u)$, at arbitrary times (seconds), the ideal low-pass interpolation of the
 * sampling theorem (Shannon, 1949). Exact for a signal band-limited below $f_s/2$ sampled forever; a finite record
 * truncates the sum, so the error grows towards its ends. The cost is the number of samples times the number of times.
 *
 * @param x The signal: a single-channel `Signal` (its $f_s$ and $t_0$ place the samples in time), or bare samples at
 *   $f_s = 1$ from $t_0 = 0$ (times are then in samples).
 * @param times The times $t$ at which to evaluate the reconstruction, in seconds.
 * @returns $x(t)$ at each time, a float64 tensor of their length.
 *
 * @example Samples are reproduced, and points between them interpolated
 * // Eight samples of sin(pi t / 2): the truncated sum falls short of the sine between samples.
 * const x = [0, 1, 0, -1, 0, 1, 0, -1]
 * print('at the samples =', sincInterpolate(x, [1, 2, 3]))
 * print('half-way =', sincInterpolate(x, [3.5, 4.5]))
 * print('the sine there =', [3.5, 4.5].map((t) => Math.sin((Math.PI * t) / 2)))
 */
export function sincInterpolate(x: SignalInput, times: VectorLike): Tensor {
  const s = readSamples(x, 'sincInterpolate')
  const t = dense.toF64(times, 'sincInterpolate')
  const out = new Float64Array(t.length)
  for (let i = 0; i < t.length; i++) {
    const u = (t[i] - s.t0) * s.fs
    let acc = 0
    for (let n = 0; n < s.values.length; n++) {
      const d = u - n
      acc += s.values[n] * (d === 0 ? 1 : Math.sin(Math.PI * d) / (Math.PI * d))
    }
    out[i] = acc
  }
  return fromData(out, [out.length])
}
