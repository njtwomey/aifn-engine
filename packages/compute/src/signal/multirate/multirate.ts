/**
 * Multirate signal processing, as scipy.signal: rational resampling by a polyphase FIR (`resamplePoly`), decimation with
 * an anti-aliasing filter (`decimateSignal`, scipy's `decimate`), the polyphase decomposition of a filter
 * (`polyphase`), and the uniform DFT analysis filter bank computed through it (`dftFilterBank`). Everything is a
 * composition over `aifn-compute/foundation/convolution`'s `upfirdn` and `conv`, `aifn-compute/foundation/fourier`'s `ifft` and the
 * filter design and filtering of `aifn-compute/signal/filters` (Crochiere and Rabiner, 1981, "Interpolation and decimation of
 * digital signals: a tutorial review", Proc. IEEE 69(3); Vaidyanathan, 1993, "Multirate Systems and Filter Banks").
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

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))

function positiveInteger(v: number, what: string, where: string) {
  if (!(Number.isInteger(v) && v >= 1)) throw new ShapeError(where, `${where}: ${what} must be a positive integer`)
}

/** The output of a resampler: a `Signal` at the new rate (t0 kept), from the input's samples and rate. */
const resampled = (values: Value, fs: Scalar, t0: Scalar, unit: string | undefined): Signal =>
  signal(values as Tensor, { fs, t0, ...(unit === undefined ? {} : { unit }) })

/** Options for `resamplePoly`. */
export type ResamplePolyOptions = {
  /**
   * The anti-aliasing FIR: a window for `firwin` (default Kaiser with β = 5, scipy's default) applied to a lowpass at
   * 1/max(up, down) of Nyquist with 2·10·max(up, down) + 1 taps; or the filter's taps themselves.
   */
  window?: WindowSpec | VectorLike
  /**
   * How the signal is extended beyond its ends: `constant` (zeros, default) or a statistic of the signal (`mean`,
   * `median`, `minimum`, `maximum`), which is subtracted before filtering and added back.
   */
  padtype?: 'constant' | 'mean' | 'median' | 'minimum' | 'maximum'
}

const isWindowSpec = (w: unknown): w is WindowSpec =>
  typeof w === 'string' || (typeof w === 'object' && w !== null && 'name' in w)

/**
 * Resample by the rational factor up/down with a polyphase FIR, as `scipy.signal.resample_poly`: reduce up/down by
 * their gcd, design (or take) a lowpass h scaled by `up`, pad it so the output is centred on the input, and keep the
 * ⌈n·up/down⌉ samples of `upfirdn(h, x, up, down)` that align with the input's span. The input is a single-channel
 * `SignalInput`; the result is a `Signal` at fs·up/down.
 *
 * @example resamplePoly(signal(x, { fs: 48000 }), 147, 160) // 48 kHz to 44.1 kHz
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
  /** The filter order: default 8 for `iir` (Chebyshev I), 20·q for `fir` (Hamming window). */
  n?: Size
  /** `iir` (default): Chebyshev type I, 0.05 dB ripple, edge 0.8/q of Nyquist; `fir`: a windowed-sinc lowpass at 1/q. */
  ftype?: 'iir' | 'fir'
  /**
   * Filter forwards and backwards (default true): no phase shift. The FIR case then resamples by 1/q with the filter
   * (`resamplePoly`), the IIR case uses `filtfilt` on second-order sections.
   */
  zeroPhase?: boolean
}

/**
 * Downsample by the integer factor q after an anti-aliasing lowpass, as `scipy.signal.decimate` (named apart from
 * `aifn-compute/foundation/trace`'s `decimate`, which thins a trace). The result is a `Signal` at fs/q.
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
 * The polyphase components of a filter h for M branches (type 1): Eₖ[p] = h[pM + k], k = 0 … M − 1, so that
 * H(z) = Σₖ z⁻ᵏ Eₖ(z^M). Returns [M, ⌈|h|/M⌉], zero-padded (Vaidyanathan, 1993, §4.3).
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
 * The uniform DFT analysis filter bank with M channels decimated by M (critically sampled): channel k filters x with
 * the modulated prototype hₖ[n] = h[n]·e^{i2πkn/M} and keeps every M-th sample, yₖ[m] = (x ∗ hₖ)[mM]. Computed by the
 * polyphase identity yₖ[m] = Σᵣ e^{i2πkr/M} uᵣ[m], with uᵣ = Eᵣ ∗ xᵣ and xᵣ[m] = x[mM − r]: one grouped convolution
 * of the M polyphase branches and an inverse DFT across them per output sample. Returns complex128 [M, ⌈(|x| + |h| −
 * 1)/M⌉] (Crochiere and Rabiner, 1983, ch. 7; Vaidyanathan, 1993, §4.5).
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
 * Whittaker–Shannon reconstruction x(t) = Σₙ x[n] sinc(fs (t − t₀) − n) at arbitrary times (seconds), the ideal
 * low-pass interpolation of the sampling theorem (Shannon, 1949). Exact for a signal band-limited below fs/2 sampled
 * forever; a finite record truncates the sum, so the error grows towards its ends.
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
