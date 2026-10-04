/**
 * Digital filters with scipy.signal's conventions: window-method FIR design (`firwin`, Kaiser's formulas), IIR design
 * from analog prototypes by the bilinear transform (`iirfilter`, `butter`, `cheby1`, `cheby2`; Oppenheim and Schafer,
 * 2010, "Discrete-Time Signal Processing", §7.1–7.3), filtering (`lfilter`, `sosfilt`), zero-phase filtering
 * (`filtfilt`, Gustafsson's initial conditions as in scipy), and frequency and group-delay responses.
 *
 * Designs return an `LtiSystem` (discrete, dt = 1/fs; dt = 1 without `fs`), and the filtering and response
 * functions take one. Frequencies: without `fs`, cutoffs are fractions of the Nyquist frequency in (0, 1), as in
 * scipy; with `fs`, they are in the same units as `fs`.
 *
 * Filtering is a composition over `aifn-compute/foundation/convolution`'s `linearFilter` primitive: `lfilter`, `sosfilt`,
 * `lfilterZi`, `sosfiltZi` and `filtfilt` accept coefficients `{ b, a }` (or sections) and samples as traced values,
 * so they are differentiable in the IIR coefficients, the initial state and the signal, and batch along other axes.
 */

import { solve } from 'aifn-compute/numerics/linalg'
import { polynomialRoots } from 'aifn-compute/numerics/polynomial'
import { ellipf, ellipj, ellipk, ellipkm1 } from 'aifn-compute/numerics/special'
import {
  add,
  complex,
  concat,
  cos,
  dense,
  div,
  expj,
  eye,
  fromData,
  full,
  imagPart,
  isTensor,
  isTraced,
  mul,
  neg,
  ones,
  outer,
  realPart,
  reshape,
  shapeOfValue,
  sin,
  slice,
  sqrt,
  square,
  stack,
  sub,
  sum,
  tensor,
  toComplexFlat,
  zeros,
  type SliceSpec,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type {
  ComplexNumber,
  LtiSystem,
  Scalar,
  Signal,
  Size,
  Spectrum,
  VectorLike,
} from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { linearFilter } from 'aifn-compute/foundation/convolution'
import {
  convert,
  frequencyResponse,
  toTransferFunction,
  transferFunction,
  zerosPolesGain,
  type LtiOf,
  type TransferFunctionForm,
} from 'aifn-compute/systems'
import { getWindow, type WindowSpec } from 'aifn-compute/signal/windows'
import { isSignal, signal, spectrum, type SignalInput } from '../signal'

export { unwrapPhase } from '../signal'

// ── FIR design ────────────────────────────────────────────────────────────────────────────────────────────────────

const sinc = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))

/** Options for `firwin`. */
export interface FirwinOptions {
  /** Window for the ideal impulse response. Default hamming. */
  window?: WindowSpec
  /**
   * The band type, or whether the DC gain is 1: `true`/`lowpass`/`bandstop` pass zero frequency, `false`/`highpass`/
   * `bandpass` do not. Default true.
   */
  passZero?: boolean | 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  /** Scale so the gain is exactly 1 at the centre of the first passband. Default true. */
  scale?: boolean
  /** Sampling frequency; the system gets dt = 1/fs. Default 2 (cutoffs as fractions of Nyquist, dt = 1). */
  fs?: Scalar
}

/**
 * A linear-phase FIR filter by the window method, as `scipy.signal.firwin`: the ideal band-pass impulse response
 * Σ (f_hi sinc(f_hi m) − f_lo sinc(f_lo m)) over the passbands, times a window. `cutoff` is one edge or a list of band
 * edges. An odd number of taps is required when the filter passes the Nyquist frequency. Returns the FIR system
 * b(z⁻¹)/1: its taps are `sys.repr.b`.
 */
export function firwin(
  numtaps: Size,
  cutoff: Scalar | readonly Scalar[],
  options: FirwinOptions = {},
): LtiOf<TransferFunctionForm> {
  const { window = 'hamming', scale = true } = options
  const nyq = (options.fs ?? 2) / 2
  const edges = (typeof cutoff === 'number' ? [cutoff] : [...cutoff]).map((f) => f / nyq)
  if (edges.some((f) => !(f > 0 && f < 1)))
    throw new DomainError('firwin', 'firwin: cutoffs must lie strictly between 0 and Nyquist')
  const pz = options.passZero ?? true
  const passZero = pz === true || pz === 'lowpass' || pz === 'bandstop'
  const passNyquist = (edges.length % 2 === 1) !== passZero
  if (passNyquist && numtaps % 2 === 0)
    throw new DomainError('firwin', 'firwin: a filter that passes the Nyquist frequency needs an odd number of taps')
  const bands = [...(passZero ? [0] : []), ...edges, ...(passNyquist ? [1] : [])]
  const alpha = (numtaps - 1) / 2
  const h = new Float64Array(numtaps)
  for (let n = 0; n < numtaps; n++) {
    const m = n - alpha
    for (let b = 0; b < bands.length; b += 2)
      h[n] += bands[b + 1] * sinc(bands[b + 1] * m) - bands[b] * sinc(bands[b] * m)
  }
  const w = getWindow(window, numtaps).data
  for (let n = 0; n < numtaps; n++) h[n] *= w[n]
  if (scale) {
    const [left, right] = [bands[0], bands[1]]
    const f = left === 0 ? 0 : right === 1 ? 1 : 0.5 * (left + right)
    let s = 0
    for (let n = 0; n < numtaps; n++) s += h[n] * Math.cos(Math.PI * (n - alpha) * f)
    for (let n = 0; n < numtaps; n++) h[n] /= s
  }
  return transferFunction(h, [1], { dt: dtOf(options.fs) })
}

/** The sampling interval of a design: 1/fs, or 1 when the frequencies are fractions of Nyquist. */
const dtOf = (fs: Scalar | undefined) => (fs === undefined ? 1 : 1 / fs)

/** Kaiser's β for a stopband attenuation of A dB, as `scipy.signal.kaiser_beta` (Kaiser, 1974). */
export function kaiserBeta(attenuation: Scalar): Scalar {
  const a = attenuation
  if (a > 50) return 0.1102 * (a - 8.7)
  if (a > 21) return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
  return 0
}

/** Attenuation (dB) of a Kaiser FIR filter of `numtaps` taps and transition width (fraction of Nyquist). */
export function kaiserAttenuation(numtaps: Size, width: Scalar): Scalar {
  return 2.285 * (numtaps - 1) * Math.PI * width + 7.95
}

/**
 * Kaiser's design formulas, as `scipy.signal.kaiserord`: the taps and β for a ripple (attenuation) of A dB and a
 * transition width given as a fraction of Nyquist.
 */
export function kaiserOrder(ripple: Scalar, width: Scalar): { numtaps: Size; beta: Scalar } {
  const a = Math.abs(ripple)
  if (a < 8) throw new DomainError('kaiserOrder', 'kaiserOrder: the attenuation must be at least 8 dB')
  return { numtaps: Math.ceil((a - 7.95) / 2.285 / (Math.PI * width) + 1), beta: kaiserBeta(a) }
}

// ── IIR design ────────────────────────────────────────────────────────────────────────────────────────────────────

/** An analog or digital zeros–poles–gain design: complex128 zeros and poles, a real gain. */
type Proto = { z: Tensor; p: Tensor; k: number }

/** The product of the entries of a complex vector (1 when empty). */
function productOf(v: Tensor): ComplexNumber {
  let re = 1
  let im = 0
  for (const z of toComplexFlat(v)) [re, im] = [re * z.re - im * z.im, re * z.im + im * z.re]
  return { re, im }
}

/** Re(Π(−u) / Π(−v)): the gain factor of the frequency transformations (scipy's `lp2hp_zpk`, `lp2bs_zpk`). */
function gainRatio(u: Tensor, v: Tensor): number {
  const a = productOf(neg(u))
  const b = productOf(neg(v))
  return (a.re * b.re + a.im * b.im) / (b.re * b.re + b.im * b.im)
}

const none = (): Tensor => zeros([0], 'complex128')
const repeat = (value: ComplexNumber, n: Size): Tensor => full([n], value, 'complex128')
/** The angles π(−n + 1 + 2i)/(2n), i = 0 … n − 1, of the prototypes' poles. */
const angles = (n: Size): Tensor => tensor(Array.from({ length: n }, (_, i) => (Math.PI * (-n + 1 + 2 * i)) / (2 * n)))

/** Analog Butterworth prototype (cutoff 1 rad/s), as `scipy.signal.buttap`: p = −e^{iθ}. */
function buttap(n: Size): Proto {
  return { z: none(), p: neg(expj(angles(n))), k: 1 }
}

/** Analog Chebyshev type I prototype with passband ripple rp dB, as `scipy.signal.cheb1ap`: p = −sinh(μ + iθ). */
function cheb1ap(n: Size, rp: number): Proto {
  const eps = Math.sqrt(10 ** (0.1 * rp) - 1)
  const mu = Math.asinh(1 / eps) / n
  const theta = angles(n)
  const p = complex(mul(-Math.sinh(mu), cos(theta)), mul(-Math.cosh(mu), sin(theta))) as Tensor
  let k = productOf(neg(p)).re
  if (n % 2 === 0) k /= Math.sqrt(1 + eps * eps)
  return { z: none(), p, k }
}

/** Analog Chebyshev type II prototype with stopband attenuation rs dB, as `scipy.signal.cheb2ap`. */
function cheb2ap(n: Size, rs: number): Proto {
  const de = 1 / Math.sqrt(10 ** (0.1 * rs) - 1)
  const mu = Math.asinh(1 / de) / n
  const ms: number[] = []
  for (let m = -n + 1; m < n; m += 2) if (n % 2 === 0 || m !== 0) ms.push(m)
  // z = −conj(i / sin(mπ/2n)) = i / sin(mπ/2n).
  const z = complex(zeros([ms.length]), tensor(ms.map((m) => 1 / Math.sin((m * Math.PI) / (2 * n))))) as Tensor
  const q = neg(expj(angles(n)))
  const p = div(1, complex(mul(Math.sinh(mu), realPart(q)), mul(Math.cosh(mu), imagPart(q)))) as Tensor
  return { z, p, k: gainRatio(p, z) }
}

/** A complex128 vector from (re, im) pairs. */
function complexVector(roots: readonly ComplexNumber[]): Tensor {
  const v = new Float64Array(2 * roots.length)
  roots.forEach((r, i) => {
    v[2 * i] = r.re
    v[2 * i + 1] = r.im
  })
  return fromData(v, [roots.length], 'complex128')
}

/** 10^{x/10} − 1 without cancellation for small x (scipy's `_pow10m1`). */
const pow10m1 = (x: number) => Math.expm1(0.1 * x * Math.LN10)

/** The modulus m of an order-n elliptic filter from m₁ = ε²/(10^{rs/10} − 1), by nomes (scipy's `_ellipdeg`). */
function ellipdeg(n: Size, m1: number): number {
  const q = Math.exp((-Math.PI * ellipkm1(m1)) / ellipk(m1) / n)
  let num = 0
  let den = 1
  for (let k = 0; k <= 7; k++) num += q ** (k * (k + 1))
  for (let k = 1; k <= 8; k++) den += 2 * q ** (k * k)
  return 16 * q * (num / den) ** 4
}

/**
 * Analog elliptic (Cauer) prototype with passband ripple rp dB and stopband attenuation rs dB, as
 * `scipy.signal.ellipap` (Orfanidis, 2006, "Lecture notes on elliptic filter design"): zeros i/(√m sn) and poles from
 * the Jacobi functions at jK/n and at v₀ = K F(arctan(1/ε) | 1 − m₁)/(n K₁).
 */
function ellipap(n: Size, rp: number, rs: number): Proto {
  const epsSq = pow10m1(rp)
  if (n === 1) {
    const p = -Math.sqrt(1 / epsSq)
    return { z: none(), p: complexVector([{ re: p, im: 0 }]), k: -p }
  }
  const m1 = epsSq / pow10m1(rs)
  if (!(m1 > 0)) throw new DomainError('ellip', 'ellip: the stopband attenuation is too large for double precision')
  const m = ellipdeg(n, m1)
  const capk = ellipk(m)
  const zs: ComplexNumber[] = []
  const ps: ComplexNumber[] = []
  // v₀ solves sc(v₀ | 1 − m₁) = 1/ε: the imaginary part of the inverse sn at i/ε (scipy's `_arc_jac_sc1`).
  const r = ellipf(Math.atan(1 / Math.sqrt(epsSq)), 1 - m1, m1)
  const v0 = (capk * r) / (n * ellipk(m1))
  const { sn: sv, cn: cv, dn: dv } = ellipj(v0, 1 - m)
  const half: ComplexNumber[] = []
  for (let j = 1 - (n % 2); j < n; j += 2) {
    const { sn: s, cn: c, dn: d } = ellipj((j * capk) / n, m)
    if (Math.abs(s) > Number.EPSILON) zs.push({ re: 0, im: 1 / (Math.sqrt(m) * s) })
    const den = 1 - (d * sv) ** 2
    half.push({ re: -(c * d * sv * cv) / den, im: -(s * dv) / den })
  }
  const norm = Math.sqrt(half.reduce((t, p) => t + p.re * p.re + p.im * p.im, 0))
  for (const p of half) {
    ps.push(p)
    if (n % 2 === 0 || Math.abs(p.im) > Number.EPSILON * norm) ps.push({ re: p.re, im: -p.im })
  }
  const z = complexVector([...zs, ...zs.map((v) => ({ re: v.re, im: -v.im }))])
  const p = complexVector(ps)
  let k = gainRatio(p, z)
  if (n % 2 === 0) k /= Math.sqrt(1 + epsSq)
  return { z, p, k }
}

/** log n!, summed (n is a filter order). */
const logFactorial = (n: number) => {
  let s = 0
  for (let i = 2; i <= n; i++) s += Math.log(i)
  return s
}

/**
 * Analog Bessel–Thomson prototype, phase-normalised as `scipy.signal.besselap(n, norm='phase')`: the roots of the
 * reverse Bessel polynomial θₙ(s) = Σₖ (2n − k)! / (2^{n−k} k! (n − k)!) sᵏ, scaled by θₙ(0)^{−1/n} so the phase
 * response matches Butterworth's at high frequency; unit DC gain. A maximally flat group delay (Thomson, 1949).
 */
function besselap(n: Size): Proto {
  const logA = (k: number) => logFactorial(2 * n - k) - (n - k) * Math.LN2 - logFactorial(k) - logFactorial(n - k)
  const log0 = logA(0)
  // q(s) = θₙ(c s)/θₙ(0) with c = θₙ(0)^{1/n}: monic with unit constant term, so well scaled. Highest power first.
  const q = Array.from({ length: n + 1 }, (_, i) => {
    const k = n - i
    return Math.exp(logA(k) + (k / n) * log0 - log0)
  })
  const found = polynomialRoots(q).roots
  const flat = toComplexFlat(found)
  // Newton polishing in complex arithmetic: the companion eigenvalues are accurate to a few ulps of the largest root.
  const polished = flat.map((z0) => {
    let { re, im } = z0
    for (let it = 0; it < 8; it++) {
      let pr = 0
      let pi = 0
      let dr = 0
      let di = 0
      for (let i = 0; i <= n; i++) {
        ;[dr, di] = [dr * re - di * im + pr, dr * im + di * re + pi]
        ;[pr, pi] = [pr * re - pi * im + q[i], pr * im + pi * re]
      }
      const d2 = dr * dr + di * di
      if (d2 === 0) break
      const sr = (pr * dr + pi * di) / d2
      const si = (pi * dr - pr * di) / d2
      re -= sr
      im -= si
      if (Math.hypot(sr, si) < 1e-16 * Math.hypot(re, im)) break
    }
    return { re, im }
  })
  return { z: none(), p: complexVector(polished), k: 1 }
}

function lp2lp({ z, p, k }: Proto, wo: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return { z: mul(z, wo), p: mul(p, wo), k: k * wo ** degree }
}

function lp2hp({ z, p, k }: Proto, wo: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return { z: concat([div(wo, z), repeat({ re: 0, im: 0 }, degree)]), p: div(wo, p), k: k * gainRatio(z, p) }
}

/** The two roots r·bw/2 ± √((r·bw/2)² − wo²) of each root r (band-pass), or of bw/2/r (band-stop). */
function splitRoots(roots: Tensor, wo: number): Tensor {
  const root = sqrt(sub(square(roots), wo * wo))
  return concat([add(roots, root), sub(roots, root)])
}

function lp2bp({ z, p, k }: Proto, wo: number, bw: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return {
    z: concat([splitRoots(mul(z, bw / 2), wo), repeat({ re: 0, im: 0 }, degree)]),
    p: splitRoots(mul(p, bw / 2), wo),
    k: k * bw ** degree,
  }
}

function lp2bs({ z, p, k }: Proto, wo: number, bw: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return {
    z: concat([splitRoots(div(bw / 2, z), wo), repeat({ re: 0, im: wo }, degree), repeat({ re: 0, im: -wo }, degree)]),
    p: splitRoots(div(bw / 2, p), wo),
    k: k * gainRatio(z, p),
  }
}

/** The bilinear transform s = 2 fs (z − 1)/(z + 1), as `scipy.signal.bilinear_zpk`. */
function bilinear({ z, p, k }: Proto, fs: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  const fs2 = 2 * fs
  const map = (r: Tensor) => div(add(fs2, r), sub(fs2, r))
  return {
    z: concat([map(z), repeat({ re: -1, im: 0 }, degree)]),
    p: map(p),
    k: k * gainRatio(sub(z, fs2), sub(p, fs2)),
  }
}

/** Options for `iirfilter`. */
export interface IirOptions {
  btype?: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  ftype?: 'butter' | 'cheby1' | 'cheby2' | 'ellip' | 'bessel'
  /** Passband ripple (dB), Chebyshev I and elliptic. Default 1. */
  rp?: number
  /** Stopband attenuation (dB), Chebyshev II and elliptic. Default 40. */
  rs?: number
  /** Sampling frequency; the system gets dt = 1/fs. Default 2 (edges as fractions of Nyquist, dt = 1). */
  fs?: Scalar
  /** The representation of the result: `zpk` (the design's exact form), `tf` or `sos`. Default `zpk`. */
  output?: 'zpk' | 'tf' | 'sos'
}

/**
 * An IIR filter of order n, as `scipy.signal.iirfilter`: an analog prototype, frequency-transformed to the pre-warped
 * edges 4 tan(πWₙ/2) (fs = 2), then mapped by the bilinear transform. `wn` is one edge (low/high-pass) or two
 * (band-pass/stop). For Butterworth the edge is the −3 dB point; Chebyshev I and elliptic, the passband edge (where
 * the gain leaves the ripple band); Chebyshev II, the stopband edge; Bessel, the phase-normalised edge (the phase
 * asymptote of Butterworth's). Returns the discrete system (dt = 1/fs) in the `output` representation.
 */
export function iirfilter(n: Size, wn: Scalar | readonly [Scalar, Scalar], options: IirOptions = {}): LtiSystem {
  const { ftype = 'butter', rp = 1, rs = 40 } = options
  const btype = options.btype ?? (typeof wn === 'number' ? 'lowpass' : 'bandpass')
  const nyq = (options.fs ?? 2) / 2
  const edges = (typeof wn === 'number' ? [wn] : [...wn]).map((f) => f / nyq)
  if (edges.some((f) => !(f > 0 && f < 1)))
    throw new DomainError('iirfilter', 'iirfilter: edges must lie strictly between 0 and Nyquist')
  const two = btype === 'bandpass' || btype === 'bandstop'
  if (two !== (edges.length === 2))
    throw new DomainError('iirfilter', `iirfilter: ${btype} needs ${two ? 'two edges' : 'one edge'}`)
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('iirfilter', 'iirfilter: the order must be a positive integer')
  const proto0: Record<NonNullable<IirOptions['ftype']>, () => Proto> = {
    butter: () => buttap(n),
    cheby1: () => cheb1ap(n, rp),
    cheby2: () => cheb2ap(n, rs),
    ellip: () => ellipap(n, rp, rs),
    bessel: () => besselap(n),
  }
  let proto = proto0[ftype]()
  const warped = edges.map((f) => 4 * Math.tan((Math.PI * f) / 2))
  if (btype === 'lowpass') proto = lp2lp(proto, warped[0])
  else if (btype === 'highpass') proto = lp2hp(proto, warped[0])
  else {
    const bw = warped[1] - warped[0]
    const wo = Math.sqrt(warped[0] * warped[1])
    proto = btype === 'bandpass' ? lp2bp(proto, wo, bw) : lp2bs(proto, wo, bw)
  }
  const d = bilinear(proto, 2)
  const sys = zerosPolesGain(d.z, d.p, d.k, { dt: dtOf(options.fs) })
  return convert(sys, options.output ?? 'zpk')
}

/** A Butterworth filter (maximally flat passband), as `scipy.signal.butter`. */
export function butter(
  n: Size,
  wn: Scalar | readonly [Scalar, Scalar],
  options: Omit<IirOptions, 'ftype'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'butter' })
}

/** A Chebyshev type I filter (equiripple passband of `rp` dB), as `scipy.signal.cheby1`. */
export function cheby1(
  n: number,
  rp: Scalar,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rp'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'cheby1', rp })
}

/** A Chebyshev type II filter (equiripple stopband `rs` dB down), as `scipy.signal.cheby2`. */
export function cheby2(
  n: number,
  rs: Scalar,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rs'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'cheby2', rs })
}

/** An elliptic (Cauer) filter: equiripple in both bands, the steepest transition for its order, as `scipy.signal.ellip`. */
export function ellip(
  n: number,
  rp: Scalar,
  rs: Scalar,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rp' | 'rs'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'ellip', rp, rs })
}

/**
 * A Bessel–Thomson filter (maximally flat group delay, phase-normalised), as `scipy.signal.bessel(norm='phase')`. The
 * bilinear transform keeps the magnitude shape but not the flat delay exactly.
 */
export function bessel(
  n: Size,
  wn: Scalar | readonly [Scalar, Scalar],
  options: Omit<IirOptions, 'ftype'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'bessel' })
}

// ── Filtering ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Filter coefficients as in scipy's `lfilter(b, a, x)`: b and a in ascending powers of z⁻¹ (numbers or traced). */
export type FilterCoefficients = { b: Value | VectorLike; a: Value | VectorLike }

/** A filter: a discrete `LtiSystem` (any representation), or coefficients `{ b, a }`. */
export type FilterSpec = LtiSystem | FilterCoefficients

/** Options for `lfilter` and `sosfilt`. */
export type FilterOptions = {
  /**
   * The initial state (transposed direct form II, coefficients normalised by a₀): x's shape with max(|a|, |b|) − 1
   * along the time axis (a vector for a vector x). For `sosfilt`, [sections, …] with 2 along the time axis. Default
   * zeros (at rest).
   */
  zi?: Value | VectorLike
  /** The time axis of the samples (default −1). */
  axis?: number
}

/** A filtered signal and the final state (the `zi` that continues it). */
export type Filtered<Y, Z> = { y: Y; zf: Z }

const isSystem = (f: unknown): f is LtiSystem => (f as { kind?: unknown }).kind === 'lti'

/** A number, array, tensor or traced value as a value (plain arrays become float64 tensors). */
function asValue(v: Value | VectorLike): Value {
  if (typeof v === 'number') return tensor([v])
  return isTraced(v) || isTensor(v) ? (v as Value) : tensor(Array.from(v as ArrayLike<number>))
}

/** A slice spec selecting `range` along `axis` of a rank-`rank` value. */
const along = (rank: number, axis: number, range: SliceSpec): SliceSpec[] =>
  Array.from({ length: rank }, (_, k) => (k === axis ? range : null))

const reverse = (x: Value, axis: number): Value => slice(x, ...along(shapeOfValue(x).length, axis, [null, null, -1]))

/** A vector reshaped to lie along `axis` of a rank-`rank` value ([1, …, n, …, 1]). */
const alongAxis = (v: Value, rank: number, axis: number): Value =>
  reshape(
    v,
    Array.from({ length: rank }, (_, k) => (k === axis ? shapeOfValue(v)[0] : 1)),
  )

/** The time axis of a rank-`rank` value, from a possibly negative `axis`. */
function timeAxis(axis: number | undefined, rank: number, where: string): number {
  const a = (axis ?? -1) < 0 ? rank + (axis ?? -1) : axis!
  if (rank === 0 || a < 0 || a >= rank) throw new ShapeError(where, `${where}: axis ${axis ?? -1} out of range`)
  return a
}

/** Samples from a `Signal` (kept for the output's metadata), a value or an array. */
function samples(x: SignalInput | Value): { data: Value; meta?: Signal } {
  if (isSignal(x)) return { data: x.data, meta: x }
  return { data: asValue(x as Value | VectorLike) }
}

/** The output as a `Signal` like the input's when the input was one and the output is concrete, else as a value. */
function wrap(y: Value, meta: Signal | undefined): Value | Signal {
  if (!meta || !isTensor(y)) return y
  return signal(y, {
    fs: meta.fs,
    t0: meta.t0,
    ...(meta.unit !== undefined ? { unit: meta.unit } : {}),
    ...(meta.channels !== undefined ? { channels: meta.channels } : {}),
  })
}

/** Transfer-function coefficients normalised by a₀ and padded to a common length K + 1. */
function normalised(f: FilterSpec, where: string): { b: Value; a: Value; K: Size } {
  let b: Value
  let a: Value
  if (isSystem(f)) {
    if (f.domain !== 'discrete') throw new DomainError(where, `${where}: the system must be discrete`)
    const tf = toTransferFunction(f).repr
    b = tf.b
    a = tf.a
  } else {
    b = asValue(f.b)
    a = asValue(f.a)
  }
  const nb = shapeOfValue(b)[0]
  const na = shapeOfValue(a)[0]
  if (!(na > 0 && nb > 0)) throw new ShapeError(where, `${where}: coefficients must not be empty`)
  if (isTensor(a) && dense.data(slice(a, [0, 1]) as Tensor)[0] === 0)
    throw new DomainError(where, `${where}: a[0] must be non-zero`)
  const K = Math.max(nb, na) - 1
  const pad = (v: Value, n: number) => (n < K + 1 ? concat([v, zeros([K + 1 - n])]) : v)
  const a0 = slice(a, 0)
  return { b: div(pad(b, nb), a0), a: div(pad(a, na), a0), K }
}

/**
 * The final transposed-direct-form-II state after filtering x (N samples along `axis`) into y with normalised b, a
 * (length K + 1): zfᵢ = Σₗ (b[i+1+l] x[N−1−l] − a[i+1+l] y[N−1−l]) + zi[i + N] (the last term while i + N < K).
 */
function finalState(b: Value, a: Value, K: Size, x: Value, y: Value, axis: number, zi: Value | null): Value {
  const shape = shapeOfValue(x)
  const rank = shape.length
  const N = shape[axis]
  if (K === 0) return zeros(shape.map((d, k) => (k === axis ? 0 : d)))
  const parts: Value[] = []
  for (let i = 0; i < K; i++) {
    const L = Math.min(K - i, N)
    let term: Value =
      L > 0
        ? sub(
            sum(
              mul(
                alongAxis(slice(b, [i + 1, i + 1 + L]), rank, axis),
                reverse(slice(x, ...along(rank, axis, [N - L, N])), axis),
              ),
              axis,
              true,
            ),
            sum(
              mul(
                alongAxis(slice(a, [i + 1, i + 1 + L]), rank, axis),
                reverse(slice(y, ...along(rank, axis, [N - L, N])), axis),
              ),
              axis,
              true,
            ),
          )
        : zeros(shape.map((d, k) => (k === axis ? 1 : d)))
    if (zi !== null && i + N < K) term = add(term, slice(zi, ...along(rank, axis, [i + N, i + N + 1])))
    parts.push(term)
  }
  return concat(parts, axis)
}

/** One transfer-function filter pass on a value: y and the final state. */
function filterValue(f: FilterSpec, x: Value, axis: number, zi: Value | null, where: string): Filtered<Value, Value> {
  const { b, a, K } = normalised(f, where)
  if (zi !== null) {
    const want = shapeOfValue(x).map((d, k) => (k === axis ? K : d))
    const got = shapeOfValue(zi)
    if (got.length !== want.length || got.some((d, k) => d !== want[k]))
      throw new ShapeError(where, `${where}: zi must have shape [${want.join(', ')}], got [${got.join(', ')}]`)
  }
  const y = linearFilter(b, a, x, zi === null || K === 0 ? { axis } : { axis, zi })
  return { y, zf: finalState(b, a, K, x, y, axis, zi) }
}

/** The initial state for a vector x as the rank of x requires (a [K] vector zi is accepted for any rank-1 x). */
const readState = (zi: Value | VectorLike | undefined): Value | null => (zi === undefined ? null : asValue(zi))

/**
 * Filters samples through a discrete system or coefficients `{ b, a }` by the difference equation
 * Σₖ aₖ y[t−k] = Σₖ bₖ x[t−k] along `axis` (every other axis a separate signal), as `scipy.signal.lfilter`. A
 * second-order-sections system runs as `sosfilt`. `zi` is the initial state (default zeros), and `zf` the final one.
 * A `Signal` input gives a `Signal` output with the same sample rate and start time; a value input gives a value, and
 * the whole computation is differentiable in b, a, zi and x.
 *
 * @example lfilter({ b: [1], a: [1, -0.9] }, [1, 0, 0]).y // 1, 0.9, 0.81
 */
export function lfilter(f: FilterSpec, x: Signal, options?: FilterOptions): Filtered<Signal, Tensor>
export function lfilter(f: FilterSpec, x: Value | VectorLike, options?: FilterOptions): Filtered<Value, Value>
export function lfilter(
  f: FilterSpec,
  x: SignalInput | Value,
  options: FilterOptions = {},
): Filtered<Value | Signal, Value> {
  if (isSystem(f) && f.repr.form === 'sos') return sosfilt(f, x as Value, options)
  const { data, meta } = samples(x)
  const axis = timeAxis(options.axis, shapeOfValue(data).length, 'lfilter')
  const { y, zf } = filterValue(f, data, axis, readState(options.zi), 'lfilter')
  return { y: wrap(y, meta), zf }
}

/** Second-order sections [S, 6] (b₀ b₁ b₂ a₀ a₁ a₂ rows) from an sos system, a matrix or a traced value. */
function sectionsOf(sos: LtiSystem | Value | readonly (readonly number[])[], where: string): Value {
  if (isSystem(sos)) {
    if (sos.repr.form !== 'sos') throw new DomainError(where, `${where}: the system is not in second-order sections`)
    return sos.repr.sections
  }
  const v: Value = isTraced(sos) || isTensor(sos) ? (sos as Value) : tensor(sos as number[][])
  const s = shapeOfValue(v)
  if (s.length !== 2 || s[1] !== 6) throw new ShapeError(where, `${where}: sections must be [S, 6]`)
  return v
}

/**
 * Filters samples through a cascade of second-order sections [S, 6] (rows b₀ b₁ b₂ a₀ a₁ a₂), as
 * `scipy.signal.sosfilt`: better conditioned than one high-order difference equation. `zi` is [S, …] with 2 along
 * the time axis; `zf` has the same shape. Differentiable in the sections, zi and x.
 */
export function sosfilt(
  sos: LtiSystem | Value | readonly (readonly number[])[],
  x: Signal,
  options?: FilterOptions,
): Filtered<Signal, Tensor>
export function sosfilt(
  sos: LtiSystem | Value | readonly (readonly number[])[],
  x: Value | VectorLike,
  options?: FilterOptions,
): Filtered<Value, Value>
export function sosfilt(
  sos: LtiSystem | Value | readonly (readonly number[])[],
  x: SignalInput | Value,
  options: FilterOptions = {},
): Filtered<Value | Signal, Value> {
  const S = sectionsOf(sos, 'sosfilt')
  const { data, meta } = samples(x)
  const axis = timeAxis(options.axis, shapeOfValue(data).length, 'sosfilt')
  const zi = readState(options.zi)
  const count = shapeOfValue(S)[0]
  let v = data
  const zfs: Value[] = []
  for (let s = 0; s < count; s++) {
    const row = slice(S, s)
    const out = filterValue(
      { b: slice(row, [0, 3]), a: slice(row, [3, 6]) },
      v,
      axis,
      zi === null ? null : slice(zi, s),
      'sosfilt',
    )
    zfs.push(out.zf)
    v = out.y
  }
  return { y: wrap(v, meta), zf: count ? stack(zfs, 0) : zeros([0]) }
}

/**
 * The initial state of `lfilter` for a step response in steady state, as `scipy.signal.lfilter_zi`: solves
 * (I − Cᵀ) zi = b[1:] − a[1:] b₀ with C the companion matrix of a (coefficients normalised by a₀). Multiply by x[0]
 * to start a signal without a transient. A second-order-sections system gives `sosfiltZi`. Differentiable in b and a.
 */
export function lfilterZi(f: FilterSpec): Value {
  if (isSystem(f) && f.repr.form === 'sos') return sosfiltZi(f)
  const { b, a, K } = normalised(f, 'lfilterZi')
  if (K === 0) return zeros([0])
  const aTail = slice(a, [1, K + 1])
  const M = sub(add(eye(K), outer(aTail, tensor(Array.from({ length: K }, (_, j) => (j === 0 ? 1 : 0))))), eye(K, K, 1))
  const rhs = sub(slice(b, [1, K + 1]), mul(aTail, slice(b, 0)))
  return solve(M, rhs)
}

/**
 * Steady-state initial states [S, 2] of a section cascade for a unit step, as `scipy.signal.sosfilt_zi`: each
 * section's `lfilterZi` scaled by the DC gain of the sections before it.
 */
export function sosfiltZi(sos: LtiSystem | Value | readonly (readonly number[])[]): Value {
  const S = sectionsOf(sos, 'sosfiltZi')
  const count = shapeOfValue(S)[0]
  let scale: Value = 1
  const rows: Value[] = []
  for (let s = 0; s < count; s++) {
    const row = slice(S, s)
    const b = slice(row, [0, 3])
    const a = slice(row, [3, 6])
    rows.push(mul(scale, lfilterZi({ b, a })))
    scale = mul(scale, div(sum(b), sum(a)))
  }
  return count ? stack(rows, 0) : zeros([0, 2])
}

/** Options for `filtfilt`. */
export type FiltfiltOptions = {
  /** How to extend the signal at each end: odd reflection (default), even reflection, the edge value, or none. */
  padtype?: 'odd' | 'even' | 'constant' | 'none'
  /** Samples added at each end. Default 3·max(|a|, |b|), or scipy's count for sections. */
  padlen?: Size
  /** The time axis (default −1). */
  axis?: number
}

/**
 * Zero-phase filtering, as `scipy.signal.filtfilt` and `sosfiltfilt` (Gustafsson, 1996, IEEE Trans. Signal Process.
 * 44(4)): extend the signal by `padlen` samples at each end, filter forwards and backwards with steady-state initial
 * conditions scaled by the first sample of each pass, and trim. The result has no phase shift and the squared
 * magnitude response. A composition, so differentiable in the coefficients (or sections) and the signal.
 */
export function filtfilt(f: FilterSpec, x: Signal, options?: FiltfiltOptions): Signal
export function filtfilt(f: FilterSpec, x: Value | VectorLike, options?: FiltfiltOptions): Value
export function filtfilt(f: FilterSpec, x: SignalInput | Value, options: FiltfiltOptions = {}): Value | Signal {
  const { data, meta } = samples(x)
  const shape = shapeOfValue(data)
  const rank = shape.length
  const axis = timeAxis(options.axis, rank, 'filtfilt')
  const n = shape[axis]
  const sos = isSystem(f) && f.repr.form === 'sos'
  let padlen: Size
  if (sos) {
    const S = sectionsOf(f as LtiSystem, 'filtfilt')
    const count = shapeOfValue(S)[0]
    const d = dense.data(S as Tensor)
    let zb = 0
    let za = 0
    for (let s = 0; s < count; s++) {
      if (d[6 * s + 2] === 0) zb++
      if (d[6 * s + 5] === 0) za++
    }
    padlen = 3 * (2 * count + 1 - Math.min(zb, za))
  } else padlen = 3 * (normalised(f, 'filtfilt').K + 1)
  const padtype = options.padtype ?? 'odd'
  const edge = padtype === 'none' ? 0 : (options.padlen ?? padlen)
  if (edge >= n) throw new DomainError('filtfilt', `filtfilt: the signal must be longer than padlen = ${edge}`)
  const at = (k: number) => slice(data, ...along(rank, axis, [k, k + 1]))
  const x0 = at(0)
  const xn = at(n - 1)
  let ext = data
  if (edge > 0) {
    const left = reverse(slice(data, ...along(rank, axis, [1, edge + 1])), axis)
    const right = reverse(slice(data, ...along(rank, axis, [n - 1 - edge, n - 1])), axis)
    const block = shape.map((d, k) => (k === axis ? edge : d))
    const [l, r] =
      padtype === 'odd'
        ? [sub(mul(2, x0), left), sub(mul(2, xn), right)]
        : padtype === 'even'
          ? [left, right]
          : [mul(x0, ones(block)), mul(xn, ones(block))]
    ext = concat([l, data, r], axis)
  }
  const zi = lfilterZi(f)
  // The steady state for a step of height v: zi along the time axis times v (per section for a cascade).
  const scaled = (v: Value): Value =>
    sos
      ? stack(
          Array.from({ length: shapeOfValue(zi)[0] }, (_, s) => mul(alongAxis(slice(zi, s), rank, axis), v)),
          0,
        )
      : mul(alongAxis(zi, rank, axis), v)
  const pass = (v: Value): Value => {
    const first = slice(v, ...along(rank, axis, [0, 1]))
    return sos
      ? sosfilt(f as LtiSystem, v, { axis, zi: scaled(first) }).y
      : lfilter(f, v, { axis, zi: scaled(first) }).y
  }
  const forward = pass(ext)
  const backward = reverse(pass(reverse(forward, axis)), axis)
  return wrap(slice(backward, ...along(rank, axis, [edge, edge + n])), meta)
}

// ── Responses ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Σₖ cₖ e^{−iωk} for real coefficients c, as (re, im). */
function evaluate(coef: Float64Array, w: number): ComplexNumber {
  let re = 0
  let im = 0
  for (let k = 0; k < coef.length; k++) {
    re += coef[k] * Math.cos(w * k)
    im -= coef[k] * Math.sin(w * k)
  }
  return { re, im }
}

/** Options for `freqz` and `groupDelay`. */
export interface ResponseOptions {
  /** Number of frequencies. Default 512. */
  n?: Size
  /** Cover [0, 2π) instead of [0, π). Default false. */
  whole?: boolean
  /** Include the last point (π) of the half range. Default false, as scipy. */
  includeNyquist?: boolean
  /**
   * The frequency axis: `hz` (in units of fs = 1/dt; cycles per sample when dt = 1) or `rad/sample` (scipy's default
   * without fs). Default `hz`.
   */
  axis?: 'hz' | 'rad/sample'
}

function frequencies({ n = 512, whole = false, includeNyquist = false }: ResponseOptions): Float64Array {
  const last = whole ? 2 * Math.PI : Math.PI
  const endpoint = includeNyquist && !whole
  const count = endpoint ? n - 1 : n
  return Float64Array.from({ length: n }, (_, i) => (last * i) / count)
}

/** The axis tag, the factor from rad/sample to it, and fs. */
function axisOf(sys: LtiSystem, options: ResponseOptions): { axis: Spectrum['axis']; scale: Scalar; fs: Scalar } {
  const fs = 1 / (sys.dt ?? 1)
  if ((options.axis ?? 'hz') === 'rad/sample') return { axis: 'rad/sample', scale: 1, fs }
  return { axis: sys.dt === 1 ? 'cycles/sample' : 'hz', scale: fs / (2 * Math.PI), fs }
}

/**
 * The frequency response H(e^{iω}) of a discrete system, as `scipy.signal.freqz`: n frequencies evenly spaced on
 * [0, π) (or [0, 2π) with `whole`), evaluated in the system's own representation (`frequencyResponse` of
 * `aifn-compute/systems`). Returns a `Spectrum` (`quantity: 'response'`, complex128 values [n]) with frequencies in Hz
 * (fs = 1/dt) or rad/sample; `magnitude`, `phase` and `decibels` of `aifn-compute/signal` read it.
 */
export function freqz(sys: LtiSystem, options: ResponseOptions = {}): Spectrum {
  if (sys.domain !== 'discrete') throw new DomainError('freqz', 'freqz: the system must be discrete')
  const w = frequencies(options)
  const { axis, scale, fs } = axisOf(sys, options)
  return spectrum({
    f: fromData(
      w.map((v) => v * scale),
      [w.length],
    ),
    axis,
    values: frequencyResponse(sys, w).values,
    quantity: 'response',
    sided: options.whole ? 'two' : 'one',
    fs,
  })
}

/** The result of `groupDelay`. */
export interface GroupDelay {
  /** Frequencies, in the units of `axis`. */
  f: Tensor
  axis: Spectrum['axis']
  /** τ(ω) in samples; NaN where the response is zero. */
  delay: Tensor
  /** How many frequencies had an undefined delay. */
  singular: Size
}

/**
 * Group delay τ(ω) = −dφ/dω in samples of a discrete system, as `scipy.signal.group_delay`: with c = b ∗ reverse(a),
 * τ = Re{Σ k c_k e^{−iωk} / Σ c_k e^{−iωk}} − (|a| − 1), plus the system's delay. Where the response is zero (the
 * ratio is undefined) the delay is NaN and `singular` counts those frequencies.
 */
export function groupDelay(sys: LtiSystem, options: ResponseOptions = {}): GroupDelay {
  if (sys.domain !== 'discrete') throw new DomainError('groupDelay', 'groupDelay: the system must be discrete')
  if (sys.repr.form === 'zpk') return zpkGroupDelay(sys.repr.zeros, sys.repr.poles, sys, options)
  const tf = toTransferFunction(sys).repr
  const bv = dense.data(tf.b)
  const av = dense.data(tf.a)
  const cc = new Float64Array(bv.length + av.length - 1)
  for (let i = 0; i < bv.length; i++) for (let j = 0; j < av.length; j++) cc[i + j] += bv[i] * av[av.length - 1 - j]
  const cr = cc.map((v, k) => v * k)
  const w = frequencies(options)
  let singular = 0
  const delay = w.map((omega) => {
    const den = evaluate(cc, omega)
    if (Math.hypot(den.re, den.im) < 10 * Number.EPSILON * cc.reduce((s, v) => s + Math.abs(v), 0)) {
      singular++
      return NaN
    }
    const num = evaluate(cr, omega)
    return (num.re * den.re + num.im * den.im) / (den.re * den.re + den.im * den.im) - (av.length - 1) + sys.delay
  })
  const { axis, scale } = axisOf(sys, options)
  return {
    f: fromData(
      w.map((v) => v * scale),
      [w.length],
    ),
    axis,
    delay: fromData(delay, [w.length]),
    singular,
  }
}

/**
 * Group delay of a discrete zeros–poles–gain system, root by root (no polynomial is formed, so high orders keep their
 * accuracy): each factor e^{iω} − r adds −Re(e^{iω}/(e^{iω} − r)) for a zero and +Re(…) for a pole. A zero on the unit
 * circle at ω makes the delay undefined there (NaN).
 */
function zpkGroupDelay(zeros: Tensor, poles: Tensor, sys: LtiSystem, options: ResponseOptions): GroupDelay {
  const z = toComplexFlat(zeros)
  const p = toComplexFlat(poles)
  const w = frequencies(options)
  let singular = 0
  const term = (c: number, s: number, r: ComplexNumber) => {
    const dr = c - r.re
    const di = s - r.im
    const d2 = dr * dr + di * di
    // Re(e^{iω} / (e^{iω} − r)) = Re(e^{iω} conj(e^{iω} − r)) / |e^{iω} − r|².
    return d2 === 0 ? NaN : (c * dr + s * di) / d2
  }
  const delay = w.map((omega) => {
    const c = Math.cos(omega)
    const s = Math.sin(omega)
    let tau = sys.delay
    for (const r of p) tau += term(c, s, r)
    for (const r of z) {
      if (Math.hypot(c - r.re, s - r.im) < 1e-12) {
        singular++
        return NaN
      }
      tau -= term(c, s, r)
    }
    return tau
  })
  const { axis, scale } = axisOf(sys, options)
  return {
    f: fromData(
      w.map((v) => v * scale),
      [w.length],
    ),
    axis,
    delay: fromData(delay, [w.length]),
    singular,
  }
}
