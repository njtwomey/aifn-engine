/**
 * Digital filters with scipy.signal's conventions: window-method FIR design (`firwin`, Kaiser's formulas), IIR design
 * from analog prototypes by the bilinear transform (`iirfilter`, `butter`, `cheby1`, `cheby2`; Oppenheim and Schafer,
 * 2010, "Discrete-Time Signal Processing", §7.1–7.3), filtering (`lfilter`, `sosfilt`), zero-phase filtering
 * (`filtfilt`, Gustafsson's initial conditions as in scipy), and frequency and group-delay responses.
 *
 * Designs return a discrete `LtiSystem` (sampling interval `dt` $= 1/f_s$, or 1 without `fs`), and the filtering and
 * response functions take one. Frequencies: without `fs`, cutoffs are fractions of the Nyquist frequency in $(0, 1)$,
 * as in scipy; with `fs`, they are in the same units as `fs`. The IIR designs return zeros, poles and gain by default
 * (scipy returns $b$ and $a$): pass `output: 'tf'` or `'sos'` for the others.
 *
 * Filtering is a composition over `aifn-compute/foundation/convolution`'s `linearFilter` primitive: `lfilter`,
 * `sosfilt`, `lfilterZi`, `sosfiltZi` and `filtfilt` accept coefficients `{ b, a }` (or sections) and samples as traced
 * values, so they are differentiable in the IIR coefficients, the initial state and the signal, and batch along other
 * axes.
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

/**
 * The normalised sinc, $\sin(\pi x)/(\pi x)$, and 1 at $x = 0$.
 *
 * @param x The argument.
 * @returns $\operatorname{sinc} x$.
 */
const sinc = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))

/** Options for `firwin`. */
export interface FirwinOptions {
  /** Window for the ideal impulse response, built symmetric with `numtaps` samples. Default hamming. */
  window?: WindowSpec
  /**
   * The band type, or whether the DC gain is 1: `true`/`lowpass`/`bandstop` pass zero frequency, `false`/`highpass`/
   * `bandpass` do not. Default true.
   */
  passZero?: boolean | 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  /** Scale so the gain is exactly 1 at the centre of the first passband. Default true. */
  scale?: boolean
  /**
   * Sampling frequency; the system gets `dt` $= 1/f_s$. Default 2 (cutoffs as fractions of Nyquist; the system's `dt`
   * is then 1).
   */
  fs?: Scalar
}

/**
 * A linear-phase FIR filter by the window method, as `scipy.signal.firwin`: the ideal impulse response
 * $h[n] = \sum_{\text{passbands}} \big(f_2 \operatorname{sinc}(f_2 m) - f_1 \operatorname{sinc}(f_1 m)\big)$, with
 * $m = n - (N - 1)/2$ and each passband $[f_1, f_2]$ in fractions of Nyquist, times a window. Throws `DomainError` for
 * a cutoff not strictly between 0 and Nyquist, and for an even number of taps when the filter passes the Nyquist
 * frequency (a high-pass or band-stop). Returns the FIR system $b(z^{-1})/1$: its taps are `sys.repr.b`.
 *
 * @param numtaps The number of taps $N$ (the filter's order plus 1).
 * @param cutoff One edge, or a list of band edges in increasing order (not checked), in units of `fs` (fractions of
 *   Nyquist without it). The bands alternate between pass and stop, starting as `passZero` says.
 * @param options The window, the band type (`passZero`), whether to scale the gain, and the sampling frequency.
 * @returns The FIR filter as a transfer function with `a` $= [1]$.
 *
 * @example A 5-tap half-band lowpass, against scipy.signal.firwin
 * // scipy: [0, 0.203712, 0.592575, 0.203712, 0].
 * const h = firwin(5, 0.5).repr.b
 * print('h =', h)
 * print('gain at DC =', sum(h))
 *
 * @example A high-pass passes Nyquist, so needs an odd number of taps
 * const h = firwin(5, 0.5, { passZero: false }).repr.b
 * print('h =', h)
 * print('gain at Nyquist =', sum(mul(h, tensor([1, -1, 1, -1, 1]))))
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

/**
 * The sampling interval of a design: $1/f_s$, or 1 when the frequencies are fractions of Nyquist.
 *
 * @param fs The design's sampling frequency, or undefined when none was given.
 * @returns The `dt` of the designed system.
 */
const dtOf = (fs: Scalar | undefined) => (fs === undefined ? 1 : 1 / fs)

/**
 * Kaiser's $\beta$ for a stopband attenuation of $A$ dB, as `scipy.signal.kaiser_beta` (Kaiser, 1974):
 * $0.1102(A - 8.7)$ for $A > 50$, $0.5842(A - 21)^{0.4} + 0.07886(A - 21)$ for $21 < A \le 50$, and 0 below.
 *
 * @param attenuation The stopband attenuation $A$, in dB (positive).
 * @returns The Kaiser window's $\beta$.
 *
 * @example The beta for 60 dB, against scipy.signal.kaiser_beta
 * // scipy: 5.65326.
 * print('beta =', kaiserBeta(60))
 * print('beta for 30 dB =', kaiserBeta(30))
 */
export function kaiserBeta(attenuation: Scalar): Scalar {
  const a = attenuation
  if (a > 50) return 0.1102 * (a - 8.7)
  if (a > 21) return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
  return 0
}

/**
 * The stopband attenuation of a Kaiser-window FIR filter, as `scipy.signal.kaiser_atten`:
 * $A = 2.285 (N - 1) \pi \Delta + 7.95$ dB for $N$ taps and transition width $\Delta$.
 *
 * @param numtaps The number of taps $N$.
 * @param width The transition width $\Delta$, as a fraction of Nyquist.
 * @returns The attenuation $A$, in dB.
 *
 * @example 65 taps and a transition of 0.1, against scipy.signal.kaiser_atten
 * // scipy: 53.8927.
 * print('attenuation (dB) =', kaiserAttenuation(65, 0.1))
 */
export function kaiserAttenuation(numtaps: Size, width: Scalar): Scalar {
  return 2.285 * (numtaps - 1) * Math.PI * width + 7.95
}

/**
 * Kaiser's design formulas, as `scipy.signal.kaiserord`: the taps $N = \lceil (A - 7.95)/(2.285 \pi \Delta) + 1 \rceil$
 * and the $\beta$ of `kaiserBeta` for a ripple (attenuation) of $A$ dB and a transition width $\Delta$. Throws
 * `DomainError` when $A < 8$ dB, where the formula does not hold.
 *
 * @param ripple The attenuation $A$ in dB; its sign is ignored.
 * @param width The transition width $\Delta$, as a fraction of Nyquist.
 * @returns `numtaps` and `beta`, for `firwin(numtaps, cutoff, { window: { name: 'kaiser', beta } })`.
 *
 * @example 60 dB and a transition of 0.1, against scipy.signal.kaiserord
 * // scipy: (74, 5.65326).
 * const { numtaps, beta } = kaiserOrder(60, 0.1)
 * print('numtaps =', numtaps)
 * print('beta =', beta)
 */
export function kaiserOrder(ripple: Scalar, width: Scalar): { numtaps: Size; beta: Scalar } {
  const a = Math.abs(ripple)
  if (a < 8) throw new DomainError('kaiserOrder', 'kaiserOrder: the attenuation must be at least 8 dB')
  return { numtaps: Math.ceil((a - 7.95) / 2.285 / (Math.PI * width) + 1), beta: kaiserBeta(a) }
}

// ── IIR design ────────────────────────────────────────────────────────────────────────────────────────────────────

/** An analog or digital zeros–poles–gain design: complex128 zeros and poles, a real gain. */
type Proto = { z: Tensor; p: Tensor; k: number }

/**
 * The product of the entries of a complex vector (1 when empty).
 *
 * @param v A complex128 vector.
 * @returns $\prod_i v_i$, as `{ re, im }`.
 */
function productOf(v: Tensor): ComplexNumber {
  let re = 1
  let im = 0
  for (const z of toComplexFlat(v)) [re, im] = [re * z.re - im * z.im, re * z.im + im * z.re]
  return { re, im }
}

/**
 * $\operatorname{Re}\big(\prod_i (-u_i) / \prod_j (-v_j)\big)$: the gain factor of the frequency transformations
 * (scipy's `lp2hp_zpk`, `lp2bs_zpk`).
 *
 * @param u The roots of the numerator, complex128.
 * @param v The roots of the denominator, complex128.
 * @returns The real part of the ratio.
 */
function gainRatio(u: Tensor, v: Tensor): number {
  const a = productOf(neg(u))
  const b = productOf(neg(v))
  return (a.re * b.re + a.im * b.im) / (b.re * b.re + b.im * b.im)
}

/** An empty complex128 vector: no zeros. */
const none = (): Tensor => zeros([0], 'complex128')
/**
 * A complex128 vector of `n` copies of `value`.
 *
 * @param value The complex value.
 * @param n How many copies.
 * @returns The vector.
 */
const repeat = (value: ComplexNumber, n: Size): Tensor => full([n], value, 'complex128')
/**
 * The angles $\theta_i = \pi(-n + 1 + 2i)/(2n)$, $i = 0, \dots, n - 1$, of the prototypes' poles.
 *
 * @param n The filter order.
 * @returns The $n$ angles, in radians, as a float64 vector.
 */
const angles = (n: Size): Tensor => tensor(Array.from({ length: n }, (_, i) => (Math.PI * (-n + 1 + 2 * i)) / (2 * n)))

/**
 * Analog Butterworth prototype (cutoff 1 rad/s), as `scipy.signal.buttap`: poles $p_i = -e^{i\theta_i}$ on the unit
 * circle in the left half-plane, no zeros, gain 1.
 *
 * @param n The filter order.
 * @returns The prototype's zeros, poles and gain.
 */
function buttap(n: Size): Proto {
  return { z: none(), p: neg(expj(angles(n))), k: 1 }
}

/**
 * Analog Chebyshev type I prototype with passband ripple $r_p$ dB, as `scipy.signal.cheb1ap`: poles
 * $p_i = -\sinh(\mu + i\theta_i)$ with $\varepsilon = \sqrt{10^{r_p/10} - 1}$ and
 * $\mu = \operatorname{asinh}(1/\varepsilon)/n$, no zeros, and a gain that makes the DC gain 1 for odd $n$ and
 * $1/\sqrt{1 + \varepsilon^2}$ (the bottom of the ripple) for even $n$.
 *
 * @param n The filter order.
 * @param rp The passband ripple $r_p$, in dB.
 * @returns The prototype's zeros, poles and gain.
 */
function cheb1ap(n: Size, rp: number): Proto {
  const eps = Math.sqrt(10 ** (0.1 * rp) - 1)
  const mu = Math.asinh(1 / eps) / n
  const theta = angles(n)
  const p = complex(mul(-Math.sinh(mu), cos(theta)), mul(-Math.cosh(mu), sin(theta))) as Tensor
  let k = productOf(neg(p)).re
  if (n % 2 === 0) k /= Math.sqrt(1 + eps * eps)
  return { z: none(), p, k }
}

/**
 * Analog Chebyshev type II prototype with stopband attenuation $r_s$ dB, as `scipy.signal.cheb2ap`: zeros
 * $i/\sin\big(m\pi/(2n)\big)$ for $m = -n + 1, -n + 3, \dots, n - 1$ ($m = 0$ left out), poles the reciprocals of the
 * type I poles for $\delta = 1/\sqrt{10^{r_s/10} - 1}$, and unit DC gain. The stopband edge is at 1 rad/s.
 *
 * @param n The filter order.
 * @param rs The stopband attenuation $r_s$, in dB.
 * @returns The prototype's zeros, poles and gain.
 */
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

/**
 * A complex128 vector from complex numbers.
 *
 * @param roots The entries, as `{ re, im }`.
 * @returns The vector, of their length.
 */
function complexVector(roots: readonly ComplexNumber[]): Tensor {
  const v = new Float64Array(2 * roots.length)
  roots.forEach((r, i) => {
    v[2 * i] = r.re
    v[2 * i + 1] = r.im
  })
  return fromData(v, [roots.length], 'complex128')
}

/**
 * $10^{x/10} - 1$ without cancellation for small $x$ (scipy's `_pow10m1`).
 *
 * @param x A level in dB.
 * @returns $10^{x/10} - 1$.
 */
const pow10m1 = (x: number) => Math.expm1(0.1 * x * Math.LN10)

/**
 * The modulus $m$ of an order-$n$ elliptic filter from $m_1 = \varepsilon^2/(10^{r_s/10} - 1)$, by nomes (scipy's
 * `_ellipdeg`): with $q = \exp\big(-\pi K'(m_1)/(n K(m_1))\big)$,
 * $m = 16 q \big(\sum_{k=0}^{7} q^{k(k+1)} / (1 + 2\sum_{k=1}^{8} q^{k^2})\big)^4$.
 *
 * @param n The filter order.
 * @param m1 The parameter $m_1$, in $(0, 1)$.
 * @returns The modulus $m$.
 */
function ellipdeg(n: Size, m1: number): number {
  const q = Math.exp((-Math.PI * ellipkm1(m1)) / ellipk(m1) / n)
  let num = 0
  let den = 1
  for (let k = 0; k <= 7; k++) num += q ** (k * (k + 1))
  for (let k = 1; k <= 8; k++) den += 2 * q ** (k * k)
  return 16 * q * (num / den) ** 4
}

/**
 * Analog elliptic (Cauer) prototype with passband ripple $r_p$ dB and stopband attenuation $r_s$ dB, as
 * `scipy.signal.ellipap` (Orfanidis, 2006, "Lecture notes on elliptic filter design"): zeros
 * $i/\big(\sqrt{m}\,\operatorname{sn}(jK/n \mid m)\big)$ and poles from the Jacobi functions at $jK/n$ and at
 * $v_0 = K F(\arctan(1/\varepsilon) \mid 1 - m_1)/(n K_1)$, with $K = K(m)$ and $K_1 = K(m_1)$. The passband edge is at
 * 1 rad/s, and the DC gain is 1 for odd $n$ and the bottom of the ripple for even $n$. Throws `DomainError` when $r_s$
 * is so large that $m_1$ underflows to 0.
 *
 * @param n The filter order.
 * @param rp The passband ripple $r_p$, in dB.
 * @param rs The stopband attenuation $r_s$, in dB.
 * @returns The prototype's zeros, poles and gain.
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

/**
 * $\log n!$, summed term by term (`n` is a small integer, a filter order).
 *
 * @param n A non-negative integer.
 * @returns $\log n!$.
 */
const logFactorial = (n: number) => {
  let s = 0
  for (let i = 2; i <= n; i++) s += Math.log(i)
  return s
}

/**
 * Analog Bessel–Thomson prototype, phase-normalised as `scipy.signal.besselap(n, norm='phase')`: the roots of the
 * reverse Bessel polynomial $\theta_n(s) = \sum_k \frac{(2n - k)!}{2^{n - k} k! (n - k)!} s^k$, scaled by
 * $\theta_n(0)^{-1/n}$ so the phase response matches Butterworth's at high frequency, and polished by Newton's method;
 * unit DC gain. A maximally flat group delay (Thomson, 1949).
 *
 * @param n The filter order.
 * @returns The prototype's zeros (none), poles and gain.
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

/**
 * Low-pass to low-pass of cutoff $\omega_0$ ($s \to s/\omega_0$), as `scipy.signal.lp2lp_zpk`: roots scaled by
 * $\omega_0$, gain by $\omega_0^{d}$ with $d$ the number of poles less the number of zeros.
 *
 * @param options The analog prototype.
 * @param options.z Its zeros, complex128.
 * @param options.p Its poles, complex128.
 * @param options.k Its gain.
 * @param wo The new cutoff $\omega_0$, in rad/s.
 * @returns The transformed zeros, poles and gain.
 */
function lp2lp({ z, p, k }: Proto, wo: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return { z: mul(z, wo), p: mul(p, wo), k: k * wo ** degree }
}

/**
 * Low-pass to high-pass of cutoff $\omega_0$ ($s \to \omega_0/s$), as `scipy.signal.lp2hp_zpk`: roots
 * $r \to \omega_0/r$, a zero at 0 for each excess pole, and the gain that keeps the high-frequency gain.
 *
 * @param options The analog prototype.
 * @param options.z Its zeros, complex128.
 * @param options.p Its poles, complex128.
 * @param options.k Its gain.
 * @param wo The cutoff $\omega_0$, in rad/s.
 * @returns The transformed zeros, poles and gain.
 */
function lp2hp({ z, p, k }: Proto, wo: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return { z: concat([div(wo, z), repeat({ re: 0, im: 0 }, degree)]), p: div(wo, p), k: k * gainRatio(z, p) }
}

/**
 * The two roots $r \pm \sqrt{r^2 - \omega_0^2}$ of $s^2 - 2rs + \omega_0^2$ for each entry $r$, which the caller has
 * already scaled ($r b_w/2$ of a prototype root for band-pass, $b_w/(2r)$ for band-stop).
 *
 * @param roots The scaled roots, complex128.
 * @param wo The centre frequency $\omega_0$, in rad/s.
 * @returns Twice as many roots: every `+` root, then every `-` root.
 */
function splitRoots(roots: Tensor, wo: number): Tensor {
  const root = sqrt(sub(square(roots), wo * wo))
  return concat([add(roots, root), sub(roots, root)])
}

/**
 * Low-pass to band-pass ($s \to (s^2 + \omega_0^2)/(s b_w)$), as `scipy.signal.lp2bp_zpk`: each root splits in two, a
 * zero at 0 is added for each excess pole, and the gain is scaled by $b_w^{d}$.
 *
 * @param options The analog prototype.
 * @param options.z Its zeros, complex128.
 * @param options.p Its poles, complex128.
 * @param options.k Its gain.
 * @param wo The centre frequency $\omega_0$ (geometric mean of the edges), in rad/s.
 * @param bw The bandwidth $b_w$ (the difference of the edges), in rad/s.
 * @returns The transformed zeros, poles and gain.
 */
function lp2bp({ z, p, k }: Proto, wo: number, bw: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return {
    z: concat([splitRoots(mul(z, bw / 2), wo), repeat({ re: 0, im: 0 }, degree)]),
    p: splitRoots(mul(p, bw / 2), wo),
    k: k * bw ** degree,
  }
}

/**
 * Low-pass to band-stop ($s \to s b_w/(s^2 + \omega_0^2)$), as `scipy.signal.lp2bs_zpk`: each root splits in two, a
 * pair of zeros at $\pm i\omega_0$ is added for each excess pole, and the gain keeps the DC gain.
 *
 * @param options The analog prototype.
 * @param options.z Its zeros, complex128.
 * @param options.p Its poles, complex128.
 * @param options.k Its gain.
 * @param wo The centre frequency $\omega_0$ (geometric mean of the edges), in rad/s.
 * @param bw The bandwidth $b_w$ (the difference of the edges), in rad/s.
 * @returns The transformed zeros, poles and gain.
 */
function lp2bs({ z, p, k }: Proto, wo: number, bw: number): Proto {
  const degree = p.shape[0] - z.shape[0]
  return {
    z: concat([splitRoots(div(bw / 2, z), wo), repeat({ re: 0, im: wo }, degree), repeat({ re: 0, im: -wo }, degree)]),
    p: splitRoots(div(bw / 2, p), wo),
    k: k * gainRatio(z, p),
  }
}

/**
 * The bilinear transform $s = 2 f_s (z - 1)/(z + 1)$, as `scipy.signal.bilinear_zpk`: each analog root $r$ maps to
 * $(2f_s + r)/(2f_s - r)$, each excess pole adds a zero at $z = -1$, and the gain is scaled by
 * $\operatorname{Re}\big(\prod (2f_s - z_i)/\prod (2f_s - p_j)\big)$.
 *
 * @param options The analog design.
 * @param options.z Its zeros, complex128.
 * @param options.p Its poles, complex128.
 * @param options.k Its gain.
 * @param fs The sampling frequency $f_s$ of the transform (`iirfilter` uses 2, with pre-warped edges).
 * @returns The digital zeros, poles and gain.
 */
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
  /** The band type. Default `lowpass` for one edge, `bandpass` for two. */
  btype?: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
  /** The prototype family. Default `butter`. */
  ftype?: 'butter' | 'cheby1' | 'cheby2' | 'ellip' | 'bessel'
  /** Passband ripple (dB), Chebyshev I and elliptic. Default 1. */
  rp?: number
  /** Stopband attenuation (dB), Chebyshev II and elliptic. Default 40. */
  rs?: number
  /**
   * Sampling frequency; the system gets `dt` $= 1/f_s$. Default 2 (edges as fractions of Nyquist; the system's `dt`
   * is then 1).
   */
  fs?: Scalar
  /**
   * The representation of the result: `zpk` (the design's exact form), `tf` or `sos`. Default `zpk` (scipy's default
   * is `ba`, here `tf`).
   */
  output?: 'zpk' | 'tf' | 'sos'
}

/**
 * An IIR filter of order $n$, as `scipy.signal.iirfilter`: an analog prototype, frequency-transformed to the
 * pre-warped edges $4\tan(\pi W_n/2)$ (for $f_s = 2$, with $W_n$ the edge as a fraction of Nyquist), then mapped by the
 * bilinear transform. For Butterworth the edge is the $-3$ dB point; Chebyshev I and elliptic, the passband edge
 * (where the gain leaves the ripple band); Chebyshev II, the stopband edge; Bessel, the phase-normalised edge (the
 * phase asymptote of Butterworth's). A band-pass or band-stop design has order $2n$. Throws `DomainError` for an edge
 * not strictly between 0 and Nyquist, a number of edges that does not suit the band type, or an order that is not a
 * positive integer.
 *
 * @param n The order $n$ of the low-pass prototype (a positive integer).
 * @param wn The edge (low-pass, high-pass) or the two edges (band-pass, band-stop), in units of `fs` (fractions of
 *   Nyquist without it).
 * @param options The band type, the family and its ripple and attenuation, the sampling frequency and the output
 *   representation.
 * @returns The discrete system (`dt` $= 1/f_s$) in the `output` representation.
 *
 * @example A first-order Butterworth band-pass, against scipy.signal.iirfilter
 * // scipy: b = [0.245237, 0, -0.245237], a = [1, -0.932938, 0.509525].
 * const { b, a } = iirfilter(1, [0.2, 0.4], { btype: 'bandpass', output: 'tf' }).repr
 * print('b =', b)
 * print('a =', a)
 *
 * @example The default output is zeros, poles and gain
 * const { zeros, poles, gain } = iirfilter(2, 0.5).repr
 * print('zeros =', zeros)
 * print('poles =', poles)
 * print('gain =', gain)
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

/**
 * A Butterworth filter (maximally flat passband), as `scipy.signal.butter`: `iirfilter` with `ftype: 'butter'`. The
 * edge is the $-3$ dB point.
 *
 * @param n The order of the low-pass prototype (a band design has twice this).
 * @param wn The edge, or two edges for a band design, in units of `fs` (fractions of Nyquist without it).
 * @param options The band type, the sampling frequency and the output representation (default `zpk`).
 * @returns The discrete system.
 *
 * @example A second-order low-pass at half Nyquist, against scipy.signal.butter
 * // scipy: b = [0.292893, 0.585786, 0.292893], a = [1, 0, 0.171573].
 * const { b, a } = butter(2, 0.5, { output: 'tf' }).repr
 * print('b =', b)
 * print('a =', a)
 * print('gain at DC =', sum(b) / sum(a))
 *
 * @example Second-order sections for a higher order
 * // scipy: [[0.004824, 0.009649, 0.004824, 1, -1.0486, 0.29614], [1, 2, 1, 1, -1.320913, 0.632739]].
 * print('sections =', butter(4, 0.2, { output: 'sos' }).repr.sections)
 */
export function butter(
  n: Size,
  wn: Scalar | readonly [Scalar, Scalar],
  options: Omit<IirOptions, 'ftype'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'butter' })
}

/**
 * A Chebyshev type I filter (equiripple passband of `rp` dB, monotone stopband), as `scipy.signal.cheby1`:
 * `iirfilter` with `ftype: 'cheby1'`. The edge is where the gain leaves the ripple band.
 *
 * @param n The order of the low-pass prototype (a band design has twice this).
 * @param rp The passband ripple, in dB.
 * @param wn The passband edge, or two edges for a band design, in units of `fs` (fractions of Nyquist without it).
 * @param options The band type, the sampling frequency and the output representation (default `zpk`).
 * @returns The discrete system.
 *
 * @example A second-order low-pass with 1 dB ripple, against scipy.signal.cheby1
 * // scipy: b = [0.307043, 0.614086, 0.307043], a = [1, 0.064064, 0.313968].
 * const { b, a } = cheby1(2, 1, 0.5, { output: 'tf' }).repr
 * print('b =', b)
 * print('a =', a)
 * // An even order starts at the bottom of the 1 dB ripple, 1 dB below unit gain.
 * print('gain at DC =', sum(b) / sum(a))
 */
export function cheby1(
  n: number,
  rp: Scalar,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rp'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'cheby1', rp })
}

/**
 * A Chebyshev type II filter (monotone passband, equiripple stopband `rs` dB down), as `scipy.signal.cheby2`:
 * `iirfilter` with `ftype: 'cheby2'`. The edge is the start of the stopband.
 *
 * @param n The order of the low-pass prototype (a band design has twice this).
 * @param rs The stopband attenuation, in dB.
 * @param wn The stopband edge, or two edges for a band design, in units of `fs` (fractions of Nyquist without it).
 * @param options The band type, the sampling frequency and the output representation (default `zpk`).
 * @returns The discrete system.
 *
 * @example A second-order low-pass 40 dB down, against scipy.signal.cheby2
 * // scipy: b = [0.02461, 0.016407, 0.02461], a = [1, -1.607879, 0.673506].
 * const { b, a } = cheby2(2, 40, 0.5, { output: 'tf' }).repr
 * print('b =', b)
 * print('a =', a)
 */
export function cheby2(
  n: number,
  rs: Scalar,
  wn: number | readonly [number, number],
  options: Omit<IirOptions, 'ftype' | 'rs'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'cheby2', rs })
}

/**
 * An elliptic (Cauer) filter: equiripple in both bands, the steepest transition for its order, as
 * `scipy.signal.ellip`: `iirfilter` with `ftype: 'ellip'`. The edge is where the gain leaves the passband ripple.
 *
 * @param n The order of the low-pass prototype (a band design has twice this).
 * @param rp The passband ripple, in dB.
 * @param rs The stopband attenuation, in dB.
 * @param wn The passband edge, or two edges for a band design, in units of `fs` (fractions of Nyquist without it).
 * @param options The band type, the sampling frequency and the output representation (default `zpk`).
 * @returns The discrete system.
 *
 * @example A second-order low-pass, against scipy.signal.ellip
 * // scipy: b = [0.31178, 0.611059, 0.31178], a = [1, 0.06755, 0.317716].
 * const { b, a } = ellip(2, 1, 40, 0.5, { output: 'tf' }).repr
 * print('b =', b)
 * print('a =', a)
 */
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
 * A Bessel–Thomson filter (maximally flat group delay, phase-normalised), as `scipy.signal.bessel(norm='phase')`:
 * `iirfilter` with `ftype: 'bessel'`. The bilinear transform keeps the magnitude shape but not the flat delay exactly.
 *
 * @param n The order of the low-pass prototype (a band design has twice this).
 * @param wn The phase-normalised edge, or two edges for a band design, in units of `fs` (fractions of Nyquist
 *   without it).
 * @param options The band type, the sampling frequency and the output representation (default `zpk`).
 * @returns The discrete system.
 *
 * @example A second-order low-pass, against scipy.signal.bessel
 * // scipy: b = [0.267949, 0.535898, 0.267949], a = [1, 0, 0.071797].
 * const { b, a } = bessel(2, 0.5, { output: 'tf' }).repr
 * print('b =', b)
 * print('a =', a)
 */
export function bessel(
  n: Size,
  wn: Scalar | readonly [Scalar, Scalar],
  options: Omit<IirOptions, 'ftype'> = {},
): LtiSystem {
  return iirfilter(n, wn, { ...options, ftype: 'bessel' })
}

// ── Filtering ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Filter coefficients as in scipy's `lfilter(b, a, x)`: the numerator `b` and denominator `a` in ascending powers of
 * $z^{-1}$ (numbers, arrays, tensors or traced values; `a[0]` must not be 0).
 */
export type FilterCoefficients = { b: Value | VectorLike; a: Value | VectorLike }

/** A filter: a discrete `LtiSystem` (any representation), or coefficients `{ b, a }`. */
export type FilterSpec = LtiSystem | FilterCoefficients

/** Options for `lfilter` and `sosfilt`. */
export type FilterOptions = {
  /**
   * The initial state (transposed direct form II, coefficients normalised by $a_0$): the shape of `x` with
   * $\max(n_a, n_b) - 1$ along the time axis, for $n_a$ and $n_b$ coefficients (a vector for a vector `x`). For
   * `sosfilt`, one such state per section stacked first, with 2 along the time axis. Default zeros (at rest).
   */
  zi?: Value | VectorLike
  /** The time axis of the samples (default $-1$, the last). */
  axis?: number
}

/**
 * A filtered signal `y` (a `Signal` for a `Signal` input) and the final state `zf`, the `zi` that continues the
 * filtering on the next block of samples.
 */
export type Filtered<Y, Z> = { y: Y; zf: Z }

/**
 * True for an `LtiSystem` (`kind: 'lti'`), as opposed to coefficients `{ b, a }` or sections.
 *
 * @param f The filter argument.
 * @returns Whether it is a system.
 */
const isSystem = (f: unknown): f is LtiSystem => (f as { kind?: unknown }).kind === 'lti'

/**
 * A number, array, tensor or traced value as a value (plain arrays become float64 tensors).
 *
 * @param v The coefficients or samples; a number becomes a vector of one.
 * @returns A tensor or traced value; tensors and traced values are returned as they are.
 */
function asValue(v: Value | VectorLike): Value {
  if (typeof v === 'number') return tensor([v])
  return isTraced(v) || isTensor(v) ? (v as Value) : tensor(Array.from(v as ArrayLike<number>))
}

/**
 * A slice spec selecting `range` along `axis` of a rank-`rank` value, and everything along the other axes.
 *
 * @param rank The number of axes of the value.
 * @param axis The axis to slice, non-negative.
 * @param range The selection along it, as `slice` takes it (`[start, stop, step]`).
 * @returns One spec per axis, for `slice(x, ...specs)`.
 */
const along = (rank: number, axis: number, range: SliceSpec): SliceSpec[] =>
  Array.from({ length: rank }, (_, k) => (k === axis ? range : null))

/**
 * The value reversed along one axis.
 *
 * @param x The value.
 * @param axis The axis to reverse, non-negative.
 * @returns `x` with its entries along `axis` in reverse order.
 */
const reverse = (x: Value, axis: number): Value => slice(x, ...along(shapeOfValue(x).length, axis, [null, null, -1]))

/**
 * A vector reshaped to lie along `axis` of a rank-`rank` value (shape $[1, \dots, n, \dots, 1]$), so that it
 * broadcasts against it.
 *
 * @param v A vector of $n$ entries.
 * @param rank The number of axes of the value it will meet.
 * @param axis The axis it lies along, non-negative.
 * @returns The reshaped vector.
 */
const alongAxis = (v: Value, rank: number, axis: number): Value =>
  reshape(
    v,
    Array.from({ length: rank }, (_, k) => (k === axis ? shapeOfValue(v)[0] : 1)),
  )

/**
 * The time axis of a rank-`rank` value, from a possibly negative `axis`. Throws `ShapeError` when it is out of range
 * or the value is a scalar.
 *
 * @param axis The axis option: negative counts from the end; undefined means $-1$.
 * @param rank The number of axes of the samples.
 * @param where The caller's name, for error messages.
 * @returns The axis, in $0, \dots, \text{rank} - 1$.
 */
function timeAxis(axis: number | undefined, rank: number, where: string): number {
  const a = (axis ?? -1) < 0 ? rank + (axis ?? -1) : axis!
  if (rank === 0 || a < 0 || a >= rank) throw new ShapeError(where, `${where}: axis ${axis ?? -1} out of range`)
  return a
}

/**
 * Samples from a `Signal` (kept for the output's metadata), a value or an array.
 *
 * @param x The samples, or a `Signal` of any number of channels.
 * @returns The samples as a value, and the `Signal` when `x` was one.
 */
function samples(x: SignalInput | Value): { data: Value; meta?: Signal } {
  if (isSignal(x)) return { data: x.data, meta: x }
  return { data: asValue(x as Value | VectorLike) }
}

/**
 * The output as a `Signal` like the input's when the input was one and the output is concrete, else as a value.
 *
 * @param y The filtered samples.
 * @param meta The input `Signal`, whose rate, start time, unit and channel names are copied; undefined for a value
 *   input.
 * @returns A `Signal`, or `y` itself.
 */
function wrap(y: Value, meta: Signal | undefined): Value | Signal {
  if (!meta || !isTensor(y)) return y
  return signal(y, {
    fs: meta.fs,
    t0: meta.t0,
    ...(meta.unit !== undefined ? { unit: meta.unit } : {}),
    ...(meta.channels !== undefined ? { channels: meta.channels } : {}),
  })
}

/**
 * Transfer-function coefficients normalised by $a_0$ and padded with zeros to a common length $K + 1$, where $K$ is the
 * filter's order. Throws `DomainError` for a continuous system or (for concrete coefficients) $a_0 = 0$, and
 * `ShapeError` for empty coefficients.
 *
 * @param f The filter: a discrete system (converted to a transfer function) or coefficients `{ b, a }`.
 * @param where The caller's name, for error messages.
 * @returns `b` and `a` divided by $a_0$, each of $K + 1$ entries, and $K$.
 */
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
 * The final transposed-direct-form-II state after filtering $x$ ($N$ samples along `axis`) into $y$ with normalised
 * $b$, $a$ (length $K + 1$):
 * $z_i = \sum_{l=0}^{L-1} \big(b_{i+1+l}\, x[N-1-l] - a_{i+1+l}\, y[N-1-l]\big) + z^{\text{init}}_{i+N}$ for
 * $i = 0, \dots, K - 1$, with $L = \min(K - i, N)$ and the last term only while $i + N < K$.
 *
 * @param b The normalised numerator, $K + 1$ entries.
 * @param a The normalised denominator, $K + 1$ entries.
 * @param K The filter order.
 * @param x The input samples.
 * @param y The filtered samples, of the shape of `x`.
 * @param axis The time axis, non-negative.
 * @param zi The initial state (the shape of `x` with $K$ along the time axis), or null for zeros.
 * @returns The final state, the shape of `x` with $K$ along the time axis.
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

/**
 * One transfer-function filter pass on a value: $y$ and the final state. Throws `ShapeError` when `zi` does not have
 * the shape of `x` with $K$ along the time axis.
 *
 * @param f The filter: a discrete system or coefficients `{ b, a }` (not sections).
 * @param x The samples.
 * @param axis The time axis, non-negative.
 * @param zi The initial state, or null to start at rest.
 * @param where The caller's name, for error messages.
 * @returns The filtered samples `y` and the final state `zf`.
 */
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

/**
 * The `zi` option as a value, or null when it was left out.
 *
 * @param zi The initial state as given.
 * @returns The state as a tensor or traced value, or null for zeros.
 */
const readState = (zi: Value | VectorLike | undefined): Value | null => (zi === undefined ? null : asValue(zi))

/**
 * Filters samples through a discrete system or coefficients `{ b, a }` by the difference equation
 * $\sum_k a_k y[t - k] = \sum_k b_k x[t - k]$ along `axis` (every other axis a separate signal), as
 * `scipy.signal.lfilter`. A second-order-sections system runs as `sosfilt`. `zi` is the initial state (default zeros),
 * and `zf` the final one. A `Signal` input gives a `Signal` output with the same sample rate and start time; a value
 * input gives a value, and the whole computation is differentiable in $b$, $a$, `zi` and $x$. Throws `DomainError`
 * for a continuous system or $a_0 = 0$, and `ShapeError` for a `zi` of the wrong shape or an axis out of range.
 *
 * @param f The filter: a discrete `LtiSystem` in any representation, or coefficients `{ b, a }`.
 * @param x The samples: a `Signal`, an array, or a tensor or traced value of any rank (filtered along `axis`).
 * @param options The initial state `zi` and the time `axis`.
 * @returns `y`, the filtered samples (a `Signal` for a `Signal` input), and `zf`, the final state.
 *
 * @example A 3-point moving average of a step
 * print('y =', lfilter({ b: [1 / 3, 1 / 3, 1 / 3], a: [1] }, [1, 1, 1, 1, 1]).y)
 *
 * @example A one-pole recursion: the impulse response decays geometrically
 * print('y =', lfilter({ b: [1], a: [1, -0.9] }, [1, 0, 0, 0]).y)
 *
 * @example The final state continues the filtering on the next block
 * const f = { b: [1 / 3, 1 / 3, 1 / 3], a: [1] }
 * const first = lfilter(f, [1, 2, 3])
 * const second = lfilter(f, [4, 5, 6], { zi: first.zf })
 * print('in blocks =', first.y, second.y)
 * print('at once =', lfilter(f, [1, 2, 3, 4, 5, 6]).y)
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

/**
 * Second-order sections, $S \times 6$ with rows $b_0, b_1, b_2, a_0, a_1, a_2$, from an sos system, a matrix or a
 * traced value. Throws `DomainError` for a system in another representation, and `ShapeError` when the matrix is not
 * $S \times 6$.
 *
 * @param sos The sections: a system with `repr.form` `'sos'`, nested arrays, or a tensor or traced value.
 * @param where The caller's name, for error messages.
 * @returns The $S \times 6$ sections as a value.
 */
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
 * Filters samples through a cascade of second-order sections, $S \times 6$ with rows $b_0, b_1, b_2, a_0, a_1, a_2$, as
 * `scipy.signal.sosfilt`: better conditioned than one high-order difference equation. `zi` has $S$ first, then the
 * shape of `x` with 2 along the time axis; `zf` has the same shape. Differentiable in the sections, `zi` and $x$.
 * Throws `DomainError` for a system not in sections, and `ShapeError` for sections that are not $S \times 6$.
 *
 * @param sos The sections: a system in second-order sections (`output: 'sos'`), nested arrays, or a tensor or traced
 *   value.
 * @param x The samples: a `Signal`, an array, or a tensor or traced value of any rank (filtered along `axis`).
 * @param options The initial state `zi` (one per section) and the time `axis`.
 * @returns `y`, the filtered samples (a `Signal` for a `Signal` input), and `zf`, the final state of every section.
 *
 * @example A fourth-order Butterworth step response, in sections and as one transfer function
 * const step = [1, 1, 1, 1, 1, 1, 1, 1]
 * print('sections =', sosfilt(butter(4, 0.2, { output: 'sos' }), step).y)
 * print('transfer function =', lfilter(butter(4, 0.2, { output: 'tf' }), step).y)
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
 * $(\Imat - \Cmat^\top)\zvec = \bvec_{1:} - \avec_{1:} b_0$ with $\Cmat$ the companion matrix of $\avec$
 * (coefficients normalised by $a_0$). Multiply by $x[0]$ to start a signal without a transient. A
 * second-order-sections system gives `sosfiltZi`. Differentiable in $b$ and $a$.
 *
 * @param f The filter: a discrete system or coefficients `{ b, a }`.
 * @returns The state $\zvec$, a vector of $K$ entries for a filter of order $K$ (empty for $K = 0$); for sections,
 *   `sosfiltZi`'s $S \times 2$.
 *
 * @example A step starts in steady state, against scipy.signal.lfilter_zi
 * // scipy: [0.902369, -0.235702].
 * const f = butter(2, 0.25, { output: 'tf' })
 * const zi = lfilterZi(f)
 * print('zi =', zi)
 * print('step of 2, no transient =', lfilter(f, [2, 2, 2, 2], { zi: mul(zi, 2) }).y)
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
 * Steady-state initial states, $S \times 2$, of a section cascade for a unit step, as `scipy.signal.sosfilt_zi`: each
 * section's `lfilterZi` scaled by the DC gain of the sections before it. Differentiable in the sections.
 *
 * @param sos The sections: a system in second-order sections, nested arrays, or a tensor or traced value.
 * @returns One state per section, $S \times 2$ ($0 \times 2$ for no sections).
 *
 * @example A fourth-order Butterworth, against scipy.signal.sosfilt_zi
 * // scipy: [[0.105049, -0.013962], [0.884742, -0.458804]].
 * const sos = butter(4, 0.25, { output: 'sos' })
 * const zi = sosfiltZi(sos)
 * print('zi =', zi)
 * print('unit step, no transient =', sosfilt(sos, [1, 1, 1, 1], { zi }).y)
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
  /**
   * Samples added at each end. Default $3\max(n_a, n_b)$ for $n_a$ and $n_b$ coefficients, or scipy's count for $S$
   * sections, $3(2S + 1 - \min(z_b, z_a))$ with $z_b$ ($z_a$) the sections whose $b_2$ ($a_2$) is zero. Ignored for
   * `padtype: 'none'`.
   */
  padlen?: Size
  /** The time axis (default $-1$, the last). */
  axis?: number
}

/**
 * Zero-phase filtering, as `scipy.signal.filtfilt` and `sosfiltfilt` (Gustafsson, 1996, IEEE Trans. Signal Process.
 * 44(4)): extend the signal by `padlen` samples at each end, filter forwards and backwards with steady-state initial
 * conditions scaled by the first sample of each pass, and trim. The result has no phase shift and the squared
 * magnitude response. A composition, so differentiable in the coefficients (or sections) and the signal. Throws
 * `DomainError` when the signal is not longer than `padlen`.
 *
 * @param f The filter: a discrete system (sections run as `sosfiltfilt`) or coefficients `{ b, a }`.
 * @param x The samples: a `Signal`, an array, or a tensor or traced value of any rank (filtered along `axis`).
 * @param options How the ends are extended (`padtype`, `padlen`) and the time `axis`.
 * @returns The filtered samples, the shape of `x` (a `Signal` for a `Signal` input).
 *
 * @example An impulse comes out symmetric: no phase shift, against scipy.signal.filtfilt
 * // scipy, samples 5 to 9: [0.117606, 0.248259, 0.313149, 0.248258, 0.117619].
 * const x = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
 * const y = filtfilt(butter(2, 0.3, { output: 'tf' }), x)
 * print('y[5..9] =', slice(y, [5, 10]))
 *
 * @example A forward pass delays the step, filtfilt does not
 * const f = butter(2, 0.2, { output: 'tf' })
 * const step = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
 * print('lfilter =', lfilter(f, step).y)
 * print('filtfilt =', filtfilt(f, step))
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

/**
 * $\sum_k c_k e^{-i\omega k}$ for real coefficients $c$, as `{ re, im }`.
 *
 * @param coef The coefficients $c_0, c_1, \dots$, in ascending powers of $z^{-1}$.
 * @param w The frequency $\omega$, in radians per sample.
 * @returns The polynomial at $z = e^{i\omega}$.
 */
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
  /** Cover $[0, 2\pi)$ instead of $[0, \pi)$. Default false. */
  whole?: boolean
  /** Include the last point ($\pi$) of the half range, so the grid is $[0, \pi]$. Default false, as scipy. */
  includeNyquist?: boolean
  /**
   * The frequency axis: `hz` (in units of $f_s = 1/$`dt`; cycles per sample when `dt` is 1) or `rad/sample` (scipy's
   * default without `fs`). Default `hz`.
   */
  axis?: 'hz' | 'rad/sample'
}

/**
 * The evaluation grid of `freqz` and `groupDelay`: $n$ evenly spaced frequencies from 0, in radians per sample.
 *
 * @param options The grid.
 * @param options.n The number of frequencies.
 * @param options.whole Whether the grid covers $[0, 2\pi)$ rather than $[0, \pi)$.
 * @param options.includeNyquist Whether the half grid ends on $\pi$ (ignored with `whole`).
 * @returns The $n$ frequencies.
 */
function frequencies({ n = 512, whole = false, includeNyquist = false }: ResponseOptions): Float64Array {
  const last = whole ? 2 * Math.PI : Math.PI
  const endpoint = includeNyquist && !whole
  const count = endpoint ? n - 1 : n
  return Float64Array.from({ length: n }, (_, i) => (last * i) / count)
}

/**
 * The axis tag, the factor from rad/sample to it, and $f_s$.
 *
 * @param sys The system, whose `dt` gives $f_s = 1/$`dt` (1 when it has none).
 * @param options The `axis` option: `rad/sample`, or Hz (tagged `cycles/sample` when `dt` is 1).
 * @returns The tag, the scale ($f_s/(2\pi)$ for Hz, 1 for rad/sample) and $f_s$.
 */
function axisOf(sys: LtiSystem, options: ResponseOptions): { axis: Spectrum['axis']; scale: Scalar; fs: Scalar } {
  const fs = 1 / (sys.dt ?? 1)
  if ((options.axis ?? 'hz') === 'rad/sample') return { axis: 'rad/sample', scale: 1, fs }
  return { axis: sys.dt === 1 ? 'cycles/sample' : 'hz', scale: fs / (2 * Math.PI), fs }
}

/**
 * The frequency response $H(e^{i\omega})$ of a discrete system, as `scipy.signal.freqz`: $n$ frequencies evenly spaced
 * on $[0, \pi)$ (or $[0, 2\pi)$ with `whole`), evaluated in the system's own representation (`frequencyResponse` of
 * `aifn-compute/systems`). Returns a `Spectrum` (`quantity: 'response'`, $n$ complex128 values) with frequencies in Hz
 * ($f_s = 1/$`dt`) or rad/sample; `magnitude`, `phase` and `decibels` of `aifn-compute/signal` read it. Throws
 * `DomainError` for a continuous system.
 *
 * @param sys The discrete system.
 * @param options The number of frequencies, the range, and the units of the frequency axis.
 * @returns The response as a `Spectrum`: `f`, `values` ($H$ at each frequency) and `fs`.
 *
 * @example Gain at DC, at the cutoff and at Nyquist
 * // A Butterworth low-pass at half Nyquist: unit gain at DC, 1/sqrt(2) at the cutoff, 0 at Nyquist.
 * const r = freqz(butter(2, 0.5), { n: 3, includeNyquist: true })
 * print('f (cycles/sample) =', r.f)
 * print('|H| =', complexAbs(r.values))
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
  /** The unit of `f`: hertz, cycles per sample (when `dt` is 1) or radians per sample. */
  axis: Spectrum['axis']
  /** $\tau(\omega)$ in samples; NaN where the response is zero. */
  delay: Tensor
  /** How many frequencies had an undefined delay. */
  singular: Size
}

/**
 * Group delay $\tau(\omega) = -d\varphi/d\omega$ in samples of a discrete system, as `scipy.signal.group_delay`: with
 * $c = b * \operatorname{reverse}(a)$,
 * $\tau = \operatorname{Re}\big(\sum_k k c_k e^{-i\omega k} / \sum_k c_k e^{-i\omega k}\big) - (n_a - 1)$ for $n_a$
 * coefficients in $a$, plus the system's delay. A zeros–poles–gain system is evaluated root by root instead. Where the
 * response is zero (the ratio is undefined) the delay is NaN and `singular` counts those frequencies. Throws
 * `DomainError` for a continuous system.
 *
 * @param sys The discrete system.
 * @param options The number of frequencies, the range, and the units of the frequency axis.
 * @returns The frequencies `f`, their `axis`, the `delay` at each, and the `singular` count.
 *
 * @example A symmetric FIR filter delays every frequency by half its length
 * // Five taps: a delay of 2 samples.
 * print('delay =', groupDelay(firwin(5, 0.5), { n: 4 }).delay)
 *
 * @example A Butterworth low-pass delays most near its cutoff
 * const g = groupDelay(butter(4, 0.25), { n: 8 })
 * print('f =', g.f)
 * print('delay =', g.delay)
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
 * accuracy): each factor $e^{i\omega} - r$ adds $-\operatorname{Re}\big(e^{i\omega}/(e^{i\omega} - r)\big)$ for a zero
 * and $+\operatorname{Re}\big(e^{i\omega}/(e^{i\omega} - r)\big)$ for a pole. A zero on the unit circle at $\omega$
 * (within $10^{-12}$) makes the delay undefined there (NaN).
 *
 * @param zeros The zeros, complex128.
 * @param poles The poles, complex128.
 * @param sys The system, for its delay and `dt`.
 * @param options The grid and the units of the frequency axis.
 * @returns The frequencies, their axis, the delay at each, and how many were undefined.
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
