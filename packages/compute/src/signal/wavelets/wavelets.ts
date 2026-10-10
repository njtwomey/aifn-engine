/**
 * Wavelets: orthogonal Daubechies filters, the periodic discrete wavelet transform by Mallat's pyramid algorithm
 * (Mallat, 1989, IEEE Trans. PAMI 11(7)), the cascade algorithm for the scaling and wavelet functions (Daubechies,
 * 1992, "Ten Lectures on Wavelets", §6.5), the Morlet continuous wavelet transform computed per scale by FFT
 * (Torrence and Compo, 1998, Bull. Amer. Meteor. Soc. 79(1)), and wavelet shrinkage (Donoho and Johnstone, 1994).
 *
 * The discrete transforms share pywt's filter convention: the scaling filter $h$ (`rec_lo`) and the high-pass
 * $g[n] = (-1)^n h[L - 1 - n]$ of an $L$-tap wavelet. They wrap around the end of the signal (periodic extension), so
 * one level halves an even length exactly and is orthogonal: energy is preserved and the inverse is the adjoint. Names
 * outside `WaveletName` throw `DomainError`; an odd length at any level throws `ShapeError`.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { fft, ifft, nextPowerOfTwo } from 'aifn-compute/foundation/fourier'
import type { Scalar, Signal, Size, TimeFrequency, VectorLike } from 'aifn-compute/foundation/contracts'
import { complexValues, readSamples, signal, timeFrequency, type SignalInput } from '../signal'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** The orthogonal wavelets available. `db1` is Haar. */
export type WaveletName = 'haar' | 'db1' | 'db2' | 'db3' | 'db4' | 'db5' | 'db6' | 'db7' | 'db8' | 'db9' | 'db10'

/**
 * Scaling (low-pass reconstruction) filters $h$ with $\sum_n h[n] = \sqrt{2}$ and $\sum_n h[n]\, h[n - 2k] = \delta[k]$
 * (pywt's `rec_lo`). `dbN` has $N$ vanishing moments and $2N$ taps (Daubechies, 1988, Comm. Pure Appl. Math. 41(7)).
 * Computed by spectral factorisation with minimum-phase roots in 60-digit arithmetic (mpmath), so each is exact to
 * double precision.
 */
const SCALING: Record<WaveletName, readonly number[]> = {
  haar: [Math.SQRT1_2, Math.SQRT1_2],
  db1: [Math.SQRT1_2, Math.SQRT1_2],
  db2: [0.48296291314453416, 0.8365163037378079, 0.2241438680420134, -0.12940952255126037],
  db3: [
    0.33267055295008263, 0.8068915093110925, 0.45987750211849154, -0.13501102001025458, -0.08544127388202666,
    0.03522629188570953,
  ],
  db4: [
    0.2303778133088965, 0.7148465705529157, 0.6308807679298589, -0.027983769416859854, -0.18703481171909309,
    0.030841381835560764, 0.0328830116668852, -0.010597401785069032,
  ],
  db5: [
    0.16010239797419293, 0.6038292697971896, 0.7243085284377729, 0.13842814590132074, -0.24229488706638203,
    -0.032244869584638375, 0.07757149384004572, -0.006241490212798274, -0.012580751999081999, 0.0033357252854737712,
  ],
  db6: [
    0.11154074335010947, 0.49462389039845306, 0.7511339080210954, 0.31525035170919763, -0.22626469396543983,
    -0.12976686756726194, 0.09750160558732304, 0.027522865530305727, -0.03158203931748603, 0.0005538422011614961,
    0.004777257510945511, -0.0010773010853084796,
  ],
  db7: [
    0.07785205408500918, 0.3965393194819173, 0.7291320908462351, 0.4697822874051931, -0.14390600392856498,
    -0.22403618499387498, 0.07130921926683026, 0.08061260915108308, -0.03802993693501441, -0.01657454163066688,
    0.01255099855609984, 0.0004295779729213665, -0.0018016407040474908, 0.00035371379997452024,
  ],
  db8: [
    0.05441584224310401, 0.31287159091429995, 0.6756307362972898, 0.5853546836542067, -0.015829105256349306,
    -0.2840155429615469, 0.0004724845739132828, 0.12874742662047847, -0.017369301001807547, -0.044088253930794755,
    0.013981027917398282, 0.008746094047405777, -0.004870352993451574, -0.00039174037337694705, 0.0006754494064505693,
    -0.00011747678412476953,
  ],
  db9: [
    0.038077947363878345, 0.24383467461259034, 0.6048231236901112, 0.6572880780513005, 0.13319738582500756,
    -0.2932737832791749, -0.09684078322297646, 0.14854074933810638, 0.03072568147933338, -0.06763282906132997,
    0.00025094711483145197, 0.022361662123679096, -0.004723204757751397, -0.00428150368246343, 0.0018476468830562265,
    0.00023038576352319597, -0.0002519631889427101, 3.93473203162716e-5,
  ],
  db10: [
    0.026670057900555554, 0.1881768000776915, 0.5272011889317256, 0.6884590394536035, 0.2811723436605775,
    -0.24984642432731538, -0.19594627437737705, 0.12736934033579325, 0.09305736460357235, -0.07139414716639708,
    -0.029457536821875813, 0.033212674059341, 0.0036065535669561697, -0.010733175483330575, 0.001395351747052901,
    0.001992405295185056, -0.0006858566949597116, -0.00011646685512928545, 9.358867032006959e-5, -1.3264202894521244e-5,
  ],
}

/** The four filters of an orthogonal wavelet, in pywt's convention, each a rank-1 tensor of $L$ taps. */
export interface WaveletFilters {
  /** The wavelet's name, as given. */
  name: WaveletName
  /** Decomposition low-pass: the scaling filter $h$ reversed. */
  decLo: Tensor
  /** Decomposition high-pass: the high-pass $g$ reversed. */
  decHi: Tensor
  /** Reconstruction low-pass $h$ (the scaling filter). */
  recLo: Tensor
  /** Reconstruction high-pass $g[n] = (-1)^n h[L - 1 - n]$. */
  recHi: Tensor
  /** The number of vanishing moments of the wavelet: $N$ for `dbN`, 1 for `haar`. */
  vanishingMoments: Size
}

/**
 * A copy of the scaling filter $h$ of a wavelet. Throws `DomainError` for an unknown name.
 *
 * @param name The wavelet, e.g. `'db2'`.
 * @returns The $L$ taps of $h$, a fresh array the caller may modify.
 */
function scaling(name: WaveletName): number[] {
  const h = SCALING[name]
  if (!h) throw new DomainError('scaling', `unknown wavelet ${name}`)
  return [...h]
}

/**
 * The quadrature-mirror high-pass $g[n] = (-1)^n h[L - 1 - n]$ of a scaling filter $h$.
 *
 * @param h The $L$ taps of the scaling filter; not modified.
 * @returns The $L$ taps of $g$.
 */
const highpass = (h: readonly number[]) => h.map((_, n) => (n % 2 === 0 ? 1 : -1) * h[h.length - 1 - n])

/**
 * The filters of an orthogonal wavelet (pywt's `Wavelet(name).filter_bank`). Throws `DomainError` for an unknown name.
 *
 * @param name The wavelet: `'haar'` (the same as `'db1'`) or `'db2'` to `'db10'`.
 * @returns The decomposition and reconstruction low- and high-pass filters, with the number of vanishing moments.
 *
 * @example Haar and Daubechies 2
 * const haar = waveletFilters('haar')
 * print('haar: recLo =', haar.recLo, ' recHi =', haar.recHi)
 * const db2 = waveletFilters('db2')
 * print('db2: recLo =', db2.recLo, ' decLo =', db2.decLo)
 * print('db2: sum of recLo =', sum(db2.recLo), ' vanishing moments =', db2.vanishingMoments)
 */
export function waveletFilters(name: WaveletName): WaveletFilters {
  const h = scaling(name)
  const g = highpass(h)
  return {
    name,
    decLo: fromData(Float64Array.from(h).reverse()),
    decHi: fromData(Float64Array.from(g).reverse()),
    recLo: fromData(Float64Array.from(h)),
    recHi: fromData(Float64Array.from(g)),
    vanishingMoments: name === 'haar' ? 1 : Number(name.slice(2)),
  }
}

/**
 * One level of the periodic analysis: $a[k] = \sum_m h[m]\, x[(2k + m) \bmod n]$ and
 * $d[k] = \sum_m g[m]\, x[(2k + m) \bmod n]$. Throws `ShapeError` when $n$ is odd.
 *
 * @param x The $n$ samples; not modified.
 * @param h The scaling filter (low-pass).
 * @param g The high-pass filter, as `highpass` makes it from `h`.
 * @returns The $n/2$ approximation coefficients `approx` ($a$) and $n/2$ detail coefficients `detail` ($d$).
 */
function analysis(x: Float64Array, h: number[], g: number[]) {
  const n = x.length
  if (n % 2) throw new ShapeError('dwt', 'dwt: the signal length must be even at every level')
  const half = n / 2
  const approx = new Float64Array(half)
  const detail = new Float64Array(half)
  for (let k = 0; k < half; k++) {
    let a = 0
    let d = 0
    for (let m = 0; m < h.length; m++) {
      const v = x[(2 * k + m) % n]
      a += h[m] * v
      d += g[m] * v
    }
    approx[k] = a
    detail[k] = d
  }
  return { approx, detail }
}

/**
 * One level of the periodic synthesis, the adjoint (and so the inverse) of `analysis`:
 * $x[(2k + m) \bmod n] \mathrel{+}= h[m]\, a[k] + g[m]\, d[k]$ with $n$ twice the number of coefficients.
 *
 * @param approx The approximation coefficients $a$; not modified.
 * @param detail The detail coefficients $d$, as many as `approx` (not checked).
 * @param h The scaling filter (low-pass).
 * @param g The high-pass filter, as `highpass` makes it from `h`.
 * @returns The $n$ reconstructed samples.
 */
function synthesis(approx: Float64Array, detail: Float64Array, h: number[], g: number[]): Float64Array {
  const n = approx.length * 2
  const x = new Float64Array(n)
  for (let k = 0; k < approx.length; k++)
    for (let m = 0; m < h.length; m++) x[(2 * k + m) % n] += h[m] * approx[k] + g[m] * detail[k]
  return x
}

/**
 * One level of the periodic orthogonal DWT: $a[k] = \sum_m h[m]\, x[(2k + m) \bmod n]$,
 * $d[k] = \sum_m g[m]\, x[(2k + m) \bmod n]$. An orthogonal transform: energy is preserved and `idwt` inverts it
 * exactly. For `dbN` the coefficients are pywt's `'periodization'` coefficients of $x$ rolled by $(N + 1) \bmod 2$
 * samples, rolled back by $\lfloor N/2 \rfloor$ coefficients (checked against PyWavelets in the fixtures); for Haar
 * they are pywt's own. Throws `ShapeError` for an odd length and `DomainError` for an unknown wavelet.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples) of even length $n$. Its sample rate is not used.
 * @param wavelet The orthogonal wavelet.
 * @returns The $n/2$ approximation coefficients `approx` ($a$) and the $n/2$ detail coefficients `detail` ($d$).
 *
 * @example Haar on [1, 2, 3, 4]
 * // Pairwise sums and differences, divided by sqrt(2), as pywt.dwt([1, 2, 3, 4], 'haar').
 * const { approx, detail } = dwt([1, 2, 3, 4])
 * print('approx =', approx)
 * print('detail =', detail)
 *
 * @example Energy is preserved
 * const x = [3, 1, 4, 1, 5, 9, 2, 6]
 * const { approx, detail } = dwt(x, 'db2')
 * print('energy of x =', sum(square(tensor(x))))
 * print('energy of the coefficients =', add(sum(square(approx)), sum(square(detail))))
 */
export function dwt(x: SignalInput, wavelet: WaveletName = 'haar'): { approx: Tensor; detail: Tensor } {
  const h = scaling(wavelet)
  const r = analysis(readSamples(x, 'dwt').values, h, highpass(h))
  return { approx: fromData(r.approx, [r.approx.length]), detail: fromData(r.detail, [r.detail.length]) }
}

/**
 * The inverse of `dwt`: $x[(2k + m) \bmod n] \mathrel{+}= h[m]\, a[k] + g[m]\, d[k]$ (the adjoint, which is the
 * inverse). Throws `DomainError` for an unknown wavelet.
 *
 * @param approx The approximation coefficients $a$, as `dwt` returns them.
 * @param detail The detail coefficients $d$, as many as `approx` (not checked).
 * @param wavelet The wavelet the coefficients were computed with.
 * @returns The $n$ samples, $n$ twice the number of coefficients, as a rank-1 tensor.
 *
 * @example Round trip
 * const { approx, detail } = dwt([1, 2, 3, 4], 'db2')
 * print('approx =', approx, ' detail =', detail)
 * print('idwt =', idwt(approx, detail, 'db2'))
 */
export function idwt(approx: VectorLike, detail: VectorLike, wavelet: WaveletName = 'haar'): Tensor {
  const h = scaling(wavelet)
  const x = synthesis(dense.toF64(approx, 'idwt'), dense.toF64(detail, 'idwt'), h, highpass(h))
  return fromData(x, [x.length])
}

/** A multilevel decomposition: the coarsest approximation and details from finest (level 1) to coarsest. */
export interface WaveletDecomposition {
  /** The approximation coefficients at the coarsest level $J$: $n / 2^J$ values. */
  approx: Tensor
  /** The detail coefficients of each level, finest first: level $j$ (entry $j - 1$) has $n / 2^j$ values. */
  details: Tensor[]
  /** The wavelet used, which `waverec` reconstructs with. */
  wavelet: WaveletName
  /** The sample rate of the decomposed signal, so `waverec` returns a `Signal` on the same axis. */
  fs: Scalar
  /** The start time of the decomposed signal, in seconds. */
  t0: Scalar
}

/**
 * $J$ levels of the periodic DWT, each applied to the previous level's approximation (Mallat's pyramid). The order of
 * `details` is the reverse of pywt's `wavedec` list, which puts the coarsest first. Throws `ShapeError` when the
 * length is not divisible by $2^J$.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples); its length must be divisible by $2^J$.
 * @param wavelet The orthogonal wavelet.
 * @param levels The number of levels $J$.
 * @returns The coarsest approximation, the details from finest to coarsest, the wavelet, and the signal's `fs` and
 *   `t0`.
 *
 * @example Two Haar levels of a ramp
 * const d = wavedec([1, 2, 3, 4, 5, 6, 7, 8], 'haar', 2)
 * print('approx (level 2) =', d.approx)
 * print('details (level 1, level 2) =', d.details)
 */
export function wavedec(x: SignalInput, wavelet: WaveletName = 'haar', levels: Size = 1): WaveletDecomposition {
  const h = scaling(wavelet)
  const g = highpass(h)
  const input = readSamples(x, 'wavedec')
  let a = input.values
  if (a.length % 2 ** levels)
    throw new ShapeError('wavedec', `wavedec: length ${a.length} is not divisible by 2^${levels}`)
  const details: Tensor[] = []
  for (let j = 0; j < levels; j++) {
    const r = analysis(a, h, g)
    details.push(fromData(r.detail, [r.detail.length]))
    a = r.approx
  }
  return { approx: fromData(a, [a.length]), details, wavelet, fs: input.fs, t0: input.t0 }
}

/**
 * Reconstruct the signal from `wavedec`, coarsest level first. Zeroing some details first gives a denoised or
 * smoothed signal.
 *
 * @param d The decomposition, as `wavedec` returns it, possibly with its coefficients changed.
 * @returns A `Signal` with the decomposition's `fs` and `t0`.
 *
 * @example Perfect reconstruction, and a smoothed ramp
 * const d = wavedec([1, 2, 3, 4, 5, 6, 7, 8], 'db2', 2)
 * print('reconstructed =', waverec(d).data)
 * const h = wavedec([1, 2, 3, 4, 5, 6, 7, 8], 'haar', 2)
 * print('without the details =', waverec({ ...h, details: h.details.map((c) => zeros(c.shape)) }).data)
 */
export function waverec(d: WaveletDecomposition): Signal {
  const h = scaling(d.wavelet)
  const g = highpass(h)
  let a: Float64Array = dense.toF64(d.approx, 'waverec')
  for (let j = d.details.length - 1; j >= 0; j--) a = synthesis(a, dense.toF64(d.details[j], 'waverec'), h, g)
  return signal(fromData(Float64Array.from(a), [a.length]), { fs: d.fs, t0: d.t0 })
}

/**
 * The scaling function $\phi$ and wavelet $\psi$ by the cascade algorithm: iterate the two-scale equation
 * $\phi(t) = \sqrt{2} \sum_n h[n]\, \phi(2t - n)$ from a unit impulse, with
 * $\psi(t) = \sqrt{2} \sum_n g[n]\, \phi(2t - n)$. After $j$ steps (`iterations`) the samples approximate $\phi$ and
 * $\psi$ on a grid of spacing $2^{-j}$ over $[0, L - 1]$ ($L$ taps), as pywt's `Wavelet.wavefun`, with
 * $\phi = \psi = 0$ at $t = 0$. Throws `DomainError` for an unknown wavelet.
 *
 * @param wavelet The orthogonal wavelet.
 * @param iterations The number of cascade steps $j$: the grid has $(L - 1) 2^j + 1$ points.
 * @returns The grid `t` and the samples `phi` and `psi` on it, rank-1 tensors of the same length.
 *
 * @example Haar's box and step
 * const { t, phi, psi } = wavefun('haar', 2)
 * print('t =', t)
 * print('phi =', phi)
 * print('psi =', psi)
 *
 * @example The scaling function integrates to one
 * const { t, phi } = wavefun('db2', 6)
 * print('grid points =', t.shape[0], ' integral of phi =', mul(sum(phi), 2 ** -6))
 */
export function wavefun(wavelet: WaveletName = 'db2', iterations: Size = 8): { t: Tensor; phi: Tensor; psi: Tensor } {
  const h = scaling(wavelet)
  const upsample = (c: number[]) => c.flatMap((v, i) => (i < c.length - 1 ? [v, 0] : [v]))
  const conv = (a: number[], b: number[]) => {
    const out = Array<number>(a.length + b.length - 1).fill(0)
    a.forEach((av, i) => b.forEach((bv, j) => (out[i + j] += av * bv)))
    return out
  }
  const scaled = h.map((v) => v * Math.SQRT2)
  const g = highpass(h).map((v) => v * Math.SQRT2)
  let phi = [1]
  let psi = [1]
  for (let i = 0; i < iterations; i++) {
    phi = conv(upsample(phi), scaled)
    // ψ(t) = √2 Σ g[n] φ(2t − n): the first step uses g, later steps refine by h.
    psi = i === 0 ? g.slice() : conv(upsample(psi), scaled)
  }
  // The cascade's sample k approximates the function at t = (k + 1)·2^−j: place it there, on the grid
  // t = i·2^−j, i = 0 … (L − 1)·2^j, with φ = ψ = 0 at both ends of the support (pywt's `wavefun`).
  const step = 2 ** -iterations
  const n = (h.length - 1) * 2 ** iterations + 1
  const onGrid = (v: number[]) => Float64Array.from({ length: n }, (_, i) => (i === 0 ? 0 : (v[i - 1] ?? 0)))
  return {
    t: fromData(Float64Array.from({ length: n }, (_, i) => i * step)),
    phi: fromData(onGrid(phi)),
    psi: fromData(onGrid(psi)),
  }
}

/**
 * The Morlet wavelet $\psi(t) = \pi^{-1/4} e^{i\omega_0 t} e^{-t^2/2}$ (without the small admissibility correction).
 *
 * @param t The times at which to evaluate it, in units of the scale.
 * @param options Options.
 * @param options.omega0 The centre frequency $\omega_0$, in radians per unit time (default 6).
 * @returns The complex values $\psi(t)$, a complex128 tensor with one entry per time.
 *
 * @example At the centre and half a unit away
 * // psi(0) = pi^(-1/4) = 0.7511; |psi(0.5)| = 0.7511 exp(-1/8) = 0.6629.
 * const psi = morlet([0, 0.5])
 * print('psi =', psi)
 * print('|psi| =', complexAbs(psi))
 */
export function morlet(t: VectorLike, { omega0 = 6 }: { omega0?: Scalar } = {}): Tensor {
  const ts = dense.toF64(t, 'morlet')
  const k = Math.PI ** -0.25
  const out = new Float64Array(2 * ts.length)
  ts.forEach((v, i) => {
    out[2 * i] = k * Math.cos(omega0 * v) * Math.exp((-v * v) / 2)
    out[2 * i + 1] = k * Math.sin(omega0 * v) * Math.exp((-v * v) / 2)
  })
  return fromData(out, [ts.length], 'complex128')
}

/**
 * True when the values are evenly spaced on a log scale (and not also evenly spaced linearly): at least three, the
 * first positive, with a constant ratio other than 1.
 *
 * @param f The frequencies, in order.
 * @returns Whether `f` is a geometric sequence.
 */
function geometric(f: ArrayLike<number>): boolean {
  if (f.length < 3 || !(f[0] > 0)) return false
  const r = f[1] / f[0]
  if (Math.abs(r - 1) < 1e-12) return false
  for (let i = 2; i < f.length; i++) if (Math.abs(f[i] / f[i - 1] - r) > 1e-9 * r) return false
  return true
}

/**
 * A continuous wavelet transform: a `TimeFrequency` raster (`method: 'cwt'`; `frequencyScale` `'log'` for geometric
 * frequencies) whose values are the complex coefficients $W(a, b)$, complex128 $[f, t]$, with the scalogram's
 * magnitude $\abs{W}$ and the scales.
 */
export type Cwt = TimeFrequency & {
  /** $\abs{W}$, the scalogram's magnitude, $[f, t]$. */
  magnitude: Tensor
  /** Scales $a = \omega_0 / (2\pi f)$, in seconds. */
  scales: Tensor
}

/**
 * The Morlet continuous wavelet transform
 * $W(a, b) = \frac{1}{\sqrt{a}} \int x(t)\, \psi^*\!\left(\frac{t - b}{a}\right) dt$ at the given frequencies,
 * with $a = \omega_0 / (2\pi f)$. Computed per scale in the frequency domain, where the analytic Morlet is a Gaussian
 * at $\omega_0 / a$ (negative frequencies are dropped); the signal is zero-padded to a power of two at least twice its
 * length to avoid wrap-around. $f_s$ comes from the signal (or the `fs` option); frequencies are in Hz.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param frequencies The analysis frequencies $f$ in Hz, each positive; one row of the result per frequency, in the
 *   order given.
 * @param options `fs`, the sample rate in Hz (default the signal's, or 1 for bare samples), and `omega0`, the
 *   Morlet's centre frequency $\omega_0$ (default 6).
 * @returns The transform: `values` (complex, $[f, t]$), `magnitude`, `scales`, and the axes `t` (the sample times) and
 *   `f`.
 *
 * @example A 4 Hz sinusoid
 * // One second at 64 Hz: halfway through, the 4 Hz row is the largest.
 * const x = Array.from({ length: 64 }, (_, i) => Math.sin((2 * Math.PI * 4 * i) / 64))
 * const W = cwt(x, [2, 4, 8], { fs: 64 })
 * print('scales =', W.scales, ' frequency scale =', W.frequencyScale)
 * print('|W| at t = 0.5 s (2, 4, 8 Hz) =', W.magnitude.data[32], W.magnitude.data[96], W.magnitude.data[160])
 */
export function cwt(x: SignalInput, frequencies: VectorLike, options: { fs?: Scalar; omega0?: Scalar } = {}): Cwt {
  const { omega0 = 6 } = options
  const input = readSamples(x, 'cwt', options.fs)
  const { fs, values: v } = input
  const freqs = dense.toF64(frequencies, 'cwt')
  const n = v.length
  const size = nextPowerOfTwo(2 * n)
  const X = fft(fromData(Float64Array.from(v)), { n: size }).data as Float64Array
  const outRe = new Float64Array(freqs.length * n)
  const outIm = new Float64Array(freqs.length * n)
  const mag = new Float64Array(freqs.length * n)
  const scales = new Float64Array(freqs.length)
  // Row r of `filtered` is the signal's spectrum times the Morlet's at scale a_r; one inverse FFT along the rows.
  const filtered = new Float64Array(2 * freqs.length * size)
  freqs.forEach((f, row) => {
    const a = omega0 / (2 * Math.PI * f)
    scales[row] = a
    for (let k = 0; k < size; k++) {
      const w = (2 * Math.PI * fs * (k <= size / 2 ? k : k - size)) / size
      const psiHat = w > 0 ? Math.PI ** -0.25 * Math.sqrt(2 * Math.PI) * Math.exp(-0.5 * (a * w - omega0) ** 2) : 0
      const gain = Math.sqrt(a) * psiHat
      const p = 2 * (row * size + k)
      filtered[p] = X[2 * k] * gain
      filtered[p + 1] = X[2 * k + 1] * gain
    }
  })
  if (freqs.length > 0) {
    const W = ifft(fromData(filtered, [freqs.length, size], 'complex128')).data as Float64Array
    for (let row = 0; row < freqs.length; row++)
      for (let i = 0; i < n; i++) {
        const re = W[2 * (row * size + i)]
        const im = W[2 * (row * size + i) + 1]
        outRe[row * n + i] = re
        outIm[row * n + i] = im
        mag[row * n + i] = Math.hypot(re, im)
      }
  }
  return {
    ...timeFrequency({
      t: fromData(
        Float64Array.from({ length: n }, (_, i) => input.t0 + i / fs),
        [n],
      ),
      f: fromData(Float64Array.from(freqs), [freqs.length]),
      values: complexValues(outRe, outIm, [freqs.length, n]),
      quantity: 'complex',
      method: 'cwt',
      frequencyScale: geometric(freqs) ? 'log' : 'linear',
    }),
    magnitude: fromData(mag, [freqs.length, n]),
    scales: fromData(scales, [scales.length]),
  }
}

/**
 * Soft or hard thresholding of coefficients, as `pywt.threshold`: hard keeps $\abs{c} \ge \lambda$ and zeroes the
 * rest; soft also shrinks the survivors towards zero by $\lambda$, $\sgn(c) \max(\abs{c} - \lambda, 0)$.
 *
 * @param values The coefficients $c$; not modified.
 * @param lambda The threshold $\lambda \ge 0$.
 * @param mode `'soft'` (shrink) or `'hard'` (keep or zero).
 * @returns The thresholded coefficients, one per value.
 *
 * @example Soft against hard
 * print('soft:', waveletThreshold([-3, -1, 0.5, 2], 1))
 * print('hard:', waveletThreshold([-3, -1, 0.5, 2], 1, 'hard'))
 */
export function waveletThreshold(values: VectorLike, lambda: Scalar, mode: 'soft' | 'hard' = 'soft'): Tensor {
  const v = dense.toF64(values, 'waveletThreshold')
  const out = Float64Array.from(v, (c) =>
    mode === 'hard' ? (Math.abs(c) >= lambda ? c : 0) : Math.sign(c) * Math.max(Math.abs(c) - lambda, 0),
  )
  return fromData(out, [out.length])
}

/** Options of `waveletDenoise`. */
export type WaveletDenoiseOptions = {
  /** The orthogonal wavelet (default `'db4'`). */
  wavelet?: WaveletName
  /**
   * Decomposition levels. Default: the most, up to 6, for which the length divides by $2^J$ and the coarsest
   * approximation keeps at least 8 coefficients.
   */
  levels?: Size
  /** `'soft'` (default) or `'hard'` thresholding, as `waveletThreshold`. */
  mode?: 'soft' | 'hard'
  /**
   * A fixed threshold, or `'universal'` (default): $\lambda = \hat\sigma \sqrt{2 \ln n}$, with
   * $\hat\sigma = \operatorname{median} \abs{d_1} / 0.6745$ from the finest details $d_1$ and $n$ the signal length.
   */
  threshold?: Scalar | 'universal'
}

/**
 * The denoised signal with the threshold and the noise estimate used: `signal` (on the input's axis), `threshold` (the
 * $\lambda$ applied), `sigma` (the noise estimate $\hat\sigma$, computed even when the threshold is fixed) and
 * `levels` (the number of levels decomposed).
 */
export type WaveletDenoised = { signal: Signal; threshold: Scalar; sigma: Scalar; levels: Size }

/**
 * Wavelet shrinkage (Donoho and Johnstone, 1994, Biometrika 81(3)): decompose with the periodic DWT, threshold every
 * detail level (the approximation is kept), reconstruct. The universal threshold $\hat\sigma \sqrt{2 \ln n}$, with
 * $\hat\sigma$ from the median absolute finest detail, removes white noise with high probability while an orthogonal
 * transform concentrates a smooth or piecewise-smooth signal into a few large coefficients. Throws `ShapeError` when
 * no level can be taken (an odd length, or `levels` 0) or the length is not divisible by $2^J$.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples); the result keeps its `fs` and `t0`.
 * @param options The wavelet, levels, thresholding mode and threshold; see `WaveletDenoiseOptions`.
 * @returns The denoised signal, with the threshold, noise estimate and number of levels used.
 *
 * @example A noisy step
 * const n = 128
 * const clean = Array.from({ length: n }, (_, i) => (i < n / 2 ? 1 : -1))
 * const noisy = add(tensor(clean), normals(stream(1), n, 0, 0.2))
 * const r = waveletDenoise(noisy, { wavelet: 'haar' })
 * print('levels =', r.levels, ' sigma =', r.sigma, ' threshold =', r.threshold)
 * const rms = (x) => Math.sqrt(mean(square(sub(x, tensor(clean)))))
 * print('rms error: noisy =', rms(noisy), ' denoised =', rms(r.signal.data))
 */
export function waveletDenoise(x: SignalInput, options: WaveletDenoiseOptions = {}): WaveletDenoised {
  const { wavelet = 'db4', mode = 'soft' } = options
  const input = readSamples(x, 'waveletDenoise')
  const n = input.values.length
  let levels = options.levels ?? 0
  if (options.levels === undefined) while (levels < 6 && n % 2 ** (levels + 1) === 0 && n >> (levels + 1) >= 8) levels++
  if (levels < 1) throw new ShapeError('waveletDenoise', 'waveletDenoise: the length must be even')
  const d = wavedec(x, wavelet, levels)
  const finest = Float64Array.from(dense.toF64(d.details[0], 'waveletDenoise'), Math.abs).sort()
  const mid = finest.length >> 1
  const median = finest.length % 2 ? finest[mid] : 0.5 * (finest[mid - 1] + finest[mid])
  const sigma = median / 0.6745
  const lambda =
    options.threshold === undefined || options.threshold === 'universal'
      ? sigma * Math.sqrt(2 * Math.log(n))
      : options.threshold
  const shrunk: WaveletDecomposition = { ...d, details: d.details.map((c) => waveletThreshold(c, lambda, mode)) }
  return { signal: waverec(shrunk), threshold: lambda, sigma, levels }
}
