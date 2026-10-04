/**
 * Parametric spectral estimation, part of `aifn-compute/signal/statistical`: the power spectral density of an ARMA model
 * (`armaSpectrum`, the analytic truth of a simulated process), autoregressive spectral estimates by Yule–Walker, Burg
 * or least squares (`arPsd`, with `leastSquaresAr`), and the subspace estimators of line spectra, MUSIC (Schmidt,
 * 1986) and ESPRIT (Roy and Kailath, 1989), with the least-squares amplitudes of sinusoids at given frequencies
 * (`sinusoidFit`). Frequencies are in Hz with `fs` (default 1: cycles per sample); densities are one-sided, as
 * `aifn-compute/signal/spectral`'s estimates, so the two can be overlaid.
 */

import { fromData, imagPart, realPart, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { rfft, rfftfreq } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size, Spectrum, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import { eig, eigh, lstsq } from 'aifn-compute/numerics/linalg'
import { readSamples, spectrum, type SignalInput } from '../signal'
import { burg, yuleWalker, type AutoregressiveFit } from './autoregression'

const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
const valuesOf = (v: VectorLike | Tensor | undefined): number[] =>
  v === undefined ? [] : Array.from(readSamples(v as VectorLike, 'parametric').values)

/** An ARMA(p, q) model x_t = Σ φᵢ x_{t−i} + ε_t + Σ θⱼ ε_{t−j}, ε_t ~ N(0, σ²). */
export type ArmaModel = {
  /** φ₁ … φ_p (none: a moving average). */
  ar?: VectorLike | Tensor
  /** θ₁ … θ_q (none: an autoregression). */
  ma?: VectorLike | Tensor
  /** Innovation variance σ² (default 1). */
  sigma2?: Scalar
}

/**
 * The power spectral density of an ARMA model, one-sided in power per Hz:
 *
 *   S(f) = (2σ²/fs) |1 + Σⱼ θⱼ e^{−iωj}|² / |1 − Σᵢ φᵢ e^{−iωi}|²,  ω = 2πf/fs,
 *
 * halved at f = 0 and f = fs/2, as a one-sided estimate is. It integrates over [0, fs/2] to the process variance
 * when the AR part is stationary (Wiener–Khinchin: the autocovariance of filtered white noise is the filter's
 * squared gain times σ²). `frequencies` (Hz) or `nfft` (the rfft grid, default 512) choose where to evaluate it.
 */
export function armaSpectrum(
  model: ArmaModel,
  { fs = 1, frequencies, nfft = 512 }: { fs?: Scalar; frequencies?: VectorLike | Tensor; nfft?: Size } = {},
): Spectrum {
  const ar = valuesOf(model.ar)
  const ma = valuesOf(model.ma)
  const sigma2 = model.sigma2 ?? 1
  const f = frequencies === undefined ? Array.from(rfftfreq(nfft, 1 / fs).data) : valuesOf(frequencies)
  const out = new Float64Array(f.length)
  for (let k = 0; k < f.length; k++) {
    const omega = (2 * Math.PI * f[k]) / fs
    let aRe = 1
    let aIm = 0
    ar.forEach((p, i) => {
      aRe -= p * Math.cos(omega * (i + 1))
      aIm += p * Math.sin(omega * (i + 1))
    })
    let bRe = 1
    let bIm = 0
    ma.forEach((q, j) => {
      bRe += q * Math.cos(omega * (j + 1))
      bIm -= q * Math.sin(omega * (j + 1))
    })
    const edge = Math.abs(f[k]) < 1e-12 * fs || Math.abs(Math.abs(f[k]) - fs / 2) < 1e-12 * fs
    out[k] = ((edge ? 1 : 2) * sigma2 * (bRe * bRe + bIm * bIm)) / (fs * (aRe * aRe + aIm * aIm))
  }
  return spectrum({ f: vec(f), axis: 'hz', values: fromData(out, [f.length]), quantity: 'psd', sided: 'one', fs })
}

/**
 * The reflection coefficients of an AR polynomial by the step-down (backward Levinson) recursion: k_m = φ_m^{(m)} and
 * φ_j^{(m−1)} = (φ_j^{(m)} + k_m φ_{m−j}^{(m)}) / (1 − k_m²). All |k| < 1 exactly when the AR part is stationary.
 */
function stepDown(phi: readonly number[]): number[] {
  let a = [...phi]
  const k = new Array<number>(a.length).fill(NaN)
  for (let m = a.length; m >= 1; m--) {
    const km = a[m - 1]
    k[m - 1] = km
    if (!(Math.abs(km) < 1)) break
    a = Array.from({ length: m - 1 }, (_, j) => (a[j] + km * a[m - 2 - j]) / (1 - km * km))
  }
  return k
}

/**
 * Least-squares AR(p) estimates: minimise the forward prediction errors Σ_{t≥p} (x_t − Σᵢ φᵢ x_{t−i})² (the
 * covariance method) or, with `forwardBackward` (default), the forward and backward errors together (the modified
 * covariance method; Kay, 1988; Stoica and Moses, 2005, §3.4). Neither pads the series with zeros, so peaks are
 * sharper than Yule–Walker's; stationarity is not guaranteed. σ² is the mean squared residual.
 */
export function leastSquaresAr(
  x: SignalInput,
  order: Size,
  { demean = true, forwardBackward = true }: { demean?: boolean; forwardBackward?: boolean } = {},
): AutoregressiveFit {
  const xs = Array.from(readSamples(x, 'leastSquaresAr').values)
  const n = xs.length
  if (!(Number.isInteger(order) && order >= 1 && order < n / 2))
    throw new DomainError('leastSquaresAr', `leastSquaresAr: order must be an integer in [1, ${Math.ceil(n / 2) - 1}]`)
  let mean = 0
  if (demean) mean = xs.reduce((s, v) => s + v, 0) / n
  const c = xs.map((v) => v - mean)
  const rows: number[] = []
  const rhs: number[] = []
  for (let t = order; t < n; t++) {
    for (let i = 1; i <= order; i++) rows.push(c[t - i])
    rhs.push(c[t])
  }
  if (forwardBackward)
    for (let t = 0; t < n - order; t++) {
      // Backward prediction of x_t from x_{t+1} … x_{t+p}, with the same coefficients.
      for (let i = 1; i <= order; i++) rows.push(c[t + i])
      rhs.push(c[t])
    }
  const m = rhs.length
  const fit = lstsq(fromData(Float64Array.from(rows), [m, order]), vec(rhs))
  const phi = toFlat(fit.x)
  const sigma2 = (fit.residuals.data as Float64Array)[0] / m
  return { ar: vec(phi), sigma2, mean, reflection: vec(stepDown(phi)) }
}

/** An AR spectral estimate: the spectrum of the fitted model, and the fit. */
export type ArPsd = Spectrum & { fit: AutoregressiveFit; method: 'yule-walker' | 'burg' | 'least-squares' }

/**
 * The autoregressive spectral estimate: fit an AR(p) by `method` (`burg`, the default; `yule-walker`;
 * `least-squares`, the modified covariance method) and return the fitted model's PSD (`armaSpectrum`) on the rfft grid
 * of `nfft` (default 512). A parametric estimate is smooth and can resolve peaks closer than fs/n, but its peaks
 * shift and split when p is too large and merge when it is too small.
 */
export function arPsd(
  x: SignalInput,
  order: Size,
  {
    method = 'burg',
    fs,
    nfft = 512,
    demean = true,
  }: { method?: 'yule-walker' | 'burg' | 'least-squares'; fs?: Scalar; nfft?: Size; demean?: boolean } = {},
): ArPsd {
  const input = readSamples(x, 'arPsd', fs)
  const fit =
    method === 'burg'
      ? burg(input.values, order, { demean })
      : method === 'yule-walker'
        ? yuleWalker(input.values, order, { demean })
        : leastSquaresAr(input.values, order, { demean })
  return { ...armaSpectrum({ ar: fit.ar, sigma2: fit.sigma2 }, { fs: input.fs, nfft }), fit, method }
}

// ── Line spectra: MUSIC and ESPRIT ───────────────────────────────────────────────────────────────────────────────────

/** Options of `music` and `esprit`. */
export type SubspaceOptions = {
  /** Number of real sinusoids K; each spans two dimensions (e^{±iωt}) of the signal subspace. */
  sinusoids: Size
  /** Size m of the correlation matrix (default min(⌊n/3⌋, 40), at least 2K + 1). */
  order?: Size
  fs?: Scalar
  /** Subtract the mean first (default true). */
  demean?: boolean
}

/**
 * The m × m forward–backward sample correlation matrix R = ½(R_f + J R_f J) from the snapshots
 * (x_i, …, x_{i+m−1}), i = 0 … n − m, and its eigendecomposition (descending eigenvalues).
 */
function subspace(x: SignalInput, options: SubspaceOptions, where: string) {
  const input = readSamples(x, where, options.fs)
  const v = Array.from(input.values)
  const n = v.length
  const K2 = 2 * options.sinusoids
  const m = options.order ?? Math.max(K2 + 1, Math.min(Math.floor(n / 3), 40))
  if (!(Number.isInteger(options.sinusoids) && options.sinusoids >= 1))
    throw new DomainError(where, `${where}: sinusoids must be a positive integer`)
  if (!(m > K2 && m <= n)) throw new DomainError(where, `${where}: order must be in (2K, n], got ${m}`)
  if (options.demean ?? true) {
    const mean = v.reduce((s, a) => s + a, 0) / n
    for (let i = 0; i < n; i++) v[i] -= mean
  }
  const L = n - m + 1
  const R = new Float64Array(m * m)
  for (let i = 0; i < L; i++)
    for (let a = 0; a < m; a++) for (let b = a; b < m; b++) R[a * m + b] += v[i + a] * v[i + b]
  for (let a = 0; a < m; a++) for (let b = 0; b < a; b++) R[a * m + b] = R[b * m + a]
  const Rfb = new Float64Array(m * m)
  for (let a = 0; a < m; a++)
    for (let b = 0; b < m; b++) Rfb[a * m + b] = (R[a * m + b] + R[(m - 1 - a) * m + (m - 1 - b)]) / (2 * L)
  const e = eigh(fromData(Rfb, [m, m]))
  return { fs: input.fs, m, K2, values: e.values.data as Float64Array, vectors: e.vectors.data as Float64Array, v }
}

/** The K largest local maxima of y over the grid f, each refined by a parabola through its neighbours. */
function topPeaks(f: ArrayLike<number>, y: ArrayLike<number>, K: number): number[] {
  const peaks: { i: number; v: number }[] = []
  for (let i = 1; i < y.length - 1; i++) if (y[i] >= y[i - 1] && y[i] > y[i + 1]) peaks.push({ i, v: y[i] })
  peaks.sort((a, b) => b.v - a.v)
  const df = f[1] - f[0]
  return peaks
    .slice(0, K)
    .map(({ i }) => {
      const [a, b, c] = [Math.log(y[i - 1]), Math.log(y[i]), Math.log(y[i + 1])]
      const den = a - 2 * b + c
      const shift = den !== 0 ? (0.5 * (a - c)) / den : 0
      return f[i] + Math.max(-0.5, Math.min(0.5, shift)) * df
    })
    .sort((a, b) => a - b)
}

/** A subspace estimate of a line spectrum. */
export type LineSpectrum = {
  /** The estimated frequencies (Hz), ascending. */
  frequencies: Tensor
  /** The least-squares power of each sinusoid, A²/2 (`sinusoidFit`). */
  powers: Tensor
  /** The correlation matrix's eigenvalues, descending: K pairs above a noise floor when the model fits. */
  eigenvalues: Tensor
  /** The noise variance estimate: the mean of the m − 2K smallest eigenvalues. */
  noiseVariance: Scalar
  order: Size
}

/**
 * MUSIC (multiple signal classification; Schmidt, 1986): the eigenvectors of the forward–backward correlation matrix
 * split into a signal subspace (the 2K largest eigenvalues, spanned by the steering vectors e(ω) = (1, e^{iω}, …) of
 * the sinusoids) and a noise subspace orthogonal to it. The pseudospectrum P(f) = 1 / Σ_noise |e(ω)ᴴ v|² is large
 * where e(ω) is nearly orthogonal to the noise subspace, so its K highest peaks estimate the frequencies, with
 * resolution beyond fs/n at high SNR. P is not a power density: its height says nothing about power (`powers` comes
 * from a least-squares fit at the estimated frequencies). Evaluated on the rfft grid of `nfft` (default 2048).
 */
export function music(x: SignalInput, options: SubspaceOptions & { nfft?: Size }): Spectrum & LineSpectrum {
  const s = subspace(x, options, 'music')
  const nfft = Math.max(options.nfft ?? 2048, s.m)
  const noise = s.m - s.K2
  // The noise eigenvectors (columns 2K … m − 1) as rows, transformed at once: |e(ω)ᴴ v|² = |DFT(v)(ω)|².
  const rows = new Float64Array(noise * s.m)
  for (let j = 0; j < noise; j++) for (let l = 0; l < s.m; l++) rows[j * s.m + l] = s.vectors[l * s.m + s.K2 + j]
  const spec = rfft(fromData(rows, [noise, s.m]), { n: nfft }).data as Float64Array
  const nfreq = Math.floor(nfft / 2) + 1
  const P = new Float64Array(nfreq)
  for (let b = 0; b < nfreq; b++) {
    let d = 0
    for (let j = 0; j < noise; j++) d += spec[2 * (j * nfreq + b)] ** 2 + spec[2 * (j * nfreq + b) + 1] ** 2
    P[b] = 1 / Math.max(d, 1e-300)
  }
  const f = rfftfreq(nfft, 1 / s.fs).data as Float64Array
  const frequencies = topPeaks(f, P, options.sinusoids)
  return {
    ...spectrum({
      f: fromData(f, [nfreq]),
      axis: 'hz',
      values: fromData(P, [nfreq]),
      quantity: 'power',
      sided: 'one',
      fs: s.fs,
    }),
    ...lineSummary(s, frequencies),
  }
}

function lineSummary(s: ReturnType<typeof subspace>, frequencies: number[]): LineSpectrum {
  let floor = 0
  for (let j = s.K2; j < s.m; j++) floor += s.values[j]
  const fit = sinusoidFit(s.v, frequencies, { fs: s.fs })
  return {
    frequencies: vec(frequencies),
    powers: fit.powers,
    eigenvalues: fromData(Float64Array.from(s.values), [s.m]),
    noiseVariance: floor / (s.m - s.K2),
    order: s.m,
  }
}

/**
 * ESPRIT (estimation of signal parameters via rotational invariance; Roy and Kailath, 1989), least-squares form: the
 * signal subspace U_s (the 2K leading eigenvectors of the forward–backward correlation matrix) is shift-invariant,
 * U_s[1:] = U_s[:−1] Φ, and the eigenvalues of Φ are e^{±iωₖ}. Φ is the least-squares solution; the frequencies are
 * the positive angles of its eigenvalues times fs/2π. No search over a grid, so the estimate is not quantised.
 */
export function esprit(x: SignalInput, options: SubspaceOptions): LineSpectrum {
  const s = subspace(x, options, 'esprit')
  const { m, K2 } = s
  const U1 = new Float64Array((m - 1) * K2)
  const U2 = new Float64Array((m - 1) * K2)
  for (let r = 0; r < m - 1; r++)
    for (let c = 0; c < K2; c++) {
      U1[r * K2 + c] = s.vectors[r * m + c]
      U2[r * K2 + c] = s.vectors[(r + 1) * m + c]
    }
  const Phi = lstsq(fromData(U1, [m - 1, K2]), fromData(U2, [m - 1, K2])).x
  const z = eig(Phi, { vectors: false }).values
  const re = toFlat(realPart(z))
  const im = toFlat(imagPart(z))
  const angles = re.map((r, i) => Math.atan2(im[i], r))
  // Conjugate pairs give ±ω; a real eigenvalue (a tone at DC or Nyquist) gives one angle. Keep the K largest |ω| ≥ 0
  // after taking one of each pair.
  const positive = angles.filter((a, i) => im[i] > 0 || (im[i] === 0 && a >= 0)).map((a) => Math.abs(a))
  const pool = positive.length >= options.sinusoids ? positive : angles.map(Math.abs)
  const frequencies = [...new Set(pool.map((a) => (a * s.fs) / (2 * Math.PI)))]
    .slice(0, options.sinusoids)
    .sort((a, b) => a - b)
  return lineSummary(s, frequencies)
}

/** Sinusoids fitted at known frequencies. */
export type SinusoidFit = {
  /** Amplitude Aₖ, phase φₖ (of Aₖ sin(2πfₖt/fs + φₖ)) and power Aₖ²/2 of each frequency. */
  amplitudes: Tensor
  phases: Tensor
  powers: Tensor
  /** The fitted constant. */
  offset: Scalar
  /** Mean squared residual. */
  residualVariance: Scalar
}

/**
 * The least-squares fit x_t ≈ c + Σₖ (aₖ cos 2πfₖt/fs + bₖ sin 2πfₖt/fs) at given frequencies (Stoica and Moses,
 * 2005, §4.3): with the frequencies fixed the model is linear. Aₖ = √(aₖ² + bₖ²), φₖ = atan2(aₖ, bₖ), and the power
 * of each line in a one-sided spectrum is Aₖ²/2.
 */
export function sinusoidFit(
  x: SignalInput,
  frequencies: VectorLike | Tensor,
  { fs }: { fs?: Scalar } = {},
): SinusoidFit {
  const input = readSamples(x, 'sinusoidFit', fs)
  const v = input.values
  const n = v.length
  const f = valuesOf(frequencies)
  const p = 1 + 2 * f.length
  const A = new Float64Array(n * p)
  for (let t = 0; t < n; t++) {
    A[t * p] = 1
    f.forEach((fk, k) => {
      const w = (2 * Math.PI * fk * t) / input.fs
      A[t * p + 1 + 2 * k] = Math.cos(w)
      A[t * p + 2 + 2 * k] = Math.sin(w)
    })
  }
  const fit = lstsq(fromData(A, [n, p]), fromData(Float64Array.from(v), [n]))
  const c = toFlat(fit.x)
  const amplitudes = f.map((_, k) => Math.hypot(c[1 + 2 * k], c[2 + 2 * k]))
  return {
    amplitudes: vec(amplitudes),
    phases: vec(f.map((_, k) => Math.atan2(c[1 + 2 * k], c[2 + 2 * k]))),
    powers: vec(amplitudes.map((a) => (a * a) / 2)),
    offset: c[0],
    residualVariance: ((fit.residuals.data as Float64Array)[0] ?? NaN) / n,
  }
}
