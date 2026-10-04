/**
 * Nonparametric spectral estimates beyond the periodogram and Welch: Bartlett's average of non-overlapping
 * periodograms (Bartlett, 1946), the Blackman–Tukey lag-window estimate, the cross-spectral density and the
 * magnitude-squared coherence (Carter, Knapp and Nuttall, 1973) as `scipy.signal.csd` and `scipy.signal.coherence`,
 * χ² confidence intervals for an estimate with ν equivalent degrees of freedom, the coherence's null threshold, and
 * the log-spectral error of an estimate against a reference spectrum.
 *
 * Conventions follow `welch`: one-sided densities (doubled off DC and Nyquist) in power per Hz with `fs`.
 */

import { dense, fromData, isTensor, type Tensor } from 'aifn-compute/foundation/tensor'
import { rfft, rfftfreq } from 'aifn-compute/foundation/fourier'
import type { Scalar, Size, Spectrum } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { regularisedGammaPInverse } from 'aifn-compute/numerics/special'
import { autocovariance } from 'aifn-compute/probability/stats'
import { windowValues, type WindowInput } from 'aifn-compute/signal/windows'
import { complexValues, readSamples, spectrum, type SignalInput } from '../signal'
import {
  powerSpectrum,
  resolveSegments,
  segmentPowers,
  segmentTransforms,
  welch,
  type Detrend,
  type SegmentOptions,
} from './spectral'

/**
 * Bartlett's method: the mean of rectangular-window periodograms of non-overlapping segments of length `nperseg`
 * (Bartlett, 1946). It is `welch` with a boxcar window and no overlap: averaging K segments divides the variance by K
 * and widens the resolution from fs/n to fs/nperseg. Default segment length: n/8 (at least 16 samples).
 */
export function bartlett(
  x: SignalInput,
  options: Omit<SegmentOptions, 'noverlap'> = {},
): Spectrum & { segments: Size; dof: number } {
  const n = readSamples(x, 'bartlett').values.length
  return welch(x, {
    window: 'boxcar',
    ...options,
    nperseg: options.nperseg ?? Math.min(n, Math.max(16, Math.floor(n / 8))),
    noverlap: 0,
  })
}

/** Options of `blackmanTukey`. */
export type BlackmanTukeyOptions = {
  fs?: Scalar
  /** The largest lag M used (default ⌊n/10⌋, at least 1): resolution ≈ fs/M. */
  maxLag?: Size
  /**
   * The lag window, any window spec, built symmetric of length 2M + 1 and centred on lag 0 (default `bartlett`, the
   * triangle 1 − |k|/M, whose estimate is never negative). Other windows can give negative values where the
   * spectrum is small.
   */
  lagWindow?: WindowInput
  /** FFT length (≥ 2M + 1; default the larger of n and 2M + 1). */
  nfft?: Size
  /** Subtract the mean first (default true). */
  detrend?: boolean
}

/**
 * The Blackman–Tukey (lag-window, correlogram) estimate: the Fourier transform of the biased sample autocovariance
 * γ̂(k) (÷ n) tapered by a lag window w of half-width M,
 *
 *   S(f) = (1/fs) Σ_{|k|≤M} w(k) γ̂(k) e^{−2πi fk/fs},
 *
 * one-sided (doubled off DC and Nyquist). With w = 1 and M = n − 1 it is exactly the periodogram (Wiener–Khinchin
 * for the sample); a short window smooths the periodogram by convolving it with the window's transform. Equivalent
 * degrees of freedom ν = 2n / Σ_{|k|≤M} w(k)² (Percival and Walden, 1993, eq. 248a).
 */
export function blackmanTukey(
  x: SignalInput,
  options: BlackmanTukeyOptions = {},
): Spectrum & { dof: number; maxLag: Size } {
  const input = readSamples(x, 'blackmanTukey', options.fs)
  const v = input.values
  const n = v.length
  const M = options.maxLag ?? Math.max(1, Math.floor(n / 10))
  if (!(Number.isInteger(M) && M >= 0 && M < n))
    throw new DomainError('blackmanTukey', `blackmanTukey: maxLag must be an integer in [0, ${n - 1}], got ${M}`)
  const nfft = Math.max(options.nfft ?? n, 2 * M + 1)
  const g = autocovariance(v, { maxLag: M, demean: options.detrend ?? true }).data as Float64Array
  const w = windowValues(options.lagWindow ?? 'bartlett', 2 * M + 1, false)
  // A real, even sequence c[k] = c[nfft − k] = w(k)γ̂(k); its DFT is real.
  const c = new Float64Array(nfft)
  let w2 = 0
  for (let k = 0; k <= M; k++) {
    const wk = w[M + k]
    c[k] = wk * g[k]
    if (k > 0) c[nfft - k] = wk * g[k]
    w2 += (k === 0 ? 1 : 2) * wk * wk
  }
  const spec = rfft(fromData(c), { n: nfft }).data as Float64Array
  const fs = input.fs
  const nfreq = Math.floor(nfft / 2) + 1
  const psd = new Float64Array(nfreq)
  for (let b = 0; b < nfreq; b++) {
    const edge = b === 0 || (nfft % 2 === 0 && b === nfft / 2)
    psd[b] = ((edge ? 1 : 2) * spec[2 * b]) / fs
  }
  const f = rfftfreq(nfft, 1 / fs).data
  return {
    ...powerSpectrum(f, psd, { fs, scaling: 'density', onesided: true }, input.unit),
    dof: (2 * n) / w2,
    maxLag: M,
  }
}

const SEGMENT_DEFAULTS = {
  window: 'hann' as WindowInput,
  nperseg: 256,
  overlap: (n: number) => Math.floor(n / 2),
  detrend: 'constant' as Detrend,
}

/** The segment transforms of two equally long signals under the same options. */
function pairedSegments(x: SignalInput, y: SignalInput, options: SegmentOptions, where: string) {
  const a = readSamples(x, where, options.fs)
  const b = readSamples(y, where, options.fs ?? a.fs)
  if (a.values.length !== b.values.length)
    throw new ShapeError(where, `${where}: x and y differ in length (${a.values.length} and ${b.values.length})`)
  const o = resolveSegments(a.values, a.fs, options, SEGMENT_DEFAULTS)
  return { o, sx: segmentTransforms(a.values, o), sy: segmentTransforms(b.values, o) }
}

/**
 * The cross-spectral density of x and y by Welch's method, as `scipy.signal.csd`: the mean over segments of
 * conj(Xₖ(f)) Yₖ(f), scaled as `welch` scales a PSD and one-sided (doubled off DC and Nyquist). For y = h ∗ x + noise
 * independent of x, P_xy(f) = H(f) P_xx(f): its phase is the phase of the filter (−2πfτ for a delay τ). Returns a
 * complex128 `Spectrum` (`quantity: 'complex'`); `segments` is the number averaged.
 */
export function csd(x: SignalInput, y: SignalInput, options: SegmentOptions = {}): Spectrum & { segments: Size } {
  const { o, sx, sy } = pairedSegments(x, y, options, 'csd')
  const K = sx.re.length
  const nfreq = sx.nfreq
  const re = new Float64Array(nfreq)
  const im = new Float64Array(nfreq)
  for (let k = 0; k < K; k++)
    for (let j = 0; j < nfreq; j++) {
      // conj(X) Y
      re[j] += sx.re[k][j] * sy.re[k][j] + sx.im[k][j] * sy.im[k][j]
      im[j] += sx.re[k][j] * sy.im[k][j] - sx.im[k][j] * sy.re[k][j]
    }
  for (let j = 0; j < nfreq; j++) {
    const nyquist = o.nfft % 2 === 0 && j === o.nfft / 2
    const scale = ((o.onesided && j > 0 && !nyquist ? 2 : 1) * sx.scale) / Math.max(K, 1)
    re[j] *= scale
    im[j] *= scale
  }
  return {
    ...spectrum({
      f: fromData(sx.freqs, [nfreq]),
      axis: 'hz',
      values: complexValues(re, im, [nfreq]),
      quantity: 'complex',
      sided: o.onesided ? 'one' : 'two',
      fs: o.fs,
    }),
    segments: K,
  }
}

/**
 * The magnitude-squared coherence C_xy(f) = |P_xy(f)|² / (P_xx(f) P_yy(f)), each density estimated by Welch's method
 * with the same segments, as `scipy.signal.coherence`. C lies in [0, 1]; for y = h ∗ x + v with v independent of x it
 * is the fraction of y's power at f explained linearly by x, |H|²P_xx / (|H|²P_xx + P_vv). With one segment it is 1
 * identically, so coherence needs averaging; with K independent segments, its estimate is biased upwards by about
 * (1 − C)²/K (Carter, Knapp and Nuttall, 1973). `segments` is K; `coherenceThreshold(K)` is the level an
 * uncoupled pair exceeds with a given probability.
 */
export function coherence(x: SignalInput, y: SignalInput, options: SegmentOptions = {}): Spectrum & { segments: Size } {
  const { o, sx, sy } = pairedSegments(x, y, options, 'coherence')
  const K = sx.re.length
  const pxx = segmentPowers(sx, o.nfft, o.onesided)
  const pyy = segmentPowers(sy, o.nfft, o.onesided)
  const nfreq = sx.nfreq
  const sxx = new Float64Array(nfreq)
  const syy = new Float64Array(nfreq)
  const re = new Float64Array(nfreq)
  const im = new Float64Array(nfreq)
  for (let k = 0; k < K; k++)
    for (let j = 0; j < nfreq; j++) {
      sxx[j] += pxx[k][j]
      syy[j] += pyy[k][j]
      re[j] += sx.re[k][j] * sy.re[k][j] + sx.im[k][j] * sy.im[k][j]
      im[j] += sx.re[k][j] * sy.im[k][j] - sx.im[k][j] * sy.re[k][j]
    }
  const c = new Float64Array(nfreq)
  for (let j = 0; j < nfreq; j++) {
    // The one-sided factor and the 1/K cancel in the ratio; only the raw cross-power and powers matter.
    const nyquist = o.nfft % 2 === 0 && j === o.nfft / 2
    const two = o.onesided && j > 0 && !nyquist ? 2 : 1
    const cross = (re[j] * re[j] + im[j] * im[j]) * (two * sx.scale) ** 2
    const den = sxx[j] * syy[j]
    c[j] = den > 0 ? cross / den : 0
  }
  return {
    ...spectrum({
      f: fromData(sx.freqs, [nfreq]),
      axis: 'hz',
      values: fromData(c, [nfreq]),
      quantity: 'coherence',
      sided: o.onesided ? 'one' : 'two',
      fs: o.fs,
    }),
    segments: K,
  }
}

/**
 * The level that the estimated coherence of two independent signals exceeds with probability 1 − `level`, from K
 * independent segments: under independence P(Ĉ ≥ c) = (1 − c)^{K−1} (Carter, Knapp and Nuttall, 1973), so
 * c = 1 − (1 − level)^{1/(K−1)}. Overlapping segments are not independent; pass their effective number.
 */
export function coherenceThreshold(segments: Scalar, { level = 0.95 }: { level?: Scalar } = {}): number {
  if (!(segments > 1)) return 1
  return 1 - Math.pow(1 - level, 1 / (segments - 1))
}

/** A two-sided χ² interval for a spectrum. */
export type SpectralInterval = {
  /** Lower and upper limits, the shape of the spectrum's values. */
  lower: Tensor
  upper: Tensor
  /** The degrees of freedom used, per frequency. */
  dof: Tensor
  level: Scalar
}

/** The p-quantile of χ²_ν: 2 P⁻¹(ν/2, p), through the inverse regularised incomplete gamma function. */
export function chiSquareQuantile(nu: Scalar, p: Scalar): number {
  return 2 * (regularisedGammaPInverse(nu / 2, p) as number)
}

/**
 * The 100·level % confidence interval of a power spectral estimate Ŝ(f) with ν equivalent degrees of freedom:
 * νŜ/S ~ χ²_ν, so [νŜ / χ²_ν(1 − α/2), νŜ / χ²_ν(α/2)] with α = 1 − level (Percival and Walden, 1993, §6.10;
 * Welch, 1967). On a dB scale the interval has the same width at every frequency. `dof` is a number (a periodogram's
 * 2, Welch's `dof`, Blackman–Tukey's `dof`) or one per frequency (adaptive multitaper). The χ² law holds away from DC
 * and Nyquist and for a spectrum smooth over the estimator's bandwidth; it fails at spectral lines.
 */
export function spectralConfidence(
  s: Spectrum,
  dof: Scalar | Tensor | ArrayLike<number>,
  { level = 0.95 }: { level?: Scalar } = {},
): SpectralInterval {
  const values = s.values.data as Float64Array
  const m = values.length
  const nu =
    typeof dof === 'number' ? new Float64Array(m).fill(dof) : Float64Array.from(isTensor(dof) ? dense.data(dof) : dof)
  if (nu.length !== m) throw new ShapeError('spectralConfidence', 'spectralConfidence: one dof per frequency')
  const alpha = 1 - level
  const lower = new Float64Array(m)
  const upper = new Float64Array(m)
  const cache = new Map<number, [number, number]>()
  for (let i = 0; i < m; i++) {
    let q = cache.get(nu[i])
    if (!q) {
      q = [chiSquareQuantile(nu[i], 1 - alpha / 2), chiSquareQuantile(nu[i], alpha / 2)]
      cache.set(nu[i], q)
    }
    lower[i] = (nu[i] * values[i]) / q[0]
    upper[i] = (nu[i] * values[i]) / q[1]
  }
  return { lower: fromData(lower, [m]), upper: fromData(upper, [m]), dof: fromData(nu, [m]), level }
}

/** Which frequencies a spectral error counts: inside `band`, outside every `exclude` interval (e.g. around lines). */
export type SpectralErrorOptions = {
  band?: readonly [number, number]
  exclude?: readonly (readonly [number, number])[]
  /** The frequencies, when the estimate is given as bare values. */
  f?: ArrayLike<number>
}

const counted = (freq: number | undefined, o: SpectralErrorOptions) =>
  freq === undefined ||
  ((!o.band || (freq >= o.band[0] && freq <= o.band[1])) && !(o.exclude ?? []).some(([a, b]) => freq >= a && freq <= b))

/**
 * The error of a spectral estimate against a reference on the same frequencies, in dB: dᵢ = 10 log₁₀(Ŝᵢ / Sᵢ).
 * `bias` is the mean of d, `sd` its standard deviation and `distance` the log-spectral distance √(mean d²), so
 * distance² = bias² + sd². Frequencies where either spectrum is not positive, outside `band` or inside an `exclude`
 * interval are left out.
 */
export function logSpectralError(
  estimate: Spectrum | ArrayLike<number>,
  reference: ArrayLike<number>,
  { band, exclude, f }: SpectralErrorOptions = {},
): { bias: number; sd: number; distance: number; count: Size } {
  const est = 'kind' in estimate ? (estimate.values.data as Float64Array) : (estimate as ArrayLike<number>)
  const freqs = f ?? ('kind' in estimate ? (estimate.f.data as Float64Array) : undefined)
  if (est.length !== reference.length)
    throw new ShapeError('logSpectralError', 'logSpectralError: estimate and reference differ in length')
  let sum = 0
  let sum2 = 0
  let count = 0
  for (let i = 0; i < est.length; i++) {
    if (!(est[i] > 0 && reference[i] > 0)) continue
    if (!counted(freqs?.[i], { band, exclude })) continue
    const d = 10 * Math.log10(est[i] / reference[i])
    sum += d
    sum2 += d * d
    count++
  }
  if (count === 0) return { bias: NaN, sd: NaN, distance: NaN, count }
  const bias = sum / count
  const ms = sum2 / count
  return { bias, sd: Math.sqrt(Math.max(0, ms - bias * bias)), distance: Math.sqrt(ms), count }
}

/**
 * The bias and variance of a spectral estimator, from R estimates of independent realisations on one frequency grid,
 * in dB: at each frequency the mean of 10 log₁₀ Ŝ_r against 10 log₁₀ S (squared: bias²) and the variance of
 * 10 log₁₀ Ŝ_r about that mean, each averaged over the counted frequencies; mse = bias² + variance. Smoothing more
 * (shorter segments, a shorter lag window, more tapers, a lower AR order) trades variance for bias. On the log scale
 * even a consistent estimator is biased by E log(χ²_ν/ν) (−2.5 dB for a periodogram's ν = 2).
 */
export function replicateSpectralError(
  estimates: readonly ArrayLike<number>[],
  reference: ArrayLike<number>,
  options: SpectralErrorOptions = {},
): { bias2: number; variance: number; mse: number; count: Size } {
  const R = estimates.length
  const m = reference.length
  let bias2 = 0
  let variance = 0
  let count = 0
  for (let i = 0; i < m; i++) {
    if (!counted(options.f?.[i], options) || !(reference[i] > 0)) continue
    const d = estimates.map((e) => 10 * Math.log10(Math.max(e[i], 1e-300)))
    const mean = d.reduce((a, b) => a + b, 0) / R
    bias2 += (mean - 10 * Math.log10(reference[i])) ** 2
    variance += d.reduce((a, b) => a + (b - mean) ** 2, 0) / R
    count++
  }
  if (count === 0) return { bias2: NaN, variance: NaN, mse: NaN, count }
  return { bias2: bias2 / count, variance: variance / count, mse: (bias2 + variance) / count, count }
}

/**
 * Whether a spectral estimate shows two peaks near f₁ < f₂ as separate: the highest value within a quarter of the
 * separation of each, and the lowest value between them; `dip` is the weaker peak's height above that valley in dB,
 * and the peaks count as resolved when the dip is at least `threshold` (default 3 dB, the Rayleigh-like criterion).
 */
export function peakDip(
  s: Spectrum,
  f1: Scalar,
  f2: Scalar,
  { threshold = 3 }: { threshold?: Scalar } = {},
): { dip: number; resolved: boolean } {
  const f = dense.data(s.f)
  const v = dense.data(s.values)
  const [lo, hi] = f1 < f2 ? [f1, f2] : [f2, f1]
  const q = (hi - lo) / 4
  let p1 = -Infinity
  let p2 = -Infinity
  let valley = Infinity
  for (let i = 0; i < f.length; i++) {
    const db = 10 * Math.log10(Math.max(v[i], 1e-300))
    if (Math.abs(f[i] - lo) <= q) p1 = Math.max(p1, db)
    if (Math.abs(f[i] - hi) <= q) p2 = Math.max(p2, db)
    if (f[i] > lo && f[i] < hi) valley = Math.min(valley, db)
  }
  if (!Number.isFinite(p1) || !Number.isFinite(p2) || !Number.isFinite(valley)) return { dip: 0, resolved: false }
  const dip = Math.min(p1, p2) - valley
  return { dip, resolved: dip >= threshold }
}

/**
 * The delay τ of y behind x from the phase of their cross-spectral density: for y = x(t − τ) filtered and with noise,
 * arg S_xy(f) = −2πfτ, so τ is the slope of the unwrapped phase through the origin, fitted by least squares weighted
 * by C/(1 − C) (the inverse phase variance, up to a constant) over the bins with coherence at least `minCoherence`
 * (default 0.5), unwrapped in order of frequency. In the time units of the spectra (samples when fs = 1).
 */
export function crossSpectralDelay(
  cross: Spectrum,
  coh: Spectrum,
  { minCoherence = 0.5 }: { minCoherence?: Scalar } = {},
): { delay: number; bins: Size } {
  const f = dense.data(cross.f)
  const z = cross.values.data as Float64Array
  const c = dense.data(coh.values)
  let num = 0
  let den = 0
  let bins = 0
  let previous: number | undefined
  let offset = 0
  for (let i = 1; i < f.length; i++) {
    if (!(c[i] >= minCoherence) || c[i] >= 1) continue
    let phase = Math.atan2(z[2 * i + 1], z[2 * i]) + offset
    if (previous !== undefined) {
      while (phase - previous > Math.PI) {
        phase -= 2 * Math.PI
        offset -= 2 * Math.PI
      }
      while (phase - previous < -Math.PI) {
        phase += 2 * Math.PI
        offset += 2 * Math.PI
      }
    }
    previous = phase
    const w = c[i] / (1 - c[i])
    num += w * f[i] * phase
    den += w * f[i] * f[i]
    bins++
  }
  return { delay: den > 0 ? -num / (2 * Math.PI * den) : NaN, bins }
}
