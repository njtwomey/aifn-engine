/**
 * Nonparametric spectral estimates beyond the periodogram and Welch: Bartlett's average of non-overlapping
 * periodograms (Bartlett, 1946), the Blackman–Tukey lag-window estimate, the cross-spectral density and the
 * magnitude-squared coherence (Carter, Knapp and Nuttall, 1973) as `scipy.signal.csd` and `scipy.signal.coherence`,
 * $\chi^2$ confidence intervals for an estimate with $\nu$ equivalent degrees of freedom, the coherence's null
 * threshold, the delay read from a cross-spectrum's phase, and the log-spectral error of an estimate against a
 * reference spectrum, with the resolution test `peakDip`.
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
 * (Bartlett, 1946). It is `welch` with a boxcar window and no overlap: averaging $K$ segments divides the variance by
 * $K$ and widens the resolution from $f_s/n$ to $f_s/\text{nperseg}$. Default segment length: $\lfloor n/8 \rfloor$,
 * at least 16 samples and at most $n$.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The `SegmentOptions` other than `noverlap` (always 0); `window` defaults to `'boxcar'`.
 * @returns The estimate as a `Spectrum`, with `segments` and `dof` ($2K$ for the boxcar).
 *
 * @example Eight segments of white noise
 * const B = bartlett(normals(stream(1), 1024))
 * print('segments =', B.segments, ' dof =', B.dof, ' bins =', B.f.shape[0])
 * print('mean density =', mean(B.values), ' (white noise of unit variance: 2)')
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
  /** The sample rate in Hz (default the signal's, or 1 for bare samples). */
  fs?: Scalar
  /** The largest lag $M$ used (default $\lfloor n/10 \rfloor$, at least 1): resolution $\approx f_s/M$. */
  maxLag?: Size
  /**
   * The lag window, any window spec, built symmetric of length $2M + 1$ and centred on lag 0 (default `bartlett`, the
   * triangle $1 - \abs{k}/M$, whose estimate is never negative). Other windows can give negative values where the
   * spectrum is small.
   */
  lagWindow?: WindowInput
  /** FFT length ($\ge 2M + 1$; default the larger of $n$ and $2M + 1$). */
  nfft?: Size
  /** Subtract the mean first (default true). */
  detrend?: boolean
}

/**
 * The Blackman–Tukey (lag-window, correlogram) estimate: the Fourier transform of the biased sample autocovariance
 * $\hat\gamma(k)$ (divided by $n$) tapered by a lag window $w$ of half-width $M$,
 * $S(f) = \frac{1}{f_s} \sum_{\abs{k} \le M} w(k)\, \hat\gamma(k)\, e^{-2\pi i f k / f_s}$, one-sided (doubled off
 * DC and Nyquist). With $w = 1$ and $M = n - 1$ it is exactly the periodogram (Wiener–Khinchin for the sample); a
 * short window smooths the periodogram by convolving it with the window's transform. Equivalent degrees of freedom
 * $\nu = 2n / \sum_{\abs{k} \le M} w(k)^2$ (Percival and Walden, 1993, eq. 248a). Throws `DomainError` unless $M$
 * is an integer in $[0, n - 1]$.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The sample rate, largest lag, lag window, FFT length and mean removal; see `BlackmanTukeyOptions`.
 * @returns The one-sided density as a `Spectrum`, with `dof` and the `maxLag` $M$ used.
 *
 * @example A smoothed estimate of white noise
 * // M = 102 lags of 1024 samples: about 30 degrees of freedom, against the periodogram's 2.
 * const T = blackmanTukey(normals(stream(1), 1024))
 * print('maxLag =', T.maxLag, ' dof =', T.dof)
 * print('mean density =', mean(T.values), ' min =', min(T.values), ' max =', max(T.values))
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

/** The segment defaults of `csd` and `coherence`, as `welch`'s: Hann, 256 samples, half overlap, constant detrend. */
const SEGMENT_DEFAULTS = {
  window: 'hann' as WindowInput,
  nperseg: 256,
  overlap: (n: number) => Math.floor(n / 2),
  detrend: 'constant' as Detrend,
}

/**
 * The segment transforms of two equally long signals under the same options. Throws `ShapeError` when the lengths
 * differ.
 *
 * @param x The first signal (a `Signal`, or bare samples); its sample rate is used unless `options.fs` is given.
 * @param y The second signal, of the same length.
 * @param options The `SegmentOptions`, resolved over `welch`'s defaults.
 * @param where The caller's name, for error messages.
 * @returns The resolved options and the segment transforms `sx` and `sy`.
 */
function pairedSegments(x: SignalInput, y: SignalInput, options: SegmentOptions, where: string) {
  const a = readSamples(x, where, options.fs)
  const b = readSamples(y, where, options.fs ?? a.fs)
  if (a.values.length !== b.values.length)
    throw new ShapeError(where, `${where}: x and y differ in length (${a.values.length} and ${b.values.length})`)
  const o = resolveSegments(a.values, a.fs, options, SEGMENT_DEFAULTS)
  return { o, sx: segmentTransforms(a.values, o), sy: segmentTransforms(b.values, o) }
}

/**
 * The cross-spectral density of $x$ and $y$ by Welch's method, as `scipy.signal.csd`: the mean over segments of
 * $X_k^*(f)\, Y_k(f)$, scaled as `welch` scales a PSD and one-sided (doubled off DC and Nyquist). For
 * $y = h * x + \text{noise}$ independent of $x$, $P_{xy}(f) = H(f) P_{xx}(f)$: its phase is the phase of the filter
 * ($-2\pi f \tau$ for a delay $\tau$). Returns a complex128 `Spectrum` (`quantity: 'complex'`); `segments` is the
 * number averaged.
 *
 * @param x The first signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param y The second signal, of the same length (`ShapeError` otherwise).
 * @param options The `SegmentOptions`, with `welch`'s defaults.
 * @returns The cross-spectral density, with the number of segments.
 *
 * @example A delayed copy
 * // y is x delayed by 3 samples plus noise: |P_xy| matches the density of x, about 2, and its phase is -2 pi f 3.
 * const x = normals(stream(1), 4096)
 * const xs = x.data
 * const y = add(tensor(Array.from(xs, (_, i) => (i >= 3 ? xs[i - 3] : 0))), normals(stream(2), 4096, 0, 0.5))
 * const P = csd(x, y, { nperseg: 128 })
 * print('segments =', P.segments, ' mean |P_xy| =', mean(complexAbs(P.values)))
 * const phase = angle(P.values).data
 * print('phase at bins 8, 16 =', phase[8], phase[16])
 * print('-2 pi f 3 there =', -6 * Math.PI * P.f.data[8], -6 * Math.PI * P.f.data[16])
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
 * The magnitude-squared coherence $C_{xy}(f) = \abs{P_{xy}(f)}^2 / (P_{xx}(f) P_{yy}(f))$, each density estimated by
 * Welch's method with the same segments, as `scipy.signal.coherence`. $C$ lies in $[0, 1]$; for $y = h * x + v$ with
 * $v$ independent of $x$ it is the fraction of $y$'s power at $f$ explained linearly by $x$,
 * $\abs{H}^2 P_{xx} / (\abs{H}^2 P_{xx} + P_{vv})$. With one segment it is 1 identically, so coherence needs
 * averaging; with $K$ independent segments, its estimate is biased upwards by about $(1 - C)^2/K$ (Carter, Knapp and
 * Nuttall, 1973). `segments` is $K$; `coherenceThreshold(K)` is the level an uncoupled pair exceeds with a given
 * probability. Bins where either density is 0 get 0.
 *
 * @param x The first signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param y The second signal, of the same length (`ShapeError` otherwise).
 * @param options The `SegmentOptions`, with `welch`'s defaults.
 * @returns The coherence as a `Spectrum` (`quantity: 'coherence'`), with the number of segments.
 *
 * @example A noisy copy
 * // y = x + noise of a quarter the power: C = 1 / (1 + 0.25) = 0.8 at every frequency.
 * const x = normals(stream(1), 4096)
 * const y = add(x, normals(stream(2), 4096, 0, 0.5))
 * const C = coherence(x, y, { nperseg: 128 })
 * print('segments =', C.segments, ' mean coherence =', mean(C.values))
 * print('an unrelated pair:', mean(coherence(x, normals(stream(3), 4096), { nperseg: 128 }).values))
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
 * The level that the estimated coherence of two independent signals exceeds with probability $1 - \text{level}$, from
 * $K$ independent segments: under independence $P(\hat{C} \ge c) = (1 - c)^{K-1}$ (Carter, Knapp and Nuttall, 1973),
 * so $c = 1 - (1 - \text{level})^{1/(K-1)}$. Overlapping segments are not independent; pass their effective number.
 *
 * @param segments The number of independent segments $K$; 1 or fewer gives 1 (one segment's coherence is always 1).
 * @param options Options.
 * @param options.level The probability that the threshold is not exceeded under independence (default 0.95).
 * @returns The threshold $c$.
 *
 * @example More segments, a lower threshold
 * for (const K of [2, 8, 32]) print(`K = ${K}:`, coherenceThreshold(K))
 */
export function coherenceThreshold(segments: Scalar, { level = 0.95 }: { level?: Scalar } = {}): number {
  if (!(segments > 1)) return 1
  return 1 - Math.pow(1 - level, 1 / (segments - 1))
}

/** A two-sided $\chi^2$ interval for a spectrum. */
export type SpectralInterval = {
  /** Lower limits, one per frequency. */
  lower: Tensor
  /** Upper limits, one per frequency. */
  upper: Tensor
  /** The degrees of freedom used, per frequency. */
  dof: Tensor
  /** The confidence level, e.g. 0.95. */
  level: Scalar
}

/**
 * The $p$-quantile of $\chi^2_\nu$: $2 P^{-1}(\nu/2, p)$, through the inverse regularised incomplete gamma function.
 *
 * @param nu The degrees of freedom $\nu > 0$ (need not be an integer).
 * @param p The probability $p$, in $[0, 1]$.
 * @returns The value below which $\chi^2_\nu$ falls with probability $p$.
 *
 * @example Two degrees of freedom
 * // chi^2_2 is exponential with mean 2: its p-quantile is -2 ln(1 - p).
 * print('2.5% and 97.5% =', chiSquareQuantile(2, 0.025), chiSquareQuantile(2, 0.975))
 * print('-2 ln(0.975), -2 ln(0.025) =', -2 * Math.log(0.975), -2 * Math.log(0.025))
 */
export function chiSquareQuantile(nu: Scalar, p: Scalar): number {
  return 2 * (regularisedGammaPInverse(nu / 2, p) as number)
}

/**
 * The $100 \cdot \text{level}\%$ confidence interval of a power spectral estimate $\hat{S}(f)$ with $\nu$ equivalent
 * degrees of freedom: $\nu\hat{S}/S \sim \chi^2_\nu$, so
 * $[\nu\hat{S} / \chi^2_\nu(1 - \alpha/2), \nu\hat{S} / \chi^2_\nu(\alpha/2)]$ with $\alpha = 1 - \text{level}$
 * (Percival and Walden, 1993, §6.10; Welch, 1967). On a dB scale the interval has the same width at every frequency.
 * `dof` is a number (a periodogram's 2, Welch's `dof`, Blackman–Tukey's `dof`) or one per frequency (adaptive
 * multitaper). The $\chi^2$ law holds away from DC and Nyquist and for a spectrum smooth over the estimator's
 * bandwidth; it fails at spectral lines. Throws `ShapeError` when `dof` has not one value per frequency.
 *
 * @param s The estimate, a real-valued `Spectrum`.
 * @param dof The equivalent degrees of freedom: one number, or one per frequency.
 * @param options Options.
 * @param options.level The confidence level (default 0.95).
 * @returns The lower and upper limits, the degrees of freedom used and the level.
 *
 * @example Welch's estimate of white noise
 * // The true density is 2 everywhere: about 95% of the intervals should contain it.
 * const W = welch(normals(stream(1), 1024), { nperseg: 128 })
 * const ci = spectralConfidence(W, W.dof)
 * print('dof =', W.dof, ' bin 10:', ci.lower.data[10], '<', W.values.data[10], '<', ci.upper.data[10])
 * const covered = Array.from(W.values.data, (_, i) => ci.lower.data[i] <= 2 && 2 <= ci.upper.data[i])
 * print('intervals containing 2:', covered.filter(Boolean).length, 'of', covered.length)
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
  /** The closed interval of frequencies counted (default all). */
  band?: readonly [number, number]
  /** Closed intervals of frequencies left out (default none). */
  exclude?: readonly (readonly [number, number])[]
  /** The frequencies, when the estimate is given as bare values. */
  f?: ArrayLike<number>
}

/**
 * Whether a frequency is counted under the options: inside `band` and outside every `exclude` interval.
 *
 * @param freq The frequency, or undefined when the frequencies are not known (then it is always counted).
 * @param o The band and exclusions.
 * @returns True when the frequency counts.
 */
const counted = (freq: number | undefined, o: SpectralErrorOptions) =>
  freq === undefined ||
  ((!o.band || (freq >= o.band[0] && freq <= o.band[1])) && !(o.exclude ?? []).some(([a, b]) => freq >= a && freq <= b))

/**
 * The error of a spectral estimate against a reference on the same frequencies, in dB:
 * $d_i = 10 \log_{10}(\hat{S}_i / S_i)$. `bias` is the mean of $d$, `sd` its standard deviation and `distance` the
 * log-spectral distance $\sqrt{\operatorname{mean} d^2}$, so $\text{distance}^2 = \text{bias}^2 + \text{sd}^2$.
 * Frequencies where either spectrum is not positive, outside `band` or inside an `exclude` interval are left out; with
 * no frequencies known (bare values and no `f`) the band and exclusions are ignored. All NaN when nothing is counted.
 * Throws `ShapeError` when the lengths differ.
 *
 * @param estimate The estimate $\hat{S}$: a `Spectrum` (whose `f` is used) or bare values.
 * @param reference The reference $S$ at the same frequencies.
 * @param options Which frequencies count; see `SpectralErrorOptions`.
 * @param options.band The closed interval of frequencies counted (default all).
 * @param options.exclude Closed intervals of frequencies left out (default none).
 * @param options.f The frequencies of the values, needed for `band` and `exclude` when `estimate` is bare values
 *   (default the estimate's `f`).
 * @returns `bias`, `sd` and `distance` in dB, and `count`, the number of frequencies counted.
 *
 * @example Welch against the flat truth
 * const W = welch(normals(stream(1), 4096), { nperseg: 128 })
 * const reference = Array(W.f.shape[0]).fill(2)
 * print('all bins:', logSpectralError(W, reference))
 * print('away from DC and Nyquist:', logSpectralError(W, reference, { band: [0.01, 0.49] }))
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
 * The bias and variance of a spectral estimator, from $R$ estimates of independent realisations on one frequency
 * grid, in dB: at each frequency the mean of $10 \log_{10} \hat{S}_r$ against $10 \log_{10} S$ (squared:
 * $\text{bias}^2$) and the variance of $10 \log_{10} \hat{S}_r$ about that mean, each averaged over the counted
 * frequencies; $\text{mse} = \text{bias}^2 + \text{variance}$. Smoothing more (shorter segments, a shorter lag
 * window, more tapers, a lower AR order) trades variance for bias. On the log scale even a consistent estimator is
 * biased by $\expect \log(\chi^2_\nu/\nu)$ ($-2.5$ dB for a periodogram's $\nu = 2$). Estimates are floored at
 * $10^{-300}$ before the log; frequencies where the reference is not positive are left out.
 *
 * @param estimates The $R$ estimates, each with one value per frequency of `reference`.
 * @param reference The true spectrum $S$.
 * @param options The band and exclusions, which need the frequencies `f`; see `SpectralErrorOptions`.
 * @returns `bias2`, `variance` and `mse` in $\text{dB}^2$, and `count`, the number of frequencies counted (all NaN
 *   when it is 0).
 *
 * @example The periodogram against Blackman–Tukey
 * // White noise of density 2: the periodogram has the -2.5 dB log bias and a large variance; the smoothed
 * // Blackman–Tukey estimate trades most of both away.
 * const f = Array.from({ length: 33 }, (_, k) => k / 64)
 * const reference = Array(33).fill(2)
 * const runs = (estimate) => Array.from({ length: 20 }, (_, r) => estimate(normals(stream(r), 64)).values.data)
 * const options = { f, band: [0.01, 0.49] }
 * print('periodogram:', replicateSpectralError(runs((x) => periodogram(x)), reference, options))
 * print('Blackman–Tukey:', replicateSpectralError(runs((x) => blackmanTukey(x, { nfft: 64 })), reference, options))
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
 * Whether a spectral estimate shows two peaks near $f_1 < f_2$ as separate: the highest value within a quarter of the
 * separation of each, and the lowest value between them; `dip` is the weaker peak's height above that valley in dB,
 * and the peaks count as resolved when the dip is at least `threshold` (default 3 dB, the Rayleigh-like criterion).
 * When no bin lies near a peak or between them, the dip is 0 and the peaks are not resolved.
 *
 * @param s The estimate, a real-valued `Spectrum`.
 * @param f1 One peak's frequency, in the units of `s.f`.
 * @param f2 The other's (the order does not matter).
 * @param options Options.
 * @param options.threshold The dip in dB at which the peaks count as resolved (default 3).
 * @returns `dip` in dB and `resolved`.
 *
 * @example Short segments blur two close tones
 * // Tones 0.02 cycles per sample apart: Welch with 32-sample segments merges them; a Hann periodogram of all 256
 * // samples separates them.
 * const tone = (f, i) => Math.sin(2 * Math.PI * f * i)
 * const x = Array.from({ length: 256 }, (_, i) => tone(0.1, i) + tone(0.12, i))
 * print('welch, nperseg 32:', peakDip(welch(x, { nperseg: 32 }), 0.1, 0.12))
 * print('periodogram, hann:', peakDip(periodogram(x, { window: 'hann' }), 0.1, 0.12))
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
 * The delay $\tau$ of $y$ behind $x$ from the phase of their cross-spectral density: for $y = x(t - \tau)$ filtered
 * and with noise, $\arg S_{xy}(f) = -2\pi f \tau$, so $\tau$ is the slope of the unwrapped phase through the origin,
 * fitted by least squares weighted by $C/(1 - C)$ (the inverse phase variance, up to a constant) over the bins with
 * coherence at least `minCoherence` (default 0.5), unwrapped in order of frequency. In the time units of the spectra
 * (samples when $f_s = 1$). The DC bin and bins with coherence exactly 1 (infinite weight) are skipped, so a
 * noise-free copy gives NaN.
 *
 * @param cross The cross-spectral density, as `csd` returns it.
 * @param coh The coherence on the same frequencies, as `coherence` returns it.
 * @param options Options.
 * @param options.minCoherence The least coherence a bin needs to be used (default 0.5).
 * @returns `delay` (NaN when no bin is used) and `bins`, the number of bins used.
 *
 * @example A 3-sample delay
 * const x = normals(stream(1), 4096)
 * const xs = x.data
 * const y = add(tensor(Array.from(xs, (_, i) => (i >= 3 ? xs[i - 3] : 0))), normals(stream(2), 4096, 0, 0.5))
 * print(crossSpectralDelay(csd(x, y, { nperseg: 128 }), coherence(x, y, { nperseg: 128 })))
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
