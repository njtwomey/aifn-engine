/**
 * Spectra of unevenly sampled series: the Lomb–Scargle periodogram (Lomb, 1976; Scargle, 1982) in its classic,
 * floating-mean and generalised (weighted) forms (Zechmeister and Kürster, 2009), a frequency grid for it, the
 * false-alarm probability of its highest peak by Baluev's (2008) bound, and the spectral window of a sampling pattern,
 * which shows the aliases the pattern creates (VanderPlas, 2018). Times and frequencies share a unit (seconds and Hz,
 * or days and cycles per day); no sample rate is involved.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Signal, Size, Spectrum, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { linearInterpolant, evaluatePiecewise } from 'aifn-compute/numerics/interpolate'
import { logGamma } from 'aifn-compute/numerics/special'
import { readSamples, signal, spectrum } from '../signal'

/** Options of `lombScargle`. */
export type LombScargleOptions = {
  /**
   * `classic`: Lomb–Scargle on the mean-subtracted series, a sinusoid fitted at each frequency (scipy's
   * `precenter=True`). `floating-mean` (default): a sinusoid plus a constant fitted at each frequency (Cumming, Marcy
   * and Butler, 1999; scipy's `floating_mean=True`), which stays right when the sampling correlates with the phase.
   * `generalised`: the floating-mean fit weighted by $1/\text{dy}^2$ (Zechmeister and Kürster, 2009).
   */
  method?: 'classic' | 'floating-mean' | 'generalised'
  /** Per-sample uncertainties, all positive (the `generalised` weights $1/\text{dy}^2$); ignored by other methods. */
  dy?: VectorLike
  /**
   * `standard` (default): the fraction of the variance the sinusoid explains,
   * $(\chi^2_\text{ref} - \chi^2(f)) / \chi^2_\text{ref} \in [0, 1]$. `psd`: $\tfrac12 (\chi^2_\text{ref} - \chi^2(f))$
   * with $\chi^2$ weighted by $1/\text{dy}^2$ (dy = 1 unless `generalised` with `dy`), which for even sampling is the
   * classical periodogram (scipy's unnormalised output).
   */
  normalization?: 'standard' | 'psd'
}

/** A Lomb–Scargle periodogram: a `Spectrum` of `power` over cycles per unit time, with what its FAP needs. */
export type LombScargle = Spectrum & {
  /** The fit used, as the `method` option. */
  method: 'classic' | 'floating-mean' | 'generalised'
  /** The normalisation of `values`, as the `normalization` option. */
  normalization: 'standard' | 'psd'
  /** Number of samples. */
  n: Size
  /** The weighted variance of the times (for the effective baseline of Baluev's bound). */
  timeVariance: Scalar
  /** The weighted variance of the values, $\chi^2_\text{ref} / \sum w$. */
  valueVariance: Scalar
  /** $\sum 1/\text{dy}^2$ ($n$ without `dy`). */
  weightSum: Scalar
}

/**
 * The values of a vector as a fresh float64 array.
 *
 * @param v A rank-1 tensor or an array.
 * @param where The caller's name, for error messages.
 * @returns A copy of the values.
 */
const toF64 = (v: VectorLike | Tensor, where: string) => readSamples(v as VectorLike, where).values

/**
 * The Lomb–Scargle periodogram of samples $y$ at times $t$, at frequencies $f$ (cycles per unit of $t$): for each $f$,
 * the least-squares fit of a sinusoid (plus a constant, unless `classic`) and the reduction in $\chi^2$ it gives. With
 * the normalised weights $w_i = (1/\text{dy}_i^2) / \sum (1/\text{dy}^2)$ and $c = \cos 2\pi f t$,
 * $s = \sin 2\pi f t$, the weighted centred sums $YY$, $YC$, $YS$, $CC$, $SS$ and $CS$ (each $\sum w_i u_i v_i$ minus
 * the product of the weighted means; `classic` centres only $y$) give
 * $p(f) = (SS \cdot YC^2 + CC \cdot YS^2 - 2\, CS \cdot YC \cdot YS) / (YY \cdot (CC \cdot SS - CS^2))$, which is
 * Scargle's $\tau$-shifted form without the shift (Zechmeister and Kürster, 2009, eq. 20). Unlike a DFT it needs no
 * grid, so gaps and jitter do not leak power through interpolation; the sampling still aliases (see
 * `spectralWindow`). A degenerate fit (a constant $y$, or a frequency the times cannot tell from 0) gives 0. Throws
 * `ShapeError` when `t`, `y` (and `dy`) differ in length and `DomainError` for fewer than three samples or a `dy`
 * that is not positive.
 *
 * @param t The sample times, in any order.
 * @param y The values, one per time.
 * @param frequencies The frequencies to evaluate, in cycles per unit of $t$ (e.g. from `lombScargleFrequencies`).
 * @param options The fit, uncertainties and normalisation; see `LombScargleOptions`.
 * @returns The periodogram as a `Spectrum` of `power`, with what `falseAlarmProbability` needs.
 *
 * @example A sinusoid sampled at random times
 * // 60 random times in [0, 20]: the peak sits at the true 0.7 cycles per unit, and explains almost all the variance.
 * const t = Array.from(uniform(stream(3), 0, 20, { shape: [60] }).data)
 * const y = t.map((ti) => Math.sin(2 * Math.PI * 0.7 * ti))
 * const ls = lombScargle(t, y, lombScargleFrequencies(t))
 * print('peak at', ls.f.data[argmax(ls.values)], ' power', max(ls.values))
 */
export function lombScargle(
  t: VectorLike | Tensor,
  y: VectorLike | Tensor,
  frequencies: VectorLike | Tensor,
  options: LombScargleOptions = {},
): LombScargle {
  const ts = toF64(t, 'lombScargle')
  const ys = toF64(y, 'lombScargle')
  const fs = toF64(frequencies, 'lombScargle')
  const n = ts.length
  if (ys.length !== n) throw new ShapeError('lombScargle', `lombScargle: t and y differ in length (${n}, ${ys.length})`)
  if (n < 3) throw new DomainError('lombScargle', 'lombScargle: needs at least three samples')
  const method = options.method ?? 'floating-mean'
  const normalization = options.normalization ?? 'standard'
  const raw = new Float64Array(n).fill(1)
  if (method === 'generalised' && options.dy !== undefined) {
    const dy = toF64(options.dy, 'lombScargle')
    if (dy.length !== n) throw new ShapeError('lombScargle', 'lombScargle: dy must have one value per sample')
    for (let i = 0; i < n; i++) {
      if (!(dy[i] > 0)) throw new DomainError('lombScargle', 'lombScargle: dy must be positive')
      raw[i] = 1 / (dy[i] * dy[i])
    }
  }
  let W = 0
  for (const v of raw) W += v
  const w = raw.map((v) => v / W)
  let Y = 0
  let tMean = 0
  for (let i = 0; i < n; i++) {
    Y += w[i] * ys[i]
    tMean += w[i] * ts[i]
  }
  let YY = 0
  let tVar = 0
  for (let i = 0; i < n; i++) {
    YY += w[i] * (ys[i] - Y) ** 2
    tVar += w[i] * (ts[i] - tMean) ** 2
  }
  const floating = method !== 'classic'
  const power = new Float64Array(fs.length)
  for (let k = 0; k < fs.length; k++) {
    const omega = 2 * Math.PI * fs[k]
    let C = 0
    let S = 0
    let YC = 0
    let YS = 0
    let CC = 0
    let SS = 0
    let CS = 0
    for (let i = 0; i < n; i++) {
      const c = Math.cos(omega * ts[i])
      const s = Math.sin(omega * ts[i])
      const yc = ys[i] - Y
      C += w[i] * c
      S += w[i] * s
      YC += w[i] * yc * c
      YS += w[i] * yc * s
      CC += w[i] * c * c
      SS += w[i] * s * s
      CS += w[i] * c * s
    }
    // y is already centred, so Σw(y−Y)c = YC − Y·C·0: only the trigonometric sums need centring.
    if (floating) {
      CC -= C * C
      SS -= S * S
      CS -= C * S
    }
    const D = CC * SS - CS * CS
    const p = D > 0 && YY > 0 ? (SS * YC * YC + CC * YS * YS - 2 * CS * YC * YS) / (YY * D) : 0
    power[k] = normalization === 'standard' ? p : 0.5 * W * YY * p
  }
  return {
    ...spectrum({
      f: fromData(Float64Array.from(fs), [fs.length]),
      axis: 'hz',
      values: fromData(power, [fs.length]),
      quantity: 'power',
      sided: 'one',
    }),
    method,
    normalization,
    n,
    timeVariance: tVar,
    valueVariance: YY,
    weightSum: W,
  }
}

/**
 * A frequency grid for `lombScargle`, as astropy's `autofrequency`: spacing $1/(\text{samplesPerPeak} \cdot T)$ for
 * the baseline $T = \max t - \min t$ (a peak's width is about $1/T$), from `minimum` (default half a spacing) to
 * `maximum` (default `nyquistFactor` times the "average Nyquist frequency" $n/(2T)$). Uneven sampling has no hard
 * Nyquist limit, so the top is a choice. Throws `DomainError` when the times span no interval.
 *
 * @param t The sample times.
 * @param options Options.
 * @param options.samplesPerPeak Grid points across a peak's width $1/T$ (default 5).
 * @param options.nyquistFactor The top of the grid as a multiple of $n/(2T)$ (default 5).
 * @param options.minimum The first frequency (default half a spacing).
 * @param options.maximum The top frequency (default from `nyquistFactor`); the grid stops at the nearest whole number
 *   of spacings.
 * @returns The evenly spaced frequencies, at least one.
 *
 * @example Ten unit-spaced times
 * // T = 9: spacing 1/45, from 1/90 up to 5 * 10 / 18 = 2.78.
 * const f = lombScargleFrequencies(Array.from({ length: 10 }, (_, i) => i))
 * print('count =', f.shape[0], ' first =', f.data[0], ' spacing =', f.data[1] - f.data[0])
 * print('last =', f.data[f.shape[0] - 1])
 */
export function lombScargleFrequencies(
  t: VectorLike | Tensor,
  {
    samplesPerPeak = 5,
    nyquistFactor = 5,
    minimum,
    maximum,
  }: { samplesPerPeak?: Scalar; nyquistFactor?: Scalar; minimum?: Scalar; maximum?: Scalar } = {},
): Tensor {
  const ts = toF64(t, 'lombScargleFrequencies')
  let lo = Infinity
  let hi = -Infinity
  for (const v of ts) {
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const T = hi - lo
  if (!(T > 0)) throw new DomainError('lombScargleFrequencies', 'lombScargleFrequencies: the times span no interval')
  const df = 1 / (samplesPerPeak * T)
  const fmin = minimum ?? 0.5 * df
  const fmax = maximum ?? (nyquistFactor * ts.length) / (2 * T)
  const count = Math.max(1, 1 + Math.round((fmax - fmin) / df))
  return fromData(Float64Array.from({ length: count }, (_, k) => fmin + k * df))
}

/**
 * The probability that noise alone gives a peak at least as high as `power` somewhere below `maximum` frequency, by
 * Baluev's (2008) upper bound for the `standard` floating-mean periodogram (as astropy's `fap_baluev`):
 * $\text{FAP} \approx 1 - (1 - \text{FAP}_1(z))\, e^{-\tau(z)}$ with $\text{FAP}_1(z) = (1 - z)^{(n-3)/2}$,
 * $\tau(z) = \gamma(n - 1)\, W (1 - z)^{(n-4)/2} \sqrt{(n - 1) z / 2}$, $W = f_{\max} \sqrt{4\pi \var t}$ and
 * $\gamma(N) = \sqrt{2/N}\, \Gamma(N/2) / \Gamma((N-1)/2)$, where $\text{FAP}_1$ is the single-frequency tail (a Beta
 * law of the explained variance) and $\tau$ counts the effective number of independent upcrossings. astropy's
 * `false_alarm_probability` takes $f_{\max}$ as the top of its `autofrequency` grid (`lombScargleFrequencies`), not
 * the requested maximum. For the `psd` normalisation (in units of the noise variance) $\text{FAP}_1 = e^{-z}$ and
 * $\tau = W e^{-z} \sqrt{z}$, which assumes `dy` are the noise's standard deviations. Tight for small FAPs;
 * conservative otherwise.
 *
 * @param power The peak height $z$, in the periodogram's normalisation (clamped to $[0, 1]$ for `standard`).
 * @param ls The periodogram the peak came from, for $n$, $\var t$ and its normalisation.
 * @param options Options.
 * @param options.maximum The top frequency $f_{\max}$ searched (default the largest of `ls.f`).
 * @returns The false-alarm probability, in $[0, 1]$.
 *
 * @example Signal against noise
 * // The same random times: a sinusoid buried in unit noise, and the noise alone.
 * const t = Array.from(uniform(stream(3), 0, 20, { shape: [60] }).data)
 * const f = lombScargleFrequencies(t)
 * const tone = add(tensor(t.map((ti) => Math.sin(2 * Math.PI * 0.7 * ti))), normals(stream(4), 60))
 * for (const [name, y] of [['tone + noise', tone], ['noise', normals(stream(5), 60)]]) {
 *   const ls = lombScargle(t, y, f)
 *   print(`${name}: highest peak`, max(ls.values), ' FAP', falseAlarmProbability(max(ls.values), ls))
 * }
 */
export function falseAlarmProbability(power: Scalar, ls: LombScargle, { maximum }: { maximum?: Scalar } = {}): number {
  const fmax = maximum ?? Math.max(...(ls.f.data as Float64Array))
  const W = fmax * Math.sqrt(4 * Math.PI * ls.timeVariance)
  const n = ls.n
  let single: number
  let tau: number
  if (ls.normalization === 'psd') {
    // ½Δχ² with χ² weighted by 1/dy²: in units of the noise variance when dy are the true noise sds.
    const z = Math.max(power, 0)
    single = Math.exp(-z)
    tau = W * Math.exp(-z) * Math.sqrt(z)
  } else {
    const z = Math.min(Math.max(power, 0), 1)
    const NH = n - 1
    const NK = n - 3
    const gamma = Math.sqrt(2 / NH) * Math.exp((logGamma(NH / 2) as number) - (logGamma((NH - 1) / 2) as number))
    single = Math.pow(1 - z, 0.5 * NK)
    tau = gamma * W * Math.pow(1 - z, 0.5 * (NK - 1)) * Math.sqrt(0.5 * NH * z)
  }
  // 1 − (1 − FAP₁)e^{−τ}, written to keep precision when the result is tiny.
  return Math.min(1, Math.max(0, -Math.expm1(-tau) + single * Math.exp(-tau)))
}

/**
 * The power whose false-alarm probability (`falseAlarmProbability`) is `probability`: the detection threshold a
 * peak must exceed to be called significant at that level. Found by bisection on $[0, 1]$. Throws `DomainError`
 * unless the periodogram has the `standard` normalisation.
 *
 * @param probability The false-alarm probability wanted, e.g. 0.01.
 * @param ls The periodogram, for $n$, $\var t$ and the frequency grid.
 * @param options `maximum`, the top frequency searched (default the largest of `ls.f`), as `falseAlarmProbability`.
 * @returns The power $z$ with that false-alarm probability.
 *
 * @example Thresholds at 1% and 10%
 * const t = Array.from(uniform(stream(3), 0, 20, { shape: [60] }).data)
 * const ls = lombScargle(t, normals(stream(5), 60), lombScargleFrequencies(t))
 * print('1%:', falseAlarmLevel(0.01, ls), ' 10%:', falseAlarmLevel(0.1, ls), ' highest noise peak:', max(ls.values))
 */
export function falseAlarmLevel(probability: Scalar, ls: LombScargle, options: { maximum?: Scalar } = {}): number {
  if (ls.normalization !== 'standard')
    throw new DomainError('falseAlarmLevel', 'falseAlarmLevel: needs the standard normalisation')
  let lo = 0
  let hi = 1
  for (let it = 0; it < 100; it++) {
    const mid = 0.5 * (lo + hi)
    if (falseAlarmProbability(mid, ls, options) > probability) lo = mid
    else hi = mid
  }
  return 0.5 * (lo + hi)
}

/**
 * The spectral window of a sampling pattern: $\abs{\sum_j e^{-2\pi i f t_j}}^2 / n^2$, the periodogram of a constant
 * observed at the times $t$ (Scargle, 1982; VanderPlas, 2018, §4). It is 1 at $f = 0$ and its other peaks are the
 * aliases the sampling creates: a true frequency $f_0$ shows up again at $f_0 \pm f_\text{alias}$ for each window peak
 * $f_\text{alias}$ (1 cycle per day for nightly observations, 1 per year for seasonal ones). Even sampling with
 * spacing $\Delta$ has peaks at multiples of $1/\Delta$.
 *
 * @param t The $n$ sample times.
 * @param frequencies The frequencies to evaluate, in cycles per unit of $t$.
 * @returns The window as a `Spectrum` of `power`, in $[0, 1]$.
 *
 * @example Even and jittered daily sampling
 * // 30 daily samples alias 1 cycle per day onto 0; jittering the times weakens the alias.
 * const daily = Array.from({ length: 30 }, (_, i) => i)
 * const jittered = daily.map((ti) => ti + 0.1 * Math.sin(7 * ti))
 * print('daily at 0, 0.5, 1:', spectralWindow(daily, [0, 0.5, 1]).values)
 * print('jittered at 0, 0.5, 1:', spectralWindow(jittered, [0, 0.5, 1]).values)
 */
export function spectralWindow(t: VectorLike | Tensor, frequencies: VectorLike | Tensor): Spectrum {
  const ts = toF64(t, 'spectralWindow')
  const fs = toF64(frequencies, 'spectralWindow')
  const n = ts.length
  const out = new Float64Array(fs.length)
  for (let k = 0; k < fs.length; k++) {
    const omega = 2 * Math.PI * fs[k]
    let re = 0
    let im = 0
    for (let i = 0; i < n; i++) {
      re += Math.cos(omega * ts[i])
      im -= Math.sin(omega * ts[i])
    }
    out[k] = (re * re + im * im) / (n * n)
  }
  return spectrum({
    f: fromData(Float64Array.from(fs), [fs.length]),
    axis: 'hz',
    values: fromData(out, [fs.length]),
    quantity: 'power',
    sided: 'one',
  })
}

/**
 * Uneven samples put on a regular grid of spacing `dt` from the first time to the last, so that a DFT-based estimate
 * can be applied: `linear` interpolates between neighbouring samples (a low-pass that also bridges every gap with a
 * straight line), `zero-fill` puts each mean-subtracted sample in its nearest grid cell (averaging cells that receive
 * several) and leaves the others at zero, whose periodogram is the uneven data's spectrum convolved with the spectral
 * window. Returns a `Signal` with $f_s = 1/\text{dt}$ and $t_0$ the first time. Throws `ShapeError` when `t` and `y`
 * differ in length and `DomainError` for a `dt` that is not positive.
 *
 * @param t The sample times, in any order (sorted here; of repeated times `linear` keeps the first).
 * @param y The values, one per time.
 * @param options Options.
 * @param options.dt The grid spacing, in the units of `t`; the grid has
 *   $\lfloor (t_{\max} - t_{\min}) / \text{dt} \rfloor + 1$ points.
 * @param options.method `'linear'` (default) or `'zero-fill'`.
 * @returns The gridded `Signal`.
 *
 * @example Interpolated and zero-filled
 * const t = [0, 0.9, 2.2, 3, 5]
 * const y = [0, 1, 2, 3, 5]
 * print('linear:', gridSamples(t, y, { dt: 1 }).data)
 * print('zero-fill (mean 2.2 removed):', gridSamples(t, y, { dt: 1, method: 'zero-fill' }).data)
 */
export function gridSamples(
  t: VectorLike | Tensor,
  y: VectorLike | Tensor,
  { dt, method = 'linear' }: { dt: Scalar; method?: 'linear' | 'zero-fill' },
): Signal {
  const ts = toF64(t, 'gridSamples')
  const ys = toF64(y, 'gridSamples')
  if (ts.length !== ys.length) throw new ShapeError('gridSamples', 'gridSamples: t and y differ in length')
  if (!(dt > 0)) throw new DomainError('gridSamples', 'gridSamples: dt must be positive')
  const order = Array.from(ts.keys()).sort((a, b) => ts[a] - ts[b])
  const st = Float64Array.from(order, (i) => ts[i])
  const sy = Float64Array.from(order, (i) => ys[i])
  const t0 = st[0]
  const m = Math.floor((st[st.length - 1] - t0) / dt) + 1
  const out = new Float64Array(m)
  if (method === 'linear') {
    // Keep the first of samples at equal times, so the knots increase strictly.
    const keep = Array.from(st.keys()).filter((i) => i === 0 || st[i] > st[i - 1])
    const pp = linearInterpolant(
      fromData(Float64Array.from(keep, (i) => st[i])),
      fromData(Float64Array.from(keep, (i) => sy[i])),
    )
    const grid = fromData(Float64Array.from({ length: m }, (_, k) => t0 + k * dt))
    out.set(evaluatePiecewise(pp, grid).data as Float64Array)
  } else {
    const mean = sy.reduce((a, b) => a + b, 0) / sy.length
    const counts = new Float64Array(m)
    for (let i = 0; i < st.length; i++) {
      const k = Math.min(m - 1, Math.round((st[i] - t0) / dt))
      out[k] += sy[i] - mean
      counts[k]++
    }
    for (let k = 0; k < m; k++) if (counts[k] > 1) out[k] /= counts[k]
  }
  return signal(fromData(out, [m]), { fs: 1 / dt, t0 })
}
