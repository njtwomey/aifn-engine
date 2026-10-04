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
   * `generalised`: the floating-mean fit weighted by 1/dy² (Zechmeister and Kürster, 2009).
   */
  method?: 'classic' | 'floating-mean' | 'generalised'
  /** Per-sample uncertainties (the `generalised` weights 1/dy²). */
  dy?: VectorLike
  /**
   * `standard` (default): the fraction of the variance the sinusoid explains, (χ²_ref − χ²(f)) / χ²_ref ∈ [0, 1].
   * `psd`: ½(χ²_ref − χ²(f)) with χ² weighted by 1/dy² (dy = 1 by default), which for even sampling is the classical
   * periodogram (scipy's unnormalised output).
   */
  normalization?: 'standard' | 'psd'
}

/** A Lomb–Scargle periodogram: a `Spectrum` of `power` over cycles per unit time, with what its FAP needs. */
export type LombScargle = Spectrum & {
  method: 'classic' | 'floating-mean' | 'generalised'
  normalization: 'standard' | 'psd'
  /** Number of samples. */
  n: Size
  /** The weighted variance of the times (for the effective baseline of Baluev's bound). */
  timeVariance: Scalar
  /** The weighted variance of the values, χ²_ref / Σ w. */
  valueVariance: Scalar
  /** Σ 1/dy² (n without dy). */
  weightSum: Scalar
}

const toF64 = (v: VectorLike | Tensor, where: string) => readSamples(v as VectorLike, where).values

/**
 * The Lomb–Scargle periodogram of samples y at times t, at frequencies f (cycles per unit of t): for each f, the
 * least-squares fit of a sinusoid (plus a constant, unless `classic`) and the reduction in χ² it gives. With the
 * normalised weights wᵢ = (1/dyᵢ²)/Σ(1/dy²) and c = cos 2πft, s = sin 2πft, the weighted centred sums YY, YC, YS, CC,
 * SS and CS (each Σwᵢ uᵢvᵢ minus the product of the weighted means; `classic` centres only y) give
 *
 *   p(f) = (SS·YC² + CC·YS² − 2·CS·YC·YS) / (YY·(CC·SS − CS²)),
 *
 * which is Scargle's τ-shifted form without the shift (Zechmeister and Kürster, 2009, eq. 20). Unlike a DFT it needs
 * no grid, so gaps and jitter do not leak power through interpolation; the sampling still aliases (see
 * `spectralWindow`).
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
 * A frequency grid for `lombScargle`, as astropy's `autofrequency`: spacing 1/(samplesPerPeak · T) for the baseline
 * T = max t − min t (a peak's width is about 1/T), from `minimum` (default half a spacing) to `maximum` (default
 * nyquistFactor × the "average Nyquist frequency" n/(2T)). Uneven sampling has no hard Nyquist limit, so the top is
 * a choice.
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
 *
 *   FAP ≈ 1 − (1 − FAP₁(z)) e^{−τ(z)},  FAP₁(z) = (1 − z)^{(n−3)/2},
 *   τ(z) = γ(n − 1) · W · (1 − z)^{(n−4)/2} · √((n − 1) z / 2),  W = f_max √(4π Var t),  γ(N) = √(2/N) Γ(N/2)/Γ((N−1)/2),
 *
 * where FAP₁ is the single-frequency tail (a Beta law of the explained variance) and τ counts the effective number of
 * independent upcrossings. astropy's `false_alarm_probability` takes f_max as the top of its `autofrequency` grid
 * (`lombScargleFrequencies`), not the requested maximum. For the `psd` normalisation (in units of the noise variance) FAP₁ = e^{−z} and
 * τ = W e^{−z} √z, which assumes dy are the noise's standard deviations. Tight for small FAPs; conservative otherwise.
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
 * peak must exceed to be called significant at that level. Found by bisection on [0, 1] (`standard`).
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
 * The spectral window of a sampling pattern: |Σⱼ e^{−2πi f tⱼ}|² / n², the periodogram of a constant observed at the
 * times t (Scargle, 1982; VanderPlas, 2018, §4). It is 1 at f = 0 and its other peaks are the aliases the sampling
 * creates: a true frequency f₀ shows up again at f₀ ± f_alias for each window peak f_alias (1 cycle per day for nightly
 * observations, 1 per year for seasonal ones). Even sampling with spacing Δ has peaks at multiples of 1/Δ.
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
 * window. Returns a `Signal` with fs = 1/dt and t0 the first time.
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
