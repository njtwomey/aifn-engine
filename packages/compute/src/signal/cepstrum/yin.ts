/**
 * The YIN fundamental-frequency estimator (de Cheveigné and Kawahara, 2002, J. Acoust. Soc. Am. 111(4)): the
 * difference function d(τ) = Σⱼ (x[j] − x[j + τ])², its cumulative-mean normalisation d′(τ) = d(τ) τ / Σ_{u ≤ τ} d(u)
 * (d′(0) = 1), which removes the dip at τ = 0 and the bias towards short lags, the first dip below an absolute
 * threshold, and parabolic interpolation. A frame-by-frame tracker runs it over a signal.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { readSamples, type SignalInput } from '../signal'
import { parabola } from './cepstrum'
import { DomainError } from 'aifn-compute/foundation/errors'

/** The YIN difference function and its normalised form over lags 0 … maxLag. */
export type YinDifference = { d: Tensor; dPrime: Tensor }

/**
 * d(τ) = Σ_{j < W} (x[j] − x[j + τ])² over a window of W = n − maxLag samples (so every lag sees the same window), and
 * d′(τ) = d(τ) τ / Σ_{u=1}^{τ} d(u), with d′(0) = 1.
 */
export function yinDifference(x: SignalInput, maxLag: Size): YinDifference {
  const v = readSamples(x, 'yinDifference').values
  const n = v.length
  if (!(maxLag >= 1 && maxLag < n)) throw new DomainError('yinDifference', 'yinDifference: maxLag must lie in [1, n)')
  const W = n - maxLag
  const d = new Float64Array(maxLag + 1)
  for (let tau = 1; tau <= maxLag; tau++) {
    let s = 0
    for (let j = 0; j < W; j++) s += (v[j] - v[j + tau]) ** 2
    d[tau] = s
  }
  const dp = new Float64Array(maxLag + 1)
  dp[0] = 1
  let running = 0
  for (let tau = 1; tau <= maxLag; tau++) {
    running += d[tau]
    dp[tau] = running > 0 ? (d[tau] * tau) / running : 1
  }
  return { d: fromData(d, [maxLag + 1]), dPrime: fromData(dp, [maxLag + 1]) }
}

/** Options of `yinPitch` and `yin`. */
export type YinOptions = {
  fs?: Scalar
  /** The pitch range searched, Hz (default 60–500). */
  fmin?: Scalar
  fmax?: Scalar
  /** The absolute threshold on d′ (default 0.1, as the paper). */
  threshold?: Scalar
}

/** One YIN estimate: f₀, the period, d′ at the chosen dip (the aperiodicity) and whether it fell below the threshold. */
export type YinEstimate = {
  f0: Scalar
  /** τ⋆ / fs, seconds, with sub-sample refinement. */
  period: Scalar
  aperiodicity: Scalar
  voiced: boolean
  /** The normalised difference d′(τ) over the lags searched, and those lags in seconds. */
  dPrime: Tensor
  lags: Tensor
}

/**
 * YIN on one frame: the smallest lag in [fs/fmax, fs/fmin] where d′ dips below `threshold`, followed to the bottom of
 * that dip and refined by a parabola; when no dip crosses the threshold, the global minimum of d′ (flagged unvoiced).
 * The frame must be longer than twice the longest period.
 */
export function yinPitch(x: SignalInput, options: YinOptions = {}): YinEstimate {
  const input = readSamples(x, 'yinPitch', options.fs)
  const { fs } = input
  const { fmin = 60, fmax = 500, threshold = 0.1 } = options
  const n = input.values.length
  const tauMax = Math.ceil(fs / fmin)
  const tauMin = Math.max(2, Math.floor(fs / fmax))
  if (2 * tauMax >= n)
    throw new DomainError('yinPitch', 'yinPitch: the frame must be longer than twice the longest period')
  const { dPrime } = yinDifference(input.values, tauMax + 1)
  const dp = dPrime.data as Float64Array
  let tau = -1
  for (let t = tauMin; t <= tauMax; t++)
    if (dp[t] < threshold) {
      while (t + 1 <= tauMax && dp[t + 1] < dp[t]) t++
      tau = t
      break
    }
  const voiced = tau >= 0
  if (!voiced) {
    tau = tauMin
    for (let t = tauMin; t <= tauMax; t++) if (dp[t] < dp[tau]) tau = t
  }
  const shift = tau > 0 && tau < dp.length - 1 ? parabola(dp[tau - 1], dp[tau], dp[tau + 1]) : 0
  const period = (tau + shift) / fs
  return {
    f0: 1 / period,
    period,
    aperiodicity: dp[tau],
    voiced,
    dPrime: fromData(dp.slice(0, tauMax + 1), [tauMax + 1]),
    lags: fromData(
      Float64Array.from({ length: tauMax + 1 }, (_, k) => k / fs),
      [tauMax + 1],
    ),
  }
}

/** A pitch track: per frame, its centre time, f₀ (NaN when unvoiced) and aperiodicity. */
export type PitchTrack = { t: Tensor; f0: Tensor; aperiodicity: Tensor }

/**
 * YIN frame by frame: frames of `frameLength` samples (default 3 periods of fmin) every `hop` samples (default a
 * quarter frame). Unvoiced frames get f₀ = NaN.
 */
export function yin(x: SignalInput, options: YinOptions & { frameLength?: Size; hop?: Size } = {}): PitchTrack {
  const input = readSamples(x, 'yin', options.fs)
  const { fs, t0, values } = input
  const fmin = options.fmin ?? 60
  const L = options.frameLength ?? Math.ceil((3 * fs) / fmin)
  const hop = options.hop ?? Math.max(1, Math.floor(L / 4))
  const t: number[] = []
  const f0: number[] = []
  const ap: number[] = []
  for (let s = 0; s + L <= values.length; s += hop) {
    const e = yinPitch(values.subarray(s, s + L), { ...options, fs })
    t.push(t0 + (s + L / 2) / fs)
    f0.push(e.voiced ? e.f0 : NaN)
    ap.push(e.aperiodicity)
  }
  return {
    t: fromData(Float64Array.from(t), [t.length]),
    f0: fromData(Float64Array.from(f0), [f0.length]),
    aperiodicity: fromData(Float64Array.from(ap), [ap.length]),
  }
}
