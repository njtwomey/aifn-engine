/**
 * The YIN fundamental-frequency estimator (de Cheveigné and Kawahara, 2002, J. Acoust. Soc. Am. 111(4)): the
 * difference function $d(\tau) = \sum_j (x[j] - x[j + \tau])^2$, its cumulative-mean normalisation
 * $d'(\tau) = d(\tau)\, \tau / \sum_{u=1}^{\tau} d(u)$ ($d'(0) = 1$), which removes the dip at $\tau = 0$ and the bias
 * towards short lags, the first dip below an absolute threshold, and parabolic interpolation. A frame-by-frame tracker
 * runs it over a signal. Lags $\tau$ are in samples; periods and times are reported in seconds and pitches in Hz.
 */

import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Scalar, Size } from 'aifn-compute/foundation/contracts'
import { readSamples, type SignalInput } from '../signal'
import { parabola } from './cepstrum'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * The YIN difference function and its normalised form over lags $0, \dots, \tau_{\max}$: `d` ($d(\tau)$) and `dPrime`
 * ($d'(\tau)$), each with $\tau_{\max} + 1$ entries indexed by lag in samples.
 */
export type YinDifference = { d: Tensor; dPrime: Tensor }

/**
 * The difference function $d(\tau) = \sum_{j < W} (x[j] - x[j + \tau])^2$ over a window of $W = n - \tau_{\max}$
 * samples (so every lag sees the same window), and $d'(\tau) = d(\tau)\, \tau / \sum_{u=1}^{\tau} d(u)$, with
 * $d'(0) = 1$ and $d'(\tau) = 1$ while the running sum is 0. Throws `DomainError` unless $1 \le \tau_{\max} < n$.
 *
 * @param x The frame of $n$ samples (a `Signal`, or bare samples); its sample rate is not used.
 * @param maxLag The largest lag $\tau_{\max}$, in samples.
 * @returns $d$ and $d'$ over lags $0, \dots, \tau_{\max}$.
 *
 * @example A sinusoid of period 4 samples
 * // Both dip to 0 at the period, tau = 4; d' starts at 1 instead of d's 0.
 * const x = Array.from({ length: 16 }, (_, i) => Math.sin((2 * Math.PI * i) / 4))
 * const { d, dPrime } = yinDifference(x, 6)
 * print('d =', d)
 * print("d' =", dPrime)
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
  /** The sample rate in Hz (default the signal's, or 1 for bare samples). */
  fs?: Scalar
  /** The lowest pitch searched, in Hz (default 60): the longest lag is $\lceil f_s / f_{\min} \rceil$ samples. */
  fmin?: Scalar
  /** The highest pitch searched, in Hz (default 500): the shortest lag is $\max(2, \lfloor f_s / f_{\max} \rfloor)$. */
  fmax?: Scalar
  /** The absolute threshold on $d'$ (default 0.1, as the paper). */
  threshold?: Scalar
}

/**
 * One YIN estimate: $f_0$, the period, $d'$ at the chosen dip (the aperiodicity) and whether it fell below the
 * threshold.
 */
export type YinEstimate = {
  /** The pitch $f_0 = 1 / \text{period}$, in Hz. */
  f0: Scalar
  /** $\tau^\star / f_s$, seconds, with sub-sample refinement. */
  period: Scalar
  /** $d'$ at the chosen lag (before refinement): near 0 for a periodic frame, near 1 for noise. */
  aperiodicity: Scalar
  /** True when a dip of $d'$ fell below the threshold; otherwise the estimate is the global minimum. */
  voiced: boolean
  /** The normalised difference $d'(\tau)$ over lags $0, \dots, \lceil f_s / f_{\min} \rceil$. */
  dPrime: Tensor
  /** The lags of `dPrime`, in seconds. */
  lags: Tensor
}

/**
 * YIN on one frame: the smallest lag in $[f_s / f_{\max}, f_s / f_{\min}]$ where $d'$ dips below `threshold`,
 * followed to the bottom of that dip and refined by a parabola; when no dip crosses the threshold, the global minimum
 * of $d'$ over those lags (flagged unvoiced). The frame must be longer than twice the longest period,
 * $2 \lceil f_s / f_{\min} \rceil$ samples, or `DomainError` is thrown.
 *
 * @param x The frame: a single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The sample rate, pitch range and threshold; see `YinOptions`.
 * @returns The estimate, with $d'$ over the lags searched.
 *
 * @example A tone with a second harmonic
 * const fs = 8000
 * const tone = (f, i) => Math.sin((2 * Math.PI * f * i) / fs)
 * const x = Array.from({ length: 400 }, (_, i) => tone(200, i) + 0.5 * tone(400, i))
 * const e = yinPitch(x, { fs })
 * print('f0 =', e.f0, 'Hz  voiced =', e.voiced, ' aperiodicity =', e.aperiodicity)
 *
 * @example White noise is unvoiced
 * const e = yinPitch(normals(stream(1), 400), { fs: 8000 })
 * print('voiced =', e.voiced, ' aperiodicity =', e.aperiodicity)
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

/**
 * A pitch track: per frame, `t` its centre time in seconds, `f0` in Hz (NaN when unvoiced) and `aperiodicity` ($d'$
 * at the chosen lag).
 */
export type PitchTrack = { t: Tensor; f0: Tensor; aperiodicity: Tensor }

/**
 * YIN frame by frame: frames of `frameLength` samples (default 3 periods of $f_{\min}$,
 * $\lceil 3 f_s / f_{\min} \rceil$) every `hop` samples (default a quarter frame), each estimated by `yinPitch`.
 * Unvoiced frames get an $f_0$ of NaN. Only whole frames are used, so a signal shorter than a frame gives an empty
 * track. Bare samples need the `fs` option: at the default $f_s = 1$ the frame is too short and `yinPitch` throws.
 *
 * @param x The single-channel signal (a `Signal`, or bare samples at the `fs` option's rate).
 * @param options The `YinOptions`, and `frameLength` and `hop`, both in samples.
 * @returns The track: frame centre times (from the signal's `t0`), $f_0$ and aperiodicity per frame.
 *
 * @example A tone that stops
 * // 75 ms of 200 Hz, then silence: frames that are half silence or more are unvoiced.
 * const fs = 8000
 * const x = Array.from({ length: 1200 }, (_, i) => (i < 600 ? Math.sin((2 * Math.PI * 200 * i) / fs) : 0))
 * const track = yin(x, { fs })
 * print('t =', track.t)
 * print('f0 =', track.f0)
 * print('aperiodicity =', track.aperiodicity)
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
