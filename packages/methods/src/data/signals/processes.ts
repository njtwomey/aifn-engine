/**
 * Seeded series with a known power spectrum, for comparing spectral estimates with the truth. Each generator returns a
 * `SignalDataset` (times in `x`, $n \times 1$; values in `y`; the samples as a `Signal`) whose `meta.truth` is a
 * `SpectralTruth`: the continuous power spectral density of its stochastic part (`psd(f)`, from
 * `aifn-compute/signal/statistical`'s `armaSpectrum`) and its line spectrum (`lines`).
 *
 * - `sinusoidsInNoise`: two sinusoids in white noise (a line spectrum on a flat floor);
 * - `arProcess`: an AR(2) with a conjugate pole pair of given radius and frequency, or any AR($p$);
 * - `armaProcess`: an ARMA(2, 2) with a pole pair (a peak) and a zero pair (a notch), or any ARMA($p$, $q$);
 * - `unevenSinusoids`: a sinusoid observed at uneven times (random, with gaps, or nightly and seasonal, as in
 *   astronomy) with per-sample uncertainties, for the Lomb–Scargle periodogram;
 * - `coupledProcesses`: an AR(2) $x$ and $y = h * x + v$ (a delayed low-pass filter plus independent noise), with
 *   known cross-spectrum and coherence.
 *
 * Processes are simulated by `aifn-compute/signal/filters`' `lfilter` from white noise after a burn-in of 2000
 * samples, whose transient decays as $r^{2000}$ for the largest pole radius $r$: below $e^{-10}$ for $r \le 0.995$,
 * the largest the registered knobs allow, so the samples are a draw from the stationary process. Every random part
 * draws from its own child of the stream (`'noise'`, `'phases'`, `'times'`, ...), so changing one knob leaves the
 * other draws alone.
 */

import type { DatasetInfo, Signal, Size } from 'aifn-compute/foundation/contracts'
import { child, normal, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import { definer } from 'aifn-compute/foundation/registry'
import { bool, int, oneOf, real, space } from 'aifn-compute/foundation/space'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { signal } from 'aifn-compute/signal'
import { lfilter } from 'aifn-compute/signal/filters'
import { spectralTruth, type ArmaParts, type SpectralLine, type SpectralModel } from '../truth'
import { checkCount, generatorRecipe, matrix, vector, type Dataset } from '../types'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A series with a known spectrum: times in `x` ($n \times 1$), values in `y`, the same samples as a `Signal`. */
export type SignalDataset = Dataset & {
  /** The samples with their rate (for uneven sampling, `fs` is the mean rate $n / T$). */
  readonly signal: Signal
  /** The second series $y$ of a coupled pair (`coupledProcesses` only). */
  readonly partner?: Signal
  /** Per-sample noise standard deviations (`unevenSinusoids` only), for the weighted Lomb–Scargle periodogram. */
  readonly dy?: Tensor
}

/** The samples simulated and discarded before a process is recorded, so that its start-up transient has decayed. */
const BURN = 2000

/**
 * Simulate $n$ samples of the ARMA process
 * $x_t = \sum_i \phi_i x_{t-i} + \varepsilon_t + \sum_j \theta_j \varepsilon_{t-j}$ driven by
 * $\varepsilon_t \sim \Gauss(0, \sigma^2)$, after a burn-in of `BURN` samples. With no coefficients it is the white
 * noise itself.
 *
 * @param s The stream the $n + 2000$ innovations are drawn from.
 * @param p The process: the AR coefficients $\phi$, the MA coefficients $\theta$ and the innovation variance
 *   $\sigma^2$.
 * @param n The number of samples kept after the burn-in.
 * @returns The $n$ samples, a fresh array.
 */
function simulate(s: Stream, p: ArmaParts, n: Size): Float64Array {
  const eps = toFlat(normals(s, n + BURN, 0, Math.sqrt(p.sigma2)))
  if (!p.ar.length && !p.ma.length) return Float64Array.from(eps.slice(BURN))
  const y = toFlat(lfilter({ b: [1, ...p.ma], a: [1, ...p.ar.map((v) => -v)] }, eps).y as Tensor)
  return Float64Array.from(y.slice(BURN))
}

/**
 * The AR coefficients $(\phi_1, \phi_2) = (2r \cos 2\pi f, -r^2)$ whose polynomial $1 - \phi_1 z^{-1} - \phi_2 z^{-2}$
 * has the conjugate pair of roots $r e^{\pm 2\pi i f}$. Negated, they are the MA coefficients of a zero pair.
 *
 * @param radius The radius $r$ of the pair.
 * @param frequency The angle of the pair over $2\pi$, $f$, in cycles per sample.
 * @returns The two coefficients $(\phi_1, \phi_2)$.
 */
function pair(radius: number, frequency: number): [number, number] {
  return [2 * radius * Math.cos(2 * Math.PI * frequency), -radius * radius]
}

/**
 * Give each tone a phase drawn uniformly on $[0, 2\pi)$ from `s`, one draw per tone in order.
 *
 * @param s The stream the phases are drawn from.
 * @param tones The tones' frequencies and amplitudes.
 * @returns The tones as spectral lines, with their phases in radians.
 */
function phased(s: Stream, tones: readonly { frequency: number; amplitude: number }[]): SpectralLine[] {
  return tones.map((t) => ({ frequency: t.frequency, amplitude: t.amplitude, phase: 2 * Math.PI * uniform(s) }))
}

/**
 * Assemble a `SignalDataset` from its times and values: `x` the times ($n \times 1$), `y` and `signal` the values,
 * and metadata with the spectral truth of `model` and the recipe of the generator call.
 *
 * @param s The stream the generator was called with; its key goes into `meta.key` and the recipe's seed.
 * @param model The spectral model the values were drawn from; its `name` names the dataset, its `fs` is the signal's
 *   rate, and it becomes `meta.truth` through `spectralTruth`.
 * @param t The $n$ sample times.
 * @param y The $n$ values (not copied).
 * @param base The generator's registry key, recorded in the recipe.
 * @param knobs The generator's knobs, recorded in the recipe.
 * @param description One sentence describing the dataset, for `meta.description`.
 * @param extra The second series of a coupled pair (`partner`) and per-sample noise standard deviations (`dy`), each
 *   $n$ values, when the generator has them.
 * @returns The dataset.
 */
function build(
  s: Stream,
  model: SpectralModel,
  t: Float64Array,
  y: Float64Array,
  base: string,
  knobs: Record<string, unknown>,
  description: string,
  extra: { partner?: Float64Array; dy?: Float64Array } = {},
): SignalDataset {
  const n = t.length
  const sig = signal(fromData(y, [n]), { fs: model.fs })
  return {
    kind: 'dataset',
    x: matrix(Float64Array.from(t), n, 1),
    y: vector(y),
    signal: sig,
    ...(extra.partner ? { partner: signal(fromData(extra.partner, [n]), { fs: model.fs }) } : {}),
    ...(extra.dy ? { dy: vector(extra.dy) } : {}),
    meta: {
      name: model.name,
      description,
      task: 'sequence',
      featureNames: ['t'],
      targetName: 'x',
      key: s.key,
      truth: spectralTruth(model),
      recipe: generatorRecipe(base, s.key, knobs),
    },
  }
}

/**
 * Evenly spaced times $t_i = i / f_s$, $i = 0, \dots, n - 1$.
 *
 * @param n The number of times.
 * @param fs The sample rate $f_s$.
 * @returns The $n$ times.
 */
const evenTimes = (n: Size, fs: number) => Float64Array.from({ length: n }, (_, i) => i / fs)
/**
 * The elementwise sum of two series of the same length, as a new array.
 *
 * @param a The first series (not modified).
 * @param b The second series, at least as long as `a`.
 * @returns $a_i + b_i$ for every index of `a`.
 */
const add = (a: Float64Array, b: ArrayLike<number>) => a.map((v, i) => v + b[i])
/**
 * The deterministic part $\sum_k A_k \sin(2\pi f_k t + \varphi_k)$ at each time.
 *
 * @param t The times.
 * @param lines The sinusoids: frequency $f_k$, amplitude $A_k$ and phase $\varphi_k$.
 * @returns The sum at each time, the length of `t`.
 */
const linesAt = (t: Float64Array, lines: readonly SpectralLine[]) =>
  t.map((ti) => lines.reduce((acc, l) => acc + l.amplitude * Math.sin(2 * Math.PI * l.frequency * ti + l.phase), 0))

// ── Sinusoids in white noise ─────────────────────────────────────────────────────────────────────────────────────────

/** Options of `sinusoidsInNoise`. */
export interface SinusoidsInNoiseOptions {
  /** The number of samples (default 512). */
  n?: Size
  /** Sample rate (default 1: frequencies in cycles per sample). */
  fs?: number
  /** Frequency of the first tone (default 0.1). */
  f1?: number
  /** Amplitude of the first tone (default 1; 0 drops it). */
  a1?: number
  /** Frequency of the second tone (default 0.13). */
  f2?: number
  /** Amplitude of the second tone (default 0.5; 0 drops it). */
  a2?: number
  /** White-noise standard deviation (default 1). */
  noise?: number
  /** Any number of tones, in place of `f1`, `a1`, `f2` and `a2` (not recorded in the recipe). */
  tones?: readonly { frequency: number; amplitude: number }[]
}

/**
 * Sinusoids $\sum_k A_k \sin(2\pi f_k t + \varphi_k)$ in white noise of standard deviation $\sigma$, at the even times
 * $t_i = i / f_s$. The phases $\varphi_k$ are uniform, from `child(s, 'phases')`, and the noise from
 * `child(s, 'noise')`; tones of amplitude 0 are dropped. The true spectrum is lines of power $A_k^2 / 2$ at $f_k$ on a
 * flat one-sided floor $2\sigma^2 / f_s$. Throws `DomainError` unless `n` is a non-negative integer.
 *
 * @param s The stream the phases and the noise are drawn from.
 * @param options The length, the sample rate, the tones and the noise level.
 * @returns The series, with its `SpectralTruth` in `meta.truth`.
 *
 * @example Two tones in noise, and their true spectrum
 * const d = sinusoidsInNoise(stream(0), { n: 64 })
 * print('samples:', d.y.shape[0], ' first:', toArray(d.y).slice(0, 3))
 * print('line frequencies:', d.meta.truth.lines.map((l) => l.frequency))
 * print('total variance (1 + 1/2 + 1/8):', d.meta.truth.variance)
 */
export function sinusoidsInNoise(s: Stream, options: SinusoidsInNoiseOptions = {}): SignalDataset {
  const { n = 512, fs = 1, f1 = 0.1, a1 = 1, f2 = 0.13, a2 = 0.5, noise = 1 } = options
  checkCount(n, 'sinusoidsInNoise')
  const tones = (
    options.tones ?? [
      { frequency: f1, amplitude: a1 },
      { frequency: f2, amplitude: a2 },
    ]
  ).filter((t) => t.amplitude !== 0)
  const lines = phased(child(s, 'phases'), tones)
  const parts: ArmaParts = { ar: [], ma: [], sigma2: noise * noise }
  const t = evenTimes(n, fs)
  const y = add(linesAt(t, lines), simulate(child(s, 'noise'), parts, n))
  const model: SpectralModel = { name: 'sinusoids in noise', fs, lines, noise: parts }
  return build(
    s,
    model,
    t,
    y,
    'sinusoidsInNoise',
    { n, fs, f1, a1, f2, a2, noise },
    `${n} samples at ${fs} Hz of ${tones.length} sinusoid${tones.length === 1 ? '' : 's'} (${tones
      .map((tn) => `${tn.frequency} Hz, amplitude ${tn.amplitude}`)
      .join('; ')}) in white noise of sd ${noise}.`,
  )
}

// ── AR and ARMA processes ────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `arProcess`. */
export interface ArProcessOptions {
  /** The number of samples (default 512). */
  n?: Size
  /** Sample rate (default 1: frequencies in cycles per sample). */
  fs?: number
  /** Radius of the conjugate pole pair, in $[0, 1)$ (default 0.95): the closer to 1, the sharper the peak. */
  radius?: number
  /**
   * Frequency of the pole pair, the pole angle over $2\pi$, in cycles per sample (default 0.2): where the peak sits.
   */
  frequency?: number
  /** Innovation standard deviation (default 1). */
  sigma?: number
  /**
   * Any stationary AR($p$) coefficients $\phi_1, \dots, \phi_p$, in place of the pole pair (not recorded in the
   * recipe).
   */
  ar?: readonly number[]
}

/**
 * An autoregression $x_t = \sum_i \phi_i x_{t-i} + \varepsilon_t$ with $\varepsilon_t \sim \Gauss(0, \sigma^2)$, at the
 * even times $t = i / f_s$: by default an AR(2) with poles $r e^{\pm 2\pi i f}$, whose spectrum peaks near $f$ with a
 * width of about $(1 - r) / \pi$ cycles per sample. The true one-sided PSD at frequency $\nu$ is
 * $\frac{2\sigma^2 / f_s}{\lvert 1 - \sum_k \phi_k e^{-2\pi i \nu k / f_s} \rvert^2}$. The innovations come from
 * `child(s, 'noise')`. Throws `DomainError` unless `n` is a non-negative integer and the radius is in $[0, 1)$ (checked
 * even when `ar` is given); stationarity of a given `ar` is not checked.
 *
 * @param s The stream the innovations are drawn from.
 * @param options The length, the sample rate, the pole pair (or any coefficients) and the innovation standard
 *   deviation.
 * @returns The series, with its `SpectralTruth` in `meta.truth`.
 *
 * @example The true spectrum peaks at the pole frequency
 * const d = arProcess(stream(1), { n: 256 })
 * print('samples:', d.y.shape[0], ' first:', toArray(d.y).slice(0, 3))
 * print('PSD at 0.2 and 0.45 cycles per sample:', d.meta.truth.psd([0.2, 0.45]))
 */
export function arProcess(s: Stream, options: ArProcessOptions = {}): SignalDataset {
  const { n = 512, fs = 1, radius = 0.95, frequency = 0.2, sigma = 1 } = options
  checkCount(n, 'arProcess')
  if (!(radius >= 0 && radius < 1)) throw new DomainError('arProcess', 'arProcess: radius must be in [0, 1)')
  const ar = options.ar ? [...options.ar] : pair(radius, frequency)
  const parts: ArmaParts = { ar, ma: [], sigma2: sigma * sigma }
  const t = evenTimes(n, fs)
  const y = simulate(child(s, 'noise'), parts, n)
  return build(
    s,
    { name: `AR(${ar.length})`, fs, lines: [], noise: parts },
    t,
    y,
    'arProcess',
    { n, fs, radius, frequency, sigma },
    options.ar
      ? `${n} samples of the AR(${ar.length}) with φ = (${ar.map((v) => v.toFixed(4)).join(', ')}), σ = ${sigma}.`
      : `${n} samples of an AR(2) with poles at radius ${radius} and ${frequency} cycles per sample, σ = ${sigma}.`,
  )
}

/** Options of `armaProcess`. */
export interface ArmaProcessOptions {
  /** The number of samples (default 512). */
  n?: Size
  /** Sample rate (default 1: frequencies in cycles per sample). */
  fs?: number
  /** Radius of the pole pair (a peak), in $[0, 1)$ (default 0.9). */
  poleRadius?: number
  /** Frequency of the pole pair, in cycles per sample (default 0.125). */
  poleFrequency?: number
  /** Radius of the zero pair (a notch) (default 0.9): the closer to 1, the deeper the notch. */
  zeroRadius?: number
  /** Frequency of the zero pair, in cycles per sample (default 0.3). */
  zeroFrequency?: number
  /** Innovation standard deviation $\sigma$ (default 1). */
  sigma?: number
  /** Any stationary AR coefficients $\phi$, in place of the pole pair (not recorded in the recipe). */
  ar?: readonly number[]
  /** Any MA coefficients $\theta$, in place of the zero pair (not recorded in the recipe). */
  ma?: readonly number[]
}

/**
 * An ARMA process $x_t = \sum_i \phi_i x_{t-i} + \varepsilon_t + \sum_j \theta_j \varepsilon_{t-j}$ with
 * $\varepsilon_t \sim \Gauss(0, \sigma^2)$: by default an ARMA(2, 2) whose pole pair makes a peak and whose zero pair
 * makes a notch. The true one-sided PSD is $\frac{2\sigma^2}{f_s} \lvert B \rvert^2 / \lvert A \rvert^2$, with $A$ and
 * $B$ the AR and MA polynomials on the unit circle, which no finite AR fits exactly. The innovations come from
 * `child(s, 'noise')`. Throws `DomainError` unless `n` is a non-negative integer; the pole radius and stationarity are
 * not checked.
 *
 * @param s The stream the innovations are drawn from.
 * @param options The length, the sample rate, the pole and zero pairs (or any coefficients) and the innovation
 *   standard deviation.
 * @returns The series, with its `SpectralTruth` in `meta.truth`.
 *
 * @example A peak at the poles and a notch at the zeros
 * const d = armaProcess(stream(2), { n: 256 })
 * print('samples:', d.y.shape[0], ' first:', toArray(d.y).slice(0, 3))
 * print('PSD at the peak (0.125) and the notch (0.3):', d.meta.truth.psd([0.125, 0.3]))
 */
export function armaProcess(s: Stream, options: ArmaProcessOptions = {}): SignalDataset {
  const {
    n = 512,
    fs = 1,
    poleRadius = 0.9,
    poleFrequency = 0.125,
    zeroRadius = 0.9,
    zeroFrequency = 0.3,
    sigma = 1,
  } = options
  checkCount(n, 'armaProcess')
  const ar = options.ar ? [...options.ar] : pair(poleRadius, poleFrequency)
  // B(z) = 1 − 2ρ cos θ z⁻¹ + ρ² z⁻² has zeros at ρe^{±iθ}: θ₁ = −2ρ cos θ, θ₂ = ρ².
  const ma = options.ma ? [...options.ma] : pair(zeroRadius, zeroFrequency).map((v) => -v)
  const parts: ArmaParts = { ar, ma, sigma2: sigma * sigma }
  const t = evenTimes(n, fs)
  const y = simulate(child(s, 'noise'), parts, n)
  return build(
    s,
    { name: `ARMA(${ar.length}, ${ma.length})`, fs, lines: [], noise: parts },
    t,
    y,
    'armaProcess',
    { n, fs, poleRadius, poleFrequency, zeroRadius, zeroFrequency, sigma },
    `${n} samples of an ARMA(${ar.length}, ${ma.length}) with φ = (${ar.map((v) => v.toFixed(3)).join(', ')}) and θ = (${ma
      .map((v) => v.toFixed(3))
      .join(', ')}), σ = ${sigma}.`,
  )
}

// ── Uneven sampling ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Sampling patterns of `unevenSinusoids`: uniform times, uniform outside gaps, or nightly within seasons. */
export type UnevenSampling = 'random' | 'gaps' | 'seasonal'

/** Options of `unevenSinusoids`. */
export interface UnevenSinusoidsOptions {
  /** Number of samples (fewer when the seasonal pattern has fewer usable nights). */
  n?: Size
  /** Time span $T$ (default 100; read as days for `seasonal`). */
  span?: number
  /**
   * `random`: times uniform on $[0, T]$. `gaps`: uniform outside three random gaps covering `gapFraction` of the span.
   * `seasonal`: one observation per clear night (a fraction `gapFraction` of nights clouded out), near the same time of
   * night (jitter of an hour), only during an observing season of 60% of each 365.25-day year: aliases at 1 cycle per
   * day and 1 per year.
   */
  sampling?: UnevenSampling
  /** The share of the span lost to gaps (`gaps`) or of nights clouded out (`seasonal`) (default 0.3). */
  gapFraction?: number
  /** Frequency of the sinusoid, in cycles per unit of time (default 0.37). */
  frequency?: number
  /** Amplitude of the sinusoid (default 1). */
  amplitude?: number
  /** Frequency of a second sinusoid (default 0.11). */
  frequency2?: number
  /** Amplitude of the second sinusoid (default 0, which drops it). */
  amplitude2?: number
  /** Noise standard deviation (default 0.5). */
  noise?: number
  /** Per-sample sds varying log-uniformly over a factor of 4 around `noise` (default false). */
  heteroscedastic?: boolean
}

/**
 * Draw sorted sample times for a sampling pattern (see `UnevenSinusoidsOptions.sampling`). The seasonal pattern has
 * an observing season of the first 60% of each 365.25-day year, one observation per clear night at 0.3 of the day
 * plus a normal jitter of one hour, and is thinned at random to $n$ when it has more nights; it may have fewer.
 *
 * @param s The stream the times (and gaps, and clouded nights) are drawn from.
 * @param n The number of times (at most, for `seasonal`).
 * @param span The span $T$: times lie in $[0, T]$ (`seasonal`: one candidate night per whole day of it).
 * @param sampling The pattern.
 * @param gapFraction The share of the span covered by the three gaps (`gaps`), or the chance that a night is clouded
 *   out (`seasonal`); unused for `random`.
 * @returns The times, ascending.
 */
function unevenTimes(s: Stream, n: Size, span: number, sampling: UnevenSampling, gapFraction: number): Float64Array {
  if (sampling === 'seasonal') {
    const year = 365.25
    const nights: number[] = []
    for (let d = 0; d < span; d++) {
      if ((d % year) / year > 0.6) continue
      if (uniform(s) < gapFraction) continue
      nights.push(d + 0.3 + (normal(s) as number) / 24)
    }
    // Thin the nights at random to n when there are more.
    while (nights.length > n) nights.splice(Math.floor(uniform(s) * nights.length), 1)
    return Float64Array.from(nights)
  }
  const gaps: [number, number][] = []
  if (sampling === 'gaps' && gapFraction > 0) {
    // Three gaps of random lengths summing to gapFraction · T, placed without overlap.
    const shares = [uniform(s) + 0.2, uniform(s) + 0.2, uniform(s) + 0.2]
    const total = shares.reduce((a, b) => a + b, 0)
    const lengths = shares.map((v) => (v / total) * gapFraction * span)
    const free = span - lengths.reduce((a, b) => a + b, 0)
    const cuts = [uniform(s), uniform(s), uniform(s)].map((v) => v * free).sort((a, b) => a - b)
    let shift = 0
    cuts.forEach((c, i) => {
      gaps.push([c + shift, c + shift + lengths[i]])
      shift += lengths[i]
    })
  }
  const free = span - gaps.reduce((acc, [a, b]) => acc + b - a, 0)
  const t = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    // Draw on the free length, then step over the gaps before the draw.
    let u = uniform(s) * free
    for (const [a, b] of gaps) if (u >= a) u += b - a
    t[i] = u
  }
  return t.sort()
}

/**
 * A sinusoid (or two) observed at uneven times with Gaussian noise, the setting of the Lomb–Scargle periodogram.
 * Each sample has its own noise standard deviation (`dy`, constant unless `heteroscedastic`). The truth has the lines
 * and a white floor of the mean noise variance; `fs` is the mean rate $m / T$ of the $m$ samples drawn, which only
 * scales the floor's density. The times, the noise levels, the phases and the noise come from the children `'times'`,
 * `'dy'`, `'phases'` and `'noise'` of `s`. Throws `DomainError` unless `n` is a non-negative integer.
 *
 * @param s The stream everything random is drawn from.
 * @param options The number of samples, the span, the sampling pattern, the sinusoids and the noise.
 * @returns The series (times in `x`, ascending), with `dy` and its `SpectralTruth` in `meta.truth`.
 *
 * @example Two seasons of nightly observations
 * const d = unevenSinusoids(stream(3), { sampling: 'seasonal', span: 730, n: 120 })
 * print('samples:', d.y.shape[0], ' mean rate:', d.signal.fs, 'per day')
 * print('first times (days):', toArray(d.x).slice(0, 3))
 */
export function unevenSinusoids(s: Stream, options: UnevenSinusoidsOptions = {}): SignalDataset {
  const {
    n = 150,
    span = 100,
    sampling = 'random',
    gapFraction = 0.3,
    frequency = 0.37,
    amplitude = 1,
    frequency2 = 0.11,
    amplitude2 = 0,
    noise = 0.5,
    heteroscedastic = false,
  } = options
  checkCount(n, 'unevenSinusoids')
  const t = unevenTimes(child(s, 'times'), n, span, sampling, gapFraction)
  const m = t.length
  const ds = child(s, 'dy')
  const dy = Float64Array.from({ length: m }, () => (heteroscedastic ? noise * Math.pow(4, uniform(ds) - 0.5) : noise))
  const tones = [
    { frequency, amplitude },
    { frequency: frequency2, amplitude: amplitude2 },
  ].filter((tn) => tn.amplitude !== 0)
  const lines = phased(child(s, 'phases'), tones)
  const eps = toFlat(normals(child(s, 'noise'), m))
  const y = linesAt(t, lines).map((v, i) => v + dy[i] * eps[i])
  const meanVar = dy.reduce((acc, v) => acc + v * v, 0) / Math.max(m, 1)
  const fs = m / span
  return build(
    s,
    { name: `uneven sinusoid (${sampling})`, fs, lines, noise: { ar: [], ma: [], sigma2: meanVar } },
    t,
    y,
    'unevenSinusoids',
    { n, span, sampling, gapFraction, frequency, amplitude, frequency2, amplitude2, noise, heteroscedastic },
    `${m} ${sampling} samples over ${span} time units of ${tones.map((tn) => `a sinusoid at ${tn.frequency} (amplitude ${tn.amplitude})`).join(' and ')} in noise of sd ${noise}${heteroscedastic ? ' (varying by sample)' : ''}.`,
    { dy },
  )
}

// ── A coupled pair ───────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `coupledProcesses`. */
export interface CoupledProcessesOptions {
  /** The number of samples of each series (default 2048). */
  n?: Size
  /** Sample rate (default 1: frequencies in cycles per sample). */
  fs?: number
  /** The radius of the pole pair of the input $x$, an AR(2) (default 0.9). */
  radius?: number
  /** The frequency of that pole pair, in cycles per sample (default 0.1). */
  frequency?: number
  /** $y$'s delay behind $x$, in samples (default 5). */
  delay?: Size
  /** The filter's gain (default 1). */
  gain?: number
  /** The standard deviation of $y$'s independent white noise $v$ (default 1). */
  noise?: number
}

/**
 * A pair $(x, y)$: $x$ an AR(2) with unit innovation variance, $y = h * x + v$ with
 * $h = g \cdot [\tfrac14, \tfrac12, \tfrac14]$ delayed by $\delta$ = `delay` samples ($g$ = `gain`; a low-pass with a
 * zero at Nyquist, so $H(f) = g \cos^2(\pi f / f_s) e^{-2\pi i f (\delta + 1) / f_s}$) and $v$ white noise
 * independent of $x$. The true cross-spectrum is $S_{xy} = H S_{xx}$, its phase $-2\pi f (\delta + 1) / f_s$ (a
 * straight line whose slope gives the delay), and the coherence
 * $\lvert H \rvert^2 S_{xx} / (\lvert H \rvert^2 S_{xx} + S_{vv})$ is high where $x$ is strong and the filter passes
 * it, low elsewhere. `y` and `signal` hold $x$, `partner` holds $y$; the first $\delta + 2$ samples are dropped so that
 * the filter has a full history. $x$ comes from `child(s, 'x')` and $v$ from `child(s, 'v')`. Throws `DomainError`
 * unless `n` is a non-negative integer.
 *
 * @param s The stream both series are drawn from.
 * @param options The length, the sample rate, the input's pole pair, the delay, the gain and the noise level.
 * @returns The series $x$, with $y$ in `partner` and the pair's spectra in `meta.truth.coupled`.
 *
 * @example Coherence is high where the input is strong, and the phase gives the delay
 * const d = coupledProcesses(stream(4), { n: 256 })
 * print('samples of x and y:', d.signal.data.shape[0], d.partner.data.shape[0])
 * print('coherence at 0.1 and 0.45:', d.meta.truth.coupled.coherence([0.1, 0.45]))
 * const p = toArray(d.meta.truth.coupled.phase([0.01, 0.02]))
 * print('delay from the phase slope:', -(p[1] - p[0]) / (2 * Math.PI * 0.01) - 1)
 */
export function coupledProcesses(s: Stream, options: CoupledProcessesOptions = {}): SignalDataset {
  const { n = 2048, fs = 1, radius = 0.9, frequency = 0.1, delay = 5, gain = 1, noise = 1 } = options
  checkCount(n, 'coupledProcesses')
  const parts: ArmaParts = { ar: pair(radius, frequency), ma: [], sigma2: 1 }
  const x = simulate(child(s, 'x'), parts, n + delay + 2)
  const b = [...new Array<number>(delay).fill(0), gain / 4, gain / 2, gain / 4]
  const hx = toFlat(lfilter({ b, a: [1] }, x).y as Tensor)
  const vNoise: ArmaParts = { ar: [], ma: [], sigma2: noise * noise }
  const v = simulate(child(s, 'v'), vNoise, n)
  // Drop the first delay + 2 samples so that y's filter has a full history.
  const skip = delay + 2
  const xs = Float64Array.from(x.slice(skip))
  const ys = Float64Array.from({ length: n }, (_, i) => hx[i + skip] + v[i])
  return build(
    s,
    { name: 'coupled pair', fs, lines: [], noise: parts, coupling: { b, a: [1], noise: vNoise } },
    evenTimes(n, fs),
    xs,
    'coupledProcesses',
    { n, fs, radius, frequency, delay, gain, noise },
    `${n} samples of an AR(2) x (poles at radius ${radius}, ${frequency} cycles per sample) and y = ${gain}·[¼, ½, ¼] ∗ x delayed ${delay} samples plus white noise of sd ${noise}.`,
    { partner: ys },
  )
}

// ── Registration ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** Registers a generator of this file (kind `dataset`, area `data/signals`). */
const dataset = definer<DatasetInfo>('dataset', 'data/signals')
/** The notes on spectral estimation that every spectral generator illustrates. */
const SPECTRA = ['periodogram', 'welch-method', 'multitaper-spectral-estimation', 'parametric-spectral-estimation']

dataset(
  {
    key: 'sinusoidsInNoise',
    name: 'Sinusoids in white noise',
    summary: 'Two sinusoids in white noise: a line spectrum on a flat floor.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(16, 16384, { default: 512 }),
      fs: real(0.01, 100000, { default: 1, scale: 'log' }),
      f1: real(0, 50000, { default: 0.1 }),
      a1: real(0, 100, { default: 1 }),
      f2: real(0, 50000, { default: 0.13 }),
      a2: real(0, 100, { default: 0.5 }),
      noise: real(0, 100, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: [...SPECTRA, 'subspace-frequency-estimation', 'spectral-leakage-and-windows'],
  },
  sinusoidsInNoise,
)

dataset(
  {
    key: 'arProcess',
    name: 'Autoregressive process',
    summary:
      'An AR(2) with a conjugate pole pair of chosen radius and frequency (or any AR(p)), with its analytic PSD.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(16, 16384, { default: 512 }),
      fs: real(0.01, 100000, { default: 1, scale: 'log' }),
      radius: real(0, 0.995, { default: 0.95 }),
      frequency: real(0, 0.5, { default: 0.2 }),
      sigma: real(0.01, 100, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: [...SPECTRA, 'autoregressive-model'],
  },
  arProcess,
)

dataset(
  {
    key: 'armaProcess',
    name: 'ARMA process',
    summary: 'An ARMA(2, 2) with a spectral peak from its poles and a notch from its zeros, with its analytic PSD.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(16, 16384, { default: 512 }),
      fs: real(0.01, 100000, { default: 1, scale: 'log' }),
      poleRadius: real(0, 0.995, { default: 0.9 }),
      poleFrequency: real(0, 0.5, { default: 0.125 }),
      zeroRadius: real(0, 1, { default: 0.9 }),
      zeroFrequency: real(0, 0.5, { default: 0.3 }),
      sigma: real(0.01, 100, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: [...SPECTRA, 'autoregressive-moving-average-model'],
  },
  armaProcess,
)

dataset(
  {
    key: 'unevenSinusoids',
    name: 'Unevenly sampled sinusoid',
    summary: 'A sinusoid observed at random, gapped or nightly-and-seasonal times, with per-sample noise.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(5, 5000, { default: 150 }),
      span: real(1, 10000, { default: 100, scale: 'log' }),
      sampling: oneOf(['random', 'gaps', 'seasonal'] as const),
      gapFraction: real(0, 0.9, { default: 0.3 }),
      frequency: real(0, 100, { default: 0.37 }),
      amplitude: real(0, 100, { default: 1 }),
      frequency2: real(0, 100, { default: 0.11 }),
      amplitude2: real(0, 100, { default: 0 }),
      noise: real(0.001, 100, { default: 0.5, scale: 'log' }),
      heteroscedastic: bool(),
    }),
    truth: true,
    random: true,
    notes: ['lomb-scargle-periodogram', 'aliasing'],
  },
  unevenSinusoids,
)

dataset(
  {
    key: 'coupledProcesses',
    name: 'Coupled pair of processes',
    summary:
      'x an AR(2) and y a delayed low-pass of x plus independent noise, with known cross-spectrum and coherence.',
    task: 'sequence',
    output: 'dataset',
    knobs: space({
      n: int(64, 65536, { default: 2048 }),
      fs: real(0.01, 100000, { default: 1, scale: 'log' }),
      radius: real(0, 0.995, { default: 0.9 }),
      frequency: real(0, 0.5, { default: 0.1 }),
      delay: int(0, 100, { default: 5 }),
      gain: real(0, 100, { default: 1 }),
      noise: real(0, 100, { default: 1 }),
    }),
    truth: true,
    random: true,
    notes: ['coherence-and-cross-spectra'],
  },
  coupledProcesses,
)
