/**
 * Seeded series with a known power spectrum, for comparing spectral estimates with the truth. Each generator returns a
 * `SignalDataset` (times in `x` [n, 1], values in `y`, the samples as a `Signal`) whose `meta.truth` is a
 * `SpectralTruth`: the continuous PSD of its stochastic part (`psd(f)`, from `aifn-compute/signal/statistical`'s
 * `armaSpectrum`) and its line spectrum (`lines`).
 *
 * - `sinusoidsInNoise`: two sinusoids in white noise (a line spectrum on a flat floor);
 * - `arProcess`: an AR(2) with a conjugate pole pair of given radius and frequency, or any AR(p);
 * - `armaProcess`: an ARMA(2, 2) with a pole pair (a peak) and a zero pair (a notch), or any ARMA(p, q);
 * - `unevenSinusoids`: a sinusoid observed at uneven times (random, with gaps, or nightly and seasonal, as in
 *   astronomy) with per-sample uncertainties, for the Lomb–Scargle periodogram;
 * - `coupledProcesses`: an AR(2) x and y = h ∗ x + v (a delayed low-pass filter plus independent noise), with known
 *   cross-spectrum and coherence.
 *
 * Processes are simulated by `aifn-compute/signal/filters`' `lfilter` from white noise after a burn-in, so the samples are
 * (to within e^{−10}) a draw from the stationary process.
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

/** A series with a known spectrum: times in `x`, values in `y`, the same samples as a `Signal`. */
export type SignalDataset = Dataset & {
  /** The samples with their rate (fs = the nominal mean rate for uneven sampling). */
  readonly signal: Signal
  /** The second series of a coupled pair. */
  readonly partner?: Signal
  /** Per-sample noise standard deviations (uneven data), for weighted Lomb–Scargle. */
  readonly dy?: Tensor
}

const BURN = 2000

/** n samples of an ARMA process driven by N(0, σ²) noise from `s`, after a burn-in. */
function simulate(s: Stream, p: ArmaParts, n: Size): Float64Array {
  const eps = toFlat(normals(s, n + BURN, 0, Math.sqrt(p.sigma2)))
  if (!p.ar.length && !p.ma.length) return Float64Array.from(eps.slice(BURN))
  const y = toFlat(lfilter({ b: [1, ...p.ma], a: [1, ...p.ar.map((v) => -v)] }, eps).y as Tensor)
  return Float64Array.from(y.slice(BURN))
}

/** A conjugate pair at radius r and frequency f (cycles per sample) as polynomial coefficients: (2r cos 2πf, −r²). */
function pair(radius: number, frequency: number): [number, number] {
  return [2 * radius * Math.cos(2 * Math.PI * frequency), -radius * radius]
}

/** Draw each line's phase uniformly from `s`. */
function phased(s: Stream, tones: readonly { frequency: number; amplitude: number }[]): SpectralLine[] {
  return tones.map((t) => ({ frequency: t.frequency, amplitude: t.amplitude, phase: 2 * Math.PI * uniform(s) }))
}

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

const evenTimes = (n: Size, fs: number) => Float64Array.from({ length: n }, (_, i) => i / fs)
const add = (a: Float64Array, b: ArrayLike<number>) => a.map((v, i) => v + b[i])
const linesAt = (t: Float64Array, lines: readonly SpectralLine[]) =>
  t.map((ti) => lines.reduce((acc, l) => acc + l.amplitude * Math.sin(2 * Math.PI * l.frequency * ti + l.phase), 0))

// ── Sinusoids in white noise ─────────────────────────────────────────────────────────────────────────────────────────

/** Options of `sinusoidsInNoise`. */
export interface SinusoidsInNoiseOptions {
  n?: Size
  /** Sample rate (default 1: frequencies in cycles per sample). */
  fs?: number
  /** First tone: frequency and amplitude (default 0.1, 1). */
  f1?: number
  a1?: number
  /** Second tone (default 0.13, 0.5; amplitude 0 drops it). */
  f2?: number
  a2?: number
  /** White-noise standard deviation (default 1). */
  noise?: number
  /** Any number of tones, in place of f1, a1, f2, a2. */
  tones?: readonly { frequency: number; amplitude: number }[]
}

/**
 * Sinusoids Σ Aₖ sin(2πfₖt + φₖ) in white noise of standard deviation σ, phases uniform from the stream `phases`. The
 * true spectrum is lines of power Aₖ²/2 at fₖ on a flat one-sided floor 2σ²/fs.
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
  n?: Size
  fs?: number
  /** Radius of the conjugate pole pair, in [0, 1) (default 0.95): the closer to 1, the sharper the peak. */
  radius?: number
  /** Frequency of the pole pair, the pole angle / 2π, in cycles per sample (default 0.2): where the peak sits. */
  frequency?: number
  /** Innovation standard deviation (default 1). */
  sigma?: number
  /** Any stationary AR(p) coefficients φ₁ … φ_p, in place of the pole pair. */
  ar?: readonly number[]
}

/**
 * An autoregression x_t = Σ φᵢ x_{t−i} + ε_t with ε_t ~ N(0, σ²): by default an AR(2) with poles r e^{±2πif}, whose
 * spectrum peaks near f with a width about (1 − r)/π cycles per sample. The true PSD is
 * 2σ²/fs / |1 − Σ φᵢ e^{−2πif'i/fs}|² (one-sided).
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
  n?: Size
  fs?: number
  /** The pole pair (a peak): radius in [0, 1) and frequency in cycles per sample (default 0.9, 0.125). */
  poleRadius?: number
  poleFrequency?: number
  /** The zero pair (a notch): radius and frequency (default 0.9, 0.3). */
  zeroRadius?: number
  zeroFrequency?: number
  sigma?: number
  /** Any coefficients, in place of the pairs: φ (stationary) and θ. */
  ar?: readonly number[]
  ma?: readonly number[]
}

/**
 * An ARMA process x_t = Σ φᵢ x_{t−i} + ε_t + Σ θⱼ ε_{t−j}: by default an ARMA(2, 2) whose pole pair makes a peak and
 * whose zero pair makes a notch. The true PSD is 2σ²/fs |B|²/|A|² (one-sided), which no finite AR fits exactly.
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

/** Sampling patterns of `unevenSinusoids`. */
export type UnevenSampling = 'random' | 'gaps' | 'seasonal'

/** Options of `unevenSinusoids`. */
export interface UnevenSinusoidsOptions {
  /** Number of samples (fewer when the seasonal pattern has fewer usable nights). */
  n?: Size
  /** Time span T (default 100; read as days for `seasonal`). */
  span?: number
  /**
   * `random`: times uniform on [0, T]. `gaps`: uniform outside a few random gaps covering `gapFraction` of the span.
   * `seasonal`: one observation per clear night (a fraction `gapFraction` of nights clouded out), near the same time of
   * night (jitter of an hour), only during an observing season of 60% of each 365.25-day year: aliases at 1 cycle per
   * day and 1 per year.
   */
  sampling?: UnevenSampling
  gapFraction?: number
  /** The sinusoid: frequency (cycles per unit time, default 0.37) and amplitude (default 1). */
  frequency?: number
  amplitude?: number
  /** A second sinusoid (amplitude 0, the default, drops it). */
  frequency2?: number
  amplitude2?: number
  /** Noise standard deviation (default 0.5). */
  noise?: number
  /** Per-sample sds varying log-uniformly over a factor of 4 around `noise` (default false). */
  heteroscedastic?: boolean
}

/** Sorted sample times for a pattern. */
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
 * Each sample has its own noise sd (`dy`, constant unless `heteroscedastic`). The truth has the lines and a white
 * floor; `fs` is the mean rate n/T, which only scales the floor's density.
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
  n?: Size
  fs?: number
  /** The input x: an AR(2) with a pole pair of this radius and frequency (default 0.9, 0.1). */
  radius?: number
  frequency?: number
  /** y's delay behind x, in samples (default 5). */
  delay?: Size
  /** The filter's gain (default 1). */
  gain?: number
  /** The sd of y's independent white noise v (default 1). */
  noise?: number
}

/**
 * A pair (x, y): x an AR(2), y = h ∗ x + v with h = gain · [¼, ½, ¼] delayed by `delay` samples (a low-pass with a
 * zero at Nyquist, so H(f) = gain · cos²(πf/fs) e^{−2πif(delay + 1)/fs}) and v white noise independent of x. The true
 * cross-spectrum is S_xy = H S_xx, its phase −2πf(delay + 1)/fs (a straight line whose slope gives the delay), and
 * the coherence |H|²S_xx / (|H|²S_xx + S_vv) is high where x is strong and the filter passes it, low elsewhere.
 * `partner` holds y.
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

const dataset = definer<DatasetInfo>('dataset', 'data/signals')
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
