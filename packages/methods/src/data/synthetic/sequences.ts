/**
 * Seeded sequences: draws from a discrete hidden Markov model (and the occasionally dishonest casino), and synthetic
 * time series (autoregressive, seasonal with trend, random walk, and a random walk with planted motifs and a discord).
 *
 * The time series share one shape, `TimeSeries`: times $0, \dots, n - 1$ in `t`, the values in `y`, and `diverged`
 * when a value overflowed. Nothing is clipped. Every generator throws `DomainError` when `n` is not a non-negative
 * integer.
 */

import { normal, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { hmm, sampleHmm } from 'aifn-methods/inference/sequence-models'
import { checkCount, labels, matrix, vector, type DatasetMeta } from '../types'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * A discrete hidden Markov model: initial distribution ($k$), transition matrix ($k \times k$) and emission matrix
 * ($k \times m$).
 */
export interface DiscreteHmm {
  /** The distribution of the first state, length $k$. */
  initial: Tensor
  /** $P(z_t = j \mid z_{t-1} = i)$ in row $i$, column $j$; $k \times k$. */
  transition: Tensor
  /** $P(x_t = v \mid z_t = i)$ in row $i$, column $v$; $k \times m$. */
  emission: Tensor
}

/** A draw from a hidden Markov model: observations `x` and hidden states `z` (int32, length n). */
export interface HmmSample {
  /** The observed symbols, length $n$ (int32). */
  x: Tensor
  /** The hidden states, length $n$ (int32). */
  z: Tensor
  /** The model the draw came from. */
  model: DiscreteHmm
  /** The name, description, task (`'sequence'`), state names and stream key. */
  meta: DatasetMeta
}

/**
 * Copies of the rows of a stochastic matrix, each checked to sum to 1 (to within $10^{-9}$). Throws `DomainError`
 * naming the first row that does not; entries are not checked for sign.
 *
 * @param m The rows, each a probability distribution.
 * @param what The caller's name for error messages.
 * @returns The rows, copied.
 */
function rows(m: readonly (readonly number[])[], what: string): number[][] {
  return m.map((r, i) => {
    const total = r.reduce((a, b) => a + b, 0)
    if (Math.abs(total - 1) > 1e-9) throw new DomainError(what, `${what}: row ${i} sums to ${total}, not 1`)
    return [...r]
  })
}

/**
 * $n$ steps of a discrete hidden Markov model: $z_1 \sim \boldsymbol{\pi}$, $z_t \mid z_{t-1} \sim \Amat_{z_{t-1}}$,
 * $x_t \mid z_t \sim \Bmat_{z_t}$, with $\boldsymbol{\pi}$ the initial distribution and $\Amat$, $\Bmat$ the
 * transition and emission matrices. Observation symbols are $0, \dots, m - 1$. Drawn by
 * `aifn-methods/inference/sequence-models`' `sampleHmm` (step $t$ from `child(s, 'step', t)`). Throws `DomainError`
 * when a distribution does not sum to 1 or `n` is not a non-negative integer, and `ShapeError` when the sizes disagree.
 *
 * @param s The stream the states and symbols are drawn from.
 * @param model The model as plain arrays: `initial` ($k$ probabilities), `transition` ($k$ rows of $k$) and `emission`
 *   ($k$ rows of $m$), each row summing to 1.
 * @param n The number of steps.
 * @returns The symbols in `x`, the states in `z`, and the model as tensors.
 *
 * @example A sticky two-state chain in which state 0 always emits 0
 * const model = { initial: [1, 0], transition: [[0.9, 0.1], [0.2, 0.8]], emission: [[1, 0], [0.5, 0.5]] }
 * const h = hmmSample(stream(1), model, 12)
 * print('states: ', h.z)
 * print('symbols:', h.x)
 * const long = toArray(hmmSample(stream(2), model, 3000).z)
 * print('time in state 1 (stationary 1/3):', long.reduce((a, v) => a + v, 0) / long.length)
 */
export function hmmSample(
  s: Stream,
  model: {
    initial: readonly number[]
    transition: readonly (readonly number[])[]
    emission: readonly (readonly number[])[]
  },
  n: number,
): HmmSample {
  checkCount(n, 'hmmSample')
  const [initial] = rows([model.initial], 'hmmSample initial')
  const A = rows(model.transition, 'hmmSample transition')
  const B = rows(model.emission, 'hmmSample emission')
  const k = initial.length
  const m = B[0].length
  const drawn = sampleHmm(s, hmm(initial, A, B), n)
  return {
    x: drawn.observations,
    z: drawn.states,
    model: {
      initial: vector(initial),
      transition: matrix(Float64Array.from(A.flat()), k, k),
      emission: matrix(Float64Array.from(B.flat()), k, m),
    },
    meta: {
      name: 'hidden Markov model',
      description: `${n} steps of a ${k}-state hidden Markov model with ${m} symbols.`,
      task: 'sequence',
      featureNames: ['symbol'],
      labelNames: Array.from({ length: k }, (_, j) => `state ${j}`),
      key: s.key,
    },
  }
}

/** Options for `casino`. */
export interface CasinoOptions {
  /** Rolls. Default 300. */
  n?: number
  /** The probability of switching from the fair die to the loaded one, per roll. Default 0.05. */
  toLoaded?: number
  /** The probability of switching from the loaded die to the fair one, per roll. Default 0.1. */
  toFair?: number
  /** The probability that the loaded die shows six; the other five faces share the rest equally. Default 0.5. */
  loadedSix?: number
}

/**
 * The occasionally dishonest casino (Durbin, Eddy, Krogh and Mitchison, 1998, "Biological Sequence Analysis", section
 * 3.2): a fair die (state 0) and a loaded die (state 1) that shows six with probability `loadedSix` (half the time by
 * default). The chain starts from its stationary distribution, loaded with probability $p_l / (p_l + p_f)$ for the
 * switching probabilities $p_l$ = `toLoaded` and $p_f$ = `toFair`. Observations `x` are faces 1 to 6 (symbols 0 to 5
 * plus one); `model.emission` is indexed by the face minus one.
 *
 * @param s The stream the rolls are drawn from.
 * @param options The number of rolls, the two switching probabilities and the loaded die's chance of a six.
 * @returns The faces in `x`, the dice in `z` (0 fair, 1 loaded), and the model.
 *
 * @example The loaded die shows six half the time
 * const c = casino(stream(1), { n: 3000 })
 * const [x, z] = [toArray(c.x), toArray(c.z)]
 * print('first rolls:', x.slice(0, 15))
 * print('first dice: ', z.slice(0, 15))
 * const sixes = (die) => x.filter((v, t) => v === 6 && z[t] === die).length / z.filter((v) => v === die).length
 * print('share of sixes, fair (1/6):', sixes(0), ' loaded (1/2):', sixes(1))
 */
export function casino(s: Stream, options: CasinoOptions = {}): HmmSample {
  const { n = 300, toLoaded = 0.05, toFair = 0.1, loadedSix = 0.5 } = options
  const loaded = toLoaded / (toLoaded + toFair)
  const other = (1 - loadedSix) / 5
  const sample = hmmSample(
    s,
    {
      initial: [1 - loaded, loaded],
      transition: [
        [1 - toLoaded, toLoaded],
        [toFair, 1 - toFair],
      ],
      emission: [Array<number>(6).fill(1 / 6), [other, other, other, other, other, loadedSix]],
    },
    n,
  )
  const faces = Int32Array.from(sample.x.data, (v) => v + 1)
  return {
    ...sample,
    x: labels(faces),
    meta: {
      name: 'occasionally dishonest casino',
      description: `${n} rolls of a casino die that switches from fair to loaded with probability ${toLoaded} and back with probability ${toFair}; the loaded die shows six with probability ${loadedSix}.`,
      task: 'sequence',
      featureNames: ['face'],
      labelNames: ['fair', 'loaded'],
      source: 'Durbin, Eddy, Krogh and Mitchison (1998), Biological Sequence Analysis, §3.2',
      key: s.key,
    },
  }
}

/** A time series: times `t` and values `y` (length $n$), with a flag for divergence. */
export interface TimeSeries {
  /** The times $0, \dots, n - 1$. */
  t: Tensor
  /** The values, length $n$. */
  y: Tensor
  /**
   * True when the series overflowed to a non-finite value (e.g. a non-stationary AR model); values are kept as drawn.
   */
  diverged: boolean
  /** The name, description, task (`'sequence'`), feature name and stream key. */
  meta: DatasetMeta
}

/**
 * The time series of some values: the times $0, \dots, n - 1$, the values copied, and whether any is not finite.
 *
 * @param y The values.
 * @param meta The metadata to attach.
 * @returns The series.
 */
function series(y: Float64Array, meta: DatasetMeta): TimeSeries {
  return {
    t: fromData(Float64Array.from({ length: y.length }, (_, i) => i)),
    y: vector(y),
    diverged: y.some((v) => !Number.isFinite(v)),
    meta,
  }
}

/** Options for `arSeries`. */
export interface ArOptions {
  /** AR coefficients $a_1, \dots, a_p$ in $x_t = c + \sum_k a_k x_{t-k} + e_t$. Default [0.8]. */
  coefficients?: readonly number[]
  /** Length of the series, after the burn-in. Default 200. */
  n?: number
  /** Standard deviation $\sigma$ of the driving noise $e_t$. Default 1. */
  sd?: number
  /** Constant $c$. Default 0. */
  constant?: number
  /** Initial steps drawn and discarded so that the series starts near stationarity. Default 500. */
  burn?: number
}

/**
 * An autoregressive AR($p$) series $x_t = c + \sum_{k=1}^{p} a_k x_{t-k} + e_t$, $e_t \sim \Gauss(0, \sigma^2)$,
 * started from zeros (missing lags count as 0) and run for `burn` steps that are discarded. No clipping: an explosive
 * model sets `diverged`. The noise is drawn from `s` itself, one draw per step.
 *
 * @param s The stream the noise is drawn from.
 * @param options The coefficients, the length, the noise, the constant and the burn-in (`ArOptions`).
 * @returns The series of $n$ steps after the burn-in.
 *
 * @example The lag-1 autocorrelation and variance of an AR(1)
 * const a = arSeries(stream(1), { n: 3000, coefficients: [0.8] })
 * const y = toArray(a.y)
 * print('first values:', y.slice(0, 5))
 * const sq = y.reduce((s, v) => s + v * v, 0)
 * print('lag-1 autocorrelation (0.8):', y.slice(1).reduce((s, v, t) => s + v * y[t], 0) / sq)
 * print('variance (1 / (1 - 0.8^2) = 2.78):', sq / y.length)
 */
export function arSeries(s: Stream, options: ArOptions = {}): TimeSeries {
  const { coefficients: a = [0.8], n = 200, sd = 1, constant = 0, burn = 500 } = options
  checkCount(n, 'arSeries')
  const total = n + burn
  const x = new Float64Array(total)
  for (let t = 0; t < total; t++) {
    let v = constant + sd * normal(s)
    for (let k = 0; k < a.length; k++) if (t - k - 1 >= 0) v += a[k] * x[t - k - 1]
    x[t] = v
  }
  return series(x.slice(burn), {
    name: `AR(${a.length})`,
    description: `${n} steps of an AR(${a.length}) process with coefficients (${a.join(', ')}) and noise sd ${sd}.`,
    task: 'sequence',
    featureNames: ['x'],
    key: s.key,
  })
}

/** Options for `seasonalSeries`. */
export interface SeasonalOptions {
  /** Length of the series. Default 120. */
  n?: number
  /** Period of the seasonal cycle in steps. Default 12. */
  period?: number
  /** Amplitude of each harmonic of the season, first harmonic first. Default [1]. */
  amplitudes?: readonly number[]
  /** Level at $t = 0$. Default 0. */
  level?: number
  /** Slope of the trend, per step. Default 0.02. */
  trend?: number
  /** Standard deviation of white observation noise. Default 0.3. */
  noise?: number
  /**
   * Coefficient $\phi$ of AR(1) noise instead of white noise (0 for white), with $\lvert \phi \rvert < 1$; the marginal
   * standard deviation stays `noise`. Default 0.
   */
  persistence?: number
}

/**
 * A trend plus a seasonal cycle plus noise: $y_t = \ell + b t + \sum_{h=1}^{H} A_h \sin(2\pi h t / P) + u_t$, with
 * level $\ell$, slope $b$, period $P$ and the amplitudes $A_h$ of $H$ harmonics, where $u_t$ is white or AR(1) noise
 * with marginal standard deviation `noise`: $u_0 \sim \Gauss(0, \sigma^2)$ and
 * $u_t = \phi u_{t-1} + \sigma\sqrt{1 - \phi^2}\, z_t$. The noise is drawn from `s` itself.
 *
 * @param s The stream the noise is drawn from.
 * @param options The length, the period and amplitudes of the season, the level and trend, and the noise
 *   (`SeasonalOptions`).
 * @returns The series of $n$ steps.
 *
 * @example Without noise, one period on is the trend times the period
 * const a = seasonalSeries(stream(1), { n: 24, noise: 0 })
 * const y = toArray(a.y)
 * print('first period:', y.slice(0, 12))
 * print('y[t + 12] - y[t] (12 x 0.02):', y[17] - y[5])
 */
export function seasonalSeries(s: Stream, options: SeasonalOptions = {}): TimeSeries {
  const { n = 120, period = 12, amplitudes = [1], level = 0, trend = 0.02, noise = 0.3, persistence = 0 } = options
  checkCount(n, 'seasonalSeries')
  const y = new Float64Array(n)
  const innovation = noise * Math.sqrt(1 - persistence * persistence)
  let u = noise * normal(s)
  for (let t = 0; t < n; t++) {
    if (t > 0) u = persistence * u + innovation * normal(s)
    let season = 0
    amplitudes.forEach((amp, h) => (season += amp * Math.sin((2 * Math.PI * (h + 1) * t) / period)))
    y[t] = level + trend * t + season + u
  }
  return series(y, {
    name: 'seasonal series',
    description: `${n} steps of a trend (slope ${trend}) plus a season of period ${period} plus noise of sd ${noise}.`,
    task: 'sequence',
    featureNames: ['y'],
    key: s.key,
  })
}

/**
 * A Gaussian random walk $y_t = y_{t-1} + \mu + \sigma z_t$, $z_t \sim \Gauss(0, 1)$, from $y_0$ = `start`. The
 * steps are drawn from `s` itself.
 *
 * @param s The stream the steps are drawn from.
 * @param options `n`, the length including the start (default 200); `sd`, the step's standard deviation $\sigma$
 *   (default 1); `drift`, the mean step $\mu$ (default 0); and `start`, $y_0$ (default 0).
 * @returns The walk of $n$ values.
 *
 * @example The steps have the drift as their mean and sd as their spread
 * const w = randomWalk(stream(1), { n: 2000, sd: 0.5, drift: 0.1 })
 * const y = toArray(w.y)
 * const steps = y.slice(1).map((v, t) => v - y[t])
 * const m = steps.reduce((a, v) => a + v, 0) / steps.length
 * print('first values:', y.slice(0, 5))
 * const sd = Math.sqrt(steps.reduce((a, v) => a + (v - m) ** 2, 0) / steps.length)
 * print('mean step (0.1):', m, ' step sd (0.5):', sd)
 */
export function randomWalk(
  s: Stream,
  options: { n?: number; sd?: number; drift?: number; start?: number } = {},
): TimeSeries {
  const { n = 200, sd = 1, drift = 0, start = 0 } = options
  checkCount(n, 'randomWalk')
  const y = new Float64Array(n)
  let v = start
  for (let t = 0; t < n; t++) {
    if (t > 0) v += drift + sd * normal(s)
    y[t] = v
  }
  return series(y, {
    name: 'random walk',
    description: `${n} steps of a Gaussian random walk with step sd ${sd} and drift ${drift}.`,
    task: 'sequence',
    featureNames: ['y'],
    key: s.key,
  })
}

/** Options for `motifSeries`. */
export interface MotifSeriesOptions {
  /** Length of the series. Default 600. */
  n?: number
  /** The motif's length $m$, at least 2. Default 40. */
  m?: number
  /** Where the motif is planted (start indices). Default $[n/7, 2n/3]$, rounded. */
  motifs?: readonly number[]
  /** Where the discord (a fast burst seen nowhere else) is planted. Default $0.43n$, rounded. */
  discord?: number
  /** The random walk's step standard deviation. Default 0.3. */
  sd?: number
}

/** A series with known planted motif occurrences and one discord. */
export interface MotifSeries extends TimeSeries {
  /** The motif's length. */
  readonly m: number
  /** The start of every planted occurrence of the motif. */
  readonly motifs: readonly number[]
  /** The start of the discord, 30 samples long. */
  readonly discord: number
}

/**
 * A Gaussian random walk (step standard deviation `sd`, from 0) with one shape, 1.5 periods of a sine of amplitude 3
 * and length $m$, added at each of `motifs`, and a discord, six periods of a fast sine of amplitude 2.5 over 30
 * samples, added at `discord`: the test bed of matrix-profile motif and discord discovery (the shapes are found by
 * their z-normalised distances). A shape that runs past the end is cut short. The steps are drawn from `s` itself.
 *
 * @param s The stream the walk's steps are drawn from.
 * @param options The length, the motif's length and places, the discord's place and the step size
 *   (`MotifSeriesOptions`).
 * @returns The series, with the motif's length and the places of the motifs and the discord.
 *
 * @example The two occurrences are close after z-normalisation
 * const d = motifSeries(stream(1))
 * const y = toArray(d.y)
 * print('length:', y.length, ' motifs at', d.motifs, ' discord at', d.discord)
 * const z = (a) => {
 *   const w = y.slice(a, a + d.m)
 *   const mu = w.reduce((p, v) => p + v, 0) / w.length
 *   const sd = Math.sqrt(w.reduce((p, v) => p + (v - mu) ** 2, 0) / w.length)
 *   return w.map((v) => (v - mu) / sd)
 * }
 * const dist = (a, b) => Math.sqrt(z(a).reduce((p, v, i) => p + (v - z(b)[i]) ** 2, 0))
 * print('between the occurrences:', dist(d.motifs[0], d.motifs[1]), ' to the window at 300:', dist(d.motifs[0], 300))
 */
export function motifSeries(s: Stream, options: MotifSeriesOptions = {}): MotifSeries {
  const { n = 600, m = 40, sd = 0.3 } = options
  checkCount(n, 'motifSeries')
  const motifs = options.motifs ?? [Math.round(n / 7), Math.round((2 * n) / 3)]
  const discord = options.discord ?? Math.round(0.43 * n)
  const y = new Float64Array(n)
  let v = 0
  for (let t = 0; t < n; t++) {
    if (t > 0) v += sd * normal(s)
    y[t] = v
  }
  for (const at of motifs)
    for (let j = 0; j < m && at + j < n; j++) y[at + j] += 3 * Math.sin((3 * Math.PI * j) / (m - 1))
  for (let j = 0; j < 30 && discord + j < n; j++) y[discord + j] += 2.5 * Math.sin((12 * Math.PI * j) / 29)
  return {
    ...series(y, {
      name: 'planted motifs',
      description: `A random walk of ${n} steps with one shape of length ${m} planted at ${motifs.join(' and ')} and a burst at ${discord}.`,
      task: 'sequence',
      featureNames: ['y'],
      key: s.key,
    }),
    m,
    motifs,
    discord,
  }
}

// ── Registry ─────────────────────────────────────────────────────────────────────────────────────────────────────────

const dataset = definer<DatasetInfo>('dataset', 'data/synthetic')

dataset(
  {
    key: 'casino',
    name: 'Occasionally dishonest casino',
    summary: 'Die rolls from a two-state hidden Markov model: a fair die and a loaded one.',
    task: 'sequence',
    output: 'sequence',
    knobs: space({
      n: int(1, 10000, { default: 300 }),
      toLoaded: real(0, 1, { default: 0.05 }),
      toFair: real(0, 1, { default: 0.1 }),
      loadedSix: real(0.17, 1, { default: 0.5 }),
    }),
    truth: false,
    random: true,
    notes: ['occasionally-dishonest-casino', 'hidden-markov-model'],
  },
  casino,
)

dataset(
  {
    key: 'arSeries',
    name: 'AR series',
    summary: 'An autoregressive series started near stationarity (coefficients default [0.8]).',
    task: 'sequence',
    output: 'series',
    knobs: space({
      n: int(1, 100000, { default: 200 }),
      sd: real(0.01, 10, { default: 1 }),
      constant: real(-10, 10, { default: 0 }),
      burn: int(0, 5000, { default: 500 }),
    }),
    truth: false,
    random: true,
    notes: ['autoregressive-model'],
  },
  arSeries,
)

dataset(
  {
    key: 'seasonalSeries',
    name: 'Seasonal series',
    summary: 'Level, trend and harmonic seasonality plus white or AR(1) noise.',
    task: 'sequence',
    output: 'series',
    knobs: space({
      n: int(1, 10000, { default: 120 }),
      period: int(2, 365, { default: 12 }),
      level: real(-10, 10, { default: 0 }),
      trend: real(-1, 1, { default: 0.02 }),
      noise: real(0, 5, { default: 0.3 }),
      persistence: real(-0.99, 0.99, { default: 0 }),
    }),
    truth: false,
    random: true,
    notes: ['seasonal-autoregressive-integrated-moving-average'],
  },
  seasonalSeries,
)

dataset(
  {
    key: 'randomWalk',
    name: 'Random walk',
    summary: 'Cumulative Gaussian steps with an optional drift.',
    task: 'sequence',
    output: 'series',
    knobs: space({
      n: int(1, 100000, { default: 200 }),
      sd: real(0.01, 10, { default: 1 }),
      drift: real(-1, 1, { default: 0 }),
      start: real(-10, 10, { default: 0 }),
    }),
    truth: false,
    random: true,
    notes: ['white-noise-and-random-walk', 'random-walk'],
  },
  randomWalk,
)

dataset(
  {
    key: 'motifSeries',
    name: 'Planted motifs and a discord',
    summary: 'A random walk with one shape planted twice and a fast burst planted once.',
    task: 'sequence',
    output: 'series',
    knobs: space({
      n: int(100, 20000, { default: 600 }),
      m: int(8, 500, { default: 40 }),
      sd: real(0.01, 5, { default: 0.3 }),
    }),
    truth: false,
    random: true,
    notes: ['matrix-profile', 'motif-discovery-variants', 'discord-discovery-at-scale'],
  },
  motifSeries,
)
