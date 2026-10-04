/**
 * Seeded sequences: draws from a discrete hidden Markov model (and the occasionally dishonest casino), and synthetic
 * time series (autoregressive, seasonal with trend, random walk).
 */

import { normal, type Stream } from 'aifn-compute/foundation/random'
import { fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { hmm, sampleHmm } from 'aifn-methods/inference/sequence-models'
import { checkCount, labels, matrix, vector, type DatasetMeta } from '../types'
import type { DatasetInfo } from 'aifn-compute/foundation/contracts'
import { definer } from 'aifn-compute/foundation/registry'
import { int, real, space } from 'aifn-compute/foundation/space'
import { DomainError } from 'aifn-compute/foundation/errors'

/** A discrete hidden Markov model: initial distribution (k), transition matrix (k × k) and emission matrix (k × m). */
export interface DiscreteHmm {
  initial: Tensor
  transition: Tensor
  emission: Tensor
}

/** A draw from a hidden Markov model: observations `x` and hidden states `z` (int32, length n). */
export interface HmmSample {
  x: Tensor
  z: Tensor
  model: DiscreteHmm
  meta: DatasetMeta
}

function rows(m: readonly (readonly number[])[], what: string): number[][] {
  return m.map((r, i) => {
    const total = r.reduce((a, b) => a + b, 0)
    if (Math.abs(total - 1) > 1e-9) throw new DomainError(what, `${what}: row ${i} sums to ${total}, not 1`)
    return [...r]
  })
}

/**
 * n steps of a discrete hidden Markov model: z₁ ~ initial, z_t | z_{t−1} ~ transition[z_{t−1}], x_t | z_t ~
 * emission[z_t]. Observation symbols are 0, …, m − 1. Drawn by `aifn-methods/inference/sequence-models`' `sampleHmm`
 * (step t from `child(s, 'step', t)`).
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
  /** P(fair → loaded) per roll. Default 0.05. */
  toLoaded?: number
  /** P(loaded → fair) per roll. Default 0.1. */
  toFair?: number
  /** P(six | loaded); the other five faces share the rest equally. Default 0.5. */
  loadedSix?: number
}

/**
 * The occasionally dishonest casino (Durbin, Eddy, Krogh and Mitchison, 1998, "Biological Sequence Analysis", §3.2): a
 * fair die (state 0) and a loaded die (state 1) that shows six half the time. The chain starts from its stationary
 * distribution. Observations `x` are faces 1–6 (symbols 0–5 plus one); `model.emission` is indexed by face − 1.
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

/** A time series: times `t` and values `y` (length n), with a flag for divergence. */
export interface TimeSeries {
  t: Tensor
  y: Tensor
  /** True when the series overflowed to a non-finite value (e.g. a non-stationary AR model); values are kept as drawn. */
  diverged: boolean
  meta: DatasetMeta
}

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
  /** AR coefficients a₁, …, a_p in x_t = c + Σ a_k x_{t−k} + e_t. Default [0.8]. */
  coefficients?: readonly number[]
  n?: number
  /** Standard deviation of the driving noise e_t. Default 1. */
  sd?: number
  /** Constant c. Default 0. */
  constant?: number
  /** Initial steps drawn and discarded so that the series starts near stationarity. Default 500. */
  burn?: number
}

/**
 * An autoregressive AR(p) series x_t = c + Σ_k a_k x_{t−k} + e_t, e_t ~ N(0, sd²), started from zeros with a burn-in.
 * No clipping: an explosive model sets `diverged`.
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
  n?: number
  /** Period of the seasonal cycle in steps. Default 12. */
  period?: number
  /** Amplitude of each harmonic of the season, first harmonic first. Default [1]. */
  amplitudes?: readonly number[]
  /** Level at t = 0 and slope per step. Defaults 0 and 0.02. */
  level?: number
  trend?: number
  /** Standard deviation of white observation noise. Default 0.3. */
  noise?: number
  /** Coefficient of AR(1) noise instead of white noise (0 for white). Default 0. */
  persistence?: number
}

/**
 * A trend plus a seasonal cycle plus noise: y_t = level + trend·t + Σ_h A_h sin(2π h t / period) + u_t, where u_t is
 * white or AR(1) noise with marginal standard deviation `noise`.
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

/** A Gaussian random walk y_t = y_{t−1} + drift + sd·z_t from y_0 = start. */
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
  n?: number
  /** The motif's length m. Default 40. */
  m?: number
  /** Where the motif is planted (start indices). Default [n/7, 2n/3]. */
  motifs?: readonly number[]
  /** Where the discord (a fast burst seen nowhere else) is planted. Default 0.43n. */
  discord?: number
  /** The random walk's step sd. Default 0.3. */
  sd?: number
}

/** A series with known planted motif occurrences and one discord. */
export interface MotifSeries extends TimeSeries {
  readonly m: number
  readonly motifs: readonly number[]
  readonly discord: number
}

/**
 * A Gaussian random walk (step sd) with one shape, 1.5 periods of a sine of amplitude 3 and length m, added at each of
 * `motifs`, and a discord, three periods of a fast sine of amplitude 2.5 over 30 samples, added at `discord`: the test
 * bed of matrix-profile motif and discord discovery (the shapes are found by their z-normalised distances).
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
