/**
 * Metrics of predicted probabilities and predictive distributions: log loss, the Brier score and Murphy's
 * decomposition, the spherical score, calibration errors (ECE, MCE, RMS, debiased squared, classwise, confidence,
 * sweep) with reliability diagrams and consistency bars, CRPS (Gaussian and ensemble), the Gaussian log score,
 * interval scores, coverage and PIT values, and perplexity.
 */

import { type Stream, uniform } from 'aifn-compute/foundation/random'
import { normalCdf, normalLogPdf, normalPdf } from 'aifn-compute/numerics/special'
import { quantile } from 'aifn-compute/probability/stats'
import { isTraced, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import type { Curve, Distribution } from 'aifn-compute/foundation/contracts'
import {
  binaryTruth,
  classesOf,
  defineMetric,
  dense,
  divide,
  encodeLabels,
  isMatrixLike,
  labelList,
  meanOf,
  nonEmpty,
  sameLength,
  values,
  vector,
  type Data,
  type Label,
  type Labels,
  type Rows,
} from './core'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** Probabilities: a vector of P(positive) for a binary problem, or an n × K matrix whose columns follow `labels`. */
export type Probabilities = Data | Rows

type ProbabilityOptions = {
  /** Binary: the positive class. */
  positive?: Label
  /** Multiclass: the classes in the order of the probability columns; default the sorted labels. */
  labels?: readonly Label[]
}

/** The probability each case gave to its true class, and the full rows, for binary or multiclass input. */
function probabilityRows(yTrue: Labels, probabilities: Probabilities, options: ProbabilityOptions, what: string) {
  if (!isMatrixLike(probabilities)) {
    const { y } = binaryTruth(yTrue, options.positive)
    const p = values(probabilities as Data)
    sameLength(y, p, what)
    nonEmpty(y.length, what)
    const rows = new Float64Array(2 * y.length)
    for (let i = 0; i < y.length; i++) {
      rows[2 * i] = 1 - p[i]
      rows[2 * i + 1] = p[i]
    }
    return { n: y.length, K: 2, rows, index: Int32Array.from(y), binary: true }
  }
  const t = labelList(yTrue)
  const P = dense(probabilities as Rows, what)
  sameLength(t, { length: P.rows }, what)
  nonEmpty(t.length, what)
  const classes = options.labels ? [...options.labels] : classesOf(t)
  if (classes.length !== P.cols)
    throw new ShapeError('metrics', `metrics: ${what}: ${P.cols} probability columns for ${classes.length} classes`)
  const index = encodeLabels(t, classes)
  if (index.includes(-1)) throw new DomainError('metrics', `metrics: ${what}: a label is not among the classes`)
  return { n: t.length, K: P.cols, rows: P.data, index, binary: false }
}

/**
 * Log loss (cross-entropy), −(1/n) Σ log q_{i,yᵢ} in nats (log-loss-and-brier-score). Probabilities are not clipped:
 * a probability of 0 for an observed class gives +∞. Pass `eps` to clip to [eps, 1 − eps] as some libraries do.
 */
export const logLoss = defineMetric(
  {
    key: 'logLoss',
    stability: 'stable',
    name: 'Log loss',
    inputs: 'probabilities',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['log-loss-and-brier-score'],
    capability: 'predictive',
  },
  (yTrue: Labels, probabilities: Probabilities, options: ProbabilityOptions & { eps?: number } = {}): number => {
    const r = probabilityRows(yTrue, probabilities, options, 'logLoss')
    let s = 0
    for (let i = 0; i < r.n; i++) {
      let q = r.rows[i * r.K + r.index[i]]
      if (options.eps !== undefined) q = Math.min(1 - options.eps, Math.max(options.eps, q))
      s -= Math.log(q)
    }
    return s / r.n
  },
)

/**
 * The Brier score (Brier 1950). For a vector of binary probabilities, the mean squared error (1/n) Σ (pᵢ − yᵢ)² (the
 * form libraries report); for an n × K matrix, Brier's original (1/n) Σᵢ Σₖ (qᵢₖ − 1[yᵢ = k])², twice the binary form
 * for two classes (log-loss-and-brier-score).
 */
export const brierScore = defineMetric(
  {
    key: 'brierScore',
    stability: 'stable',
    name: 'Brier score',
    inputs: 'probabilities',
    direction: 'lower',
    range: [0, 2],
    notes: ['log-loss-and-brier-score'],
    capability: 'predictive',
  },
  (yTrue: Labels, probabilities: Probabilities, options: ProbabilityOptions = {}): number => {
    const r = probabilityRows(yTrue, probabilities, options, 'brierScore')
    let s = 0
    for (let i = 0; i < r.n; i++) {
      if (r.binary) s += (r.rows[2 * i + 1] - r.index[i]) ** 2
      else for (let k = 0; k < r.K; k++) s += (r.rows[i * r.K + k] - (k === r.index[i] ? 1 : 0)) ** 2
    }
    return s / r.n
  },
)

/**
 * The spherical score as a loss, 1 − q_y/‖q‖, averaged over cases (proper-scoring-rule): strictly proper by the
 * Cauchy–Schwarz inequality.
 */
export const sphericalScore = defineMetric(
  {
    key: 'sphericalScore',
    stability: 'stable',
    name: 'Spherical score (loss)',
    inputs: 'probabilities',
    direction: 'lower',
    range: [0, 1],
    notes: ['proper-scoring-rule'],
    capability: 'predictive',
  },
  (yTrue: Labels, probabilities: Probabilities, options: ProbabilityOptions = {}): number => {
    const r = probabilityRows(yTrue, probabilities, options, 'sphericalScore')
    let s = 0
    for (let i = 0; i < r.n; i++) {
      let norm = 0
      for (let k = 0; k < r.K; k++) norm += r.rows[i * r.K + k] ** 2
      s += 1 - r.rows[i * r.K + r.index[i]] / Math.sqrt(norm)
    }
    return s / r.n
  },
)

// ── Binning and reliability ──────────────────────────────────────────────────────────────────────────────────────────

/** How predictions are grouped: equal-width bins on [0, 1], or equal-mass bins holding (nearly) equal counts. */
export type BinStrategy = 'uniform' | 'quantile'

/**
 * A reliability diagram as a binned `Curve` (no thresholds): per bin, `x` the mean prediction p̄ₘ against `y` the
 * observed frequency ȳₘ of the event. Empty bins hold NaN.
 */
export type ReliabilityDiagram = Curve<'reliability'> & {
  /** Bin edges (length M + 1). Equal-mass bins report the smallest prediction of each bin and 1 as the last edge. */
  readonly edges: Tensor
  /** Cases in each bin. */
  readonly counts: Tensor
  /** ȳₘ − p̄ₘ. */
  readonly gap: Tensor
  /** The binned ECE, Σ (nₘ/n)|ȳₘ − p̄ₘ|. */
  readonly ece: number
}

/** Bin index of each prediction, and the bin edges. */
function binAssignments(p: Float64Array, bins: number, strategy: BinStrategy): { bin: Int32Array; edges: number[] } {
  if (!(bins >= 1 && Number.isInteger(bins)))
    throw new DomainError('metrics', 'metrics: bins must be a positive integer')
  const n = p.length
  const bin = new Int32Array(n)
  if (strategy === 'uniform') {
    // Bins [k/M, (k+1)/M), the last closed at 1 (the notes' convention).
    for (let i = 0; i < n; i++) {
      if (p[i] < 0 || p[i] > 1) throw new DomainError('metrics', `metrics: probability ${p[i]} lies outside [0, 1]`)
      bin[i] = Math.min(bins - 1, Math.floor(p[i] * bins))
    }
    return { bin, edges: Array.from({ length: bins + 1 }, (_, k) => k / bins) }
  }
  // Equal mass: sort, then give ranks floor(r·M/n) (Nixon et al. 2019).
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b] || a - b)
  const edges: number[] = []
  order.forEach((i, r) => {
    bin[i] = Math.floor((r * bins) / n)
    if (edges.length === bin[i]) edges.push(p[i])
  })
  while (edges.length < bins) edges.push(NaN)
  edges.push(1)
  return { bin, edges }
}

/** Per-bin counts, mean predictions and event frequencies. */
function binStatistics(y: ArrayLike<number>, p: Float64Array, bins: number, strategy: BinStrategy) {
  const { bin, edges } = binAssignments(p, bins, strategy)
  const counts = new Float64Array(bins)
  const sumP = new Float64Array(bins)
  const sumY = new Float64Array(bins)
  for (let i = 0; i < p.length; i++) {
    counts[bin[i]]++
    sumP[bin[i]] += p[i]
    sumY[bin[i]] += y[i]
  }
  const meanP = Float64Array.from(sumP, (s, m) => divide(s, counts[m]))
  const freq = Float64Array.from(sumY, (s, m) => divide(s, counts[m]))
  return { counts, meanP, freq, edges, n: p.length, bin }
}

type CalibrationOptions = {
  /** Number of bins M (default 10). */
  bins?: number
  /** `uniform` (default) or `quantile` (equal-mass). */
  strategy?: BinStrategy
  /** The positive class. */
  positive?: Label
}

/**
 * The reliability diagram of binary probabilities (reliability-diagrams-and-consistency-bars; Murphy and Winkler
 * 1977): predictions binned into M bins, and for each bin the mean prediction p̄ₘ and the fraction of positives ȳₘ.
 */
export function reliabilityDiagram(
  yTrue: Labels,
  probabilities: Data,
  options: CalibrationOptions = {},
): ReliabilityDiagram {
  const { y } = binaryTruth(yTrue, options.positive)
  const p = values(probabilities)
  sameLength(y, p, 'reliabilityDiagram')
  nonEmpty(y.length, 'reliabilityDiagram')
  const s = binStatistics(y, p, options.bins ?? 10, options.strategy ?? 'uniform')
  let ece = 0
  s.counts.forEach((c, m) => {
    if (c > 0) ece += (c / s.n) * Math.abs(s.freq[m] - s.meanP[m])
  })
  return {
    kind: 'curve',
    curve: 'reliability',
    x: vector(s.meanP),
    y: vector(s.freq),
    edges: vector(s.edges),
    counts: vector(s.counts),
    gap: vector(Float64Array.from(s.freq, (f, m) => f - s.meanP[m])),
    ece,
  }
}

/** Binned calibration error of binary probabilities with an Lʳ norm over bins (r = 1: ECE, 2: RMS, ∞: MCE). */
function binnedCalibrationError(y: ArrayLike<number>, p: Float64Array, o: CalibrationOptions, norm: 1 | 2 | 'max') {
  const s = binStatistics(y, p, o.bins ?? 10, o.strategy ?? 'uniform')
  let total = 0
  s.counts.forEach((c, m) => {
    if (c === 0) return
    const gap = Math.abs(s.freq[m] - s.meanP[m])
    if (norm === 'max') total = Math.max(total, gap)
    else total += (c / s.n) * gap ** norm
  })
  return norm === 2 ? Math.sqrt(total) : total
}

function binaryInputs(yTrue: Labels, probabilities: Data, positive: Label | undefined, what: string) {
  const { y } = binaryTruth(yTrue, positive)
  const p = values(probabilities)
  sameLength(y, p, what)
  nonEmpty(y.length, what)
  return { y, p }
}

const calibrationInfo = (key: string, name: string, note = 'calibration-error') =>
  ({
    key,
    stability: 'stable',
    name,
    inputs: 'probabilities',
    direction: 'lower',
    range: [0, 1],
    notes: [note],
    capability: 'predictive',
  }) as const

/**
 * The expected calibration error of binary probabilities, Σₘ (|Bₘ|/n)|ȳ(Bₘ) − p̄(Bₘ)| over M bins (Naeini et al.
 * 2015). Depends on the binning: report `bins` and `strategy`.
 */
export const expectedCalibrationError = defineMetric(
  calibrationInfo('expectedCalibrationError', 'Expected calibration error'),
  (yTrue: Labels, probabilities: Data, options: CalibrationOptions = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'expectedCalibrationError')
    return binnedCalibrationError(y, p, options, 1)
  },
)

/** The maximum calibration error: the largest bin gap maxₘ |ȳₘ − p̄ₘ| over non-empty bins (Naeini et al. 2015). */
export const maximumCalibrationError = defineMetric(
  calibrationInfo('maximumCalibrationError', 'Maximum calibration error'),
  (yTrue: Labels, probabilities: Data, options: CalibrationOptions = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'maximumCalibrationError')
    return binnedCalibrationError(y, p, options, 'max')
  },
)

/** The RMS (L²) calibration error, √(Σₘ (nₘ/n)(ȳₘ − p̄ₘ)²) (Kumar et al. 2019), the plug-in CE₂. */
export const rmsCalibrationError = defineMetric(
  calibrationInfo('rmsCalibrationError', 'RMS calibration error', 'estimating-calibration-error'),
  (yTrue: Labels, probabilities: Data, options: CalibrationOptions = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'rmsCalibrationError')
    return binnedCalibrationError(y, p, options, 2)
  },
)

/**
 * The debiased squared calibration error (Kumar et al. 2019; estimating-calibration-error):
 * Σₘ (nₘ/n)[(ȳₘ − p̄ₘ)² − ȳₘ(1 − ȳₘ)/(nₘ − 1)], an estimate of CE₂² with the binomial noise of each bin subtracted.
 * It can be negative; bins with one case contribute nothing to the correction. Default equal-mass bins.
 */
export const debiasedSquaredCalibrationError = defineMetric(
  {
    ...calibrationInfo(
      'debiasedSquaredCalibrationError',
      'Debiased squared calibration error',
      'estimating-calibration-error',
    ),
    range: [-1, 1],
  },
  (yTrue: Labels, probabilities: Data, options: CalibrationOptions = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'debiasedSquaredCalibrationError')
    const s = binStatistics(y, p, options.bins ?? 10, options.strategy ?? 'quantile')
    let total = 0
    s.counts.forEach((c, m) => {
      if (c === 0) return
      const noise = c > 1 ? (s.freq[m] * (1 - s.freq[m])) / (c - 1) : 0
      total += (c / s.n) * ((s.freq[m] - s.meanP[m]) ** 2 - noise)
    })
    return total
  },
)

/**
 * ECE_sweep (Roelofs et al. 2022): equal-mass bins, with the largest number of bins (up to `maxBins`, default n) for
 * which the bin frequencies are still non-decreasing.
 */
export const sweepCalibrationError = defineMetric(
  calibrationInfo('sweepCalibrationError', 'ECE sweep', 'estimating-calibration-error'),
  (yTrue: Labels, probabilities: Data, options: { positive?: Label; maxBins?: number } = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'sweepCalibrationError')
    let best = 1
    for (let m = 1; m <= (options.maxBins ?? p.length); m++) {
      const s = binStatistics(y, p, m, 'quantile')
      let monotone = true
      for (let k = 1; k < m; k++) if (s.freq[k] < s.freq[k - 1]) monotone = false
      if (!monotone) break
      best = m
    }
    return binnedCalibrationError(y, p, { bins: best, strategy: 'quantile' }, 1)
  },
)

/**
 * Confidence (top-label) ECE for multiclass probabilities (Guo et al. 2017): bin the largest probability of each case
 * and compare it with the accuracy of the predicted class in each bin.
 */
export const confidenceCalibrationError = defineMetric(
  calibrationInfo('confidenceCalibrationError', 'Confidence ECE'),
  (yTrue: Labels, probabilities: Rows, options: CalibrationOptions & { labels?: readonly Label[] } = {}): number => {
    const r = probabilityRows(yTrue, probabilities, options, 'confidenceCalibrationError')
    const conf = new Float64Array(r.n)
    const correct = new Float64Array(r.n)
    for (let i = 0; i < r.n; i++) {
      let best = 0
      for (let k = 1; k < r.K; k++) if (r.rows[i * r.K + k] > r.rows[i * r.K + best]) best = k
      conf[i] = r.rows[i * r.K + best]
      correct[i] = best === r.index[i] ? 1 : 0
    }
    return binnedCalibrationError(correct, conf, options, 1)
  },
)

/**
 * Classwise ECE (Kull et al. 2019): the binary ECE of each class's probability column against "is this class",
 * averaged over the K classes.
 */
export const classwiseCalibrationError = defineMetric(
  calibrationInfo('classwiseCalibrationError', 'Classwise ECE'),
  (yTrue: Labels, probabilities: Rows, options: CalibrationOptions & { labels?: readonly Label[] } = {}): number => {
    const r = probabilityRows(yTrue, probabilities, options, 'classwiseCalibrationError')
    let s = 0
    for (let k = 0; k < r.K; k++) {
      const y = Float64Array.from(r.index, (v) => (v === k ? 1 : 0))
      const p = Float64Array.from({ length: r.n }, (_, i) => r.rows[i * r.K + k])
      s += binnedCalibrationError(y, p, options, 1)
    }
    return s / r.K
  },
)

/**
 * Consistency bars (Bröcker and Smith 2007): for each bin of a reliability diagram, the central `level` interval
 * (default 0.9) of the observed frequency that a calibrated model would produce, from `resamples` (default 1000)
 * redraws of every label from Bernoulli(pᵢ) with the stream.
 */
export function consistencyBars(
  s: Stream,
  probabilities: Data,
  options: { bins?: number; strategy?: BinStrategy; resamples?: number; level?: number } = {},
): { lower: Tensor; upper: Tensor; meanPredicted: Tensor } {
  const p = values(probabilities)
  const bins = options.bins ?? 10
  const { bin } = binAssignments(p, bins, options.strategy ?? 'uniform')
  const resamples = options.resamples ?? 1000
  const level = options.level ?? 0.9
  const counts = new Float64Array(bins)
  const sumP = new Float64Array(bins)
  for (let i = 0; i < p.length; i++) {
    counts[bin[i]]++
    sumP[bin[i]] += p[i]
  }
  const draws = Array.from({ length: bins }, () => new Float64Array(resamples))
  for (let r = 0; r < resamples; r++) {
    const hits = new Float64Array(bins)
    for (let i = 0; i < p.length; i++) if (uniform(s) < p[i]) hits[bin[i]]++
    for (let m = 0; m < bins; m++) draws[m][r] = divide(hits[m], counts[m])
  }
  const lo = (1 - level) / 2
  const q = (m: number, at: number) => (counts[m] > 0 ? quantile(draws[m], at) : NaN)
  return {
    lower: vector(Array.from({ length: bins }, (_, m) => q(m, lo))),
    upper: vector(Array.from({ length: bins }, (_, m) => q(m, 1 - lo))),
    meanPredicted: vector(Float64Array.from(sumP, (v, m) => divide(v, counts[m]))),
  }
}

/** Murphy's decomposition of the binary Brier score. */
export type BrierDecomposition = {
  brier: number
  reliability: number
  resolution: number
  uncertainty: number
  /**
   * brier − (reliability − resolution + uncertainty): zero when the forecasts are grouped by their distinct values,
   * and the within-bin variance terms when bins pool different forecasts.
   */
  residual: number
}

/**
 * Murphy's decomposition of the binary Brier score (Murphy 1973; log-loss-and-brier-score):
 * Brier = reliability − resolution + uncertainty, with reliability (1/n) Σ_b n_b(f_b − ō_b)², resolution
 * (1/n) Σ_b n_b(ō_b − ō)² and uncertainty ō(1 − ō). Without `bins`, forecasts are grouped by distinct value and the
 * identity is exact; with `bins`, f_b is each bin's mean forecast and `residual` reports what the bins leave out.
 */
export function brierDecomposition(
  yTrue: Labels,
  probabilities: Data,
  options: { bins?: number; positive?: Label } = {},
): BrierDecomposition {
  const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'brierDecomposition')
  const n = y.length
  const groups = new Map<number, { n: number; f: number; o: number }>()
  for (let i = 0; i < n; i++) {
    const key = options.bins ? Math.min(options.bins - 1, Math.floor(p[i] * options.bins)) : p[i]
    const g = groups.get(key) ?? { n: 0, f: 0, o: 0 }
    g.n++
    g.f += p[i]
    g.o += y[i]
    groups.set(key, g)
  }
  const base = meanOf(y)
  let reliability = 0
  let resolution = 0
  for (const g of groups.values()) {
    reliability += g.n * (g.f / g.n - g.o / g.n) ** 2
    resolution += g.n * (g.o / g.n - base) ** 2
  }
  reliability /= n
  resolution /= n
  let brier = 0
  for (let i = 0; i < n; i++) brier += (p[i] - y[i]) ** 2
  brier /= n
  const uncertainty = base * (1 - base)
  return { brier, reliability, resolution, uncertainty, residual: brier - (reliability - resolution + uncertainty) }
}

// ── Predictive distributions ─────────────────────────────────────────────────────────────────────────────────────────

/** A value per case, or one value shared by all cases. */
export type PerCase = Data | number

function perCase(x: PerCase | Value, n: number, what: string): Float64Array {
  if (typeof x === 'number') return new Float64Array(n).fill(x)
  if (isTraced(x)) throw new DomainError('metrics', `metrics: ${what}: traced values are not accepted`)
  const v = values(x as Data)
  if (v.length !== n) throw new ShapeError('metrics', `metrics: ${what}: ${v.length} values for ${n} cases`)
  return v
}

/**
 * A Gaussian forecast per case: means and standard deviations, or a model's predictive `Normal` distribution (the
 * `predictive` capability, as `evaluate` passes it), batch shape [n].
 */
export type GaussianForecast = { mean: PerCase; sd: PerCase } | Distribution

const isDistribution = (x: unknown): x is Distribution =>
  typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'distribution'

/** The means and standard deviations of a Gaussian forecast for n cases; a predictive must be a `Normal`. */
function gaussianMoments(f: GaussianForecast, n: number, what: string): { mu: Float64Array; sd: Float64Array } {
  if (isDistribution(f)) {
    if (f.name !== 'Normal')
      throw new DomainError(
        'metrics',
        `metrics: ${what}: needs a Normal predictive, got ${f.name} (use logScore for any distribution)`,
      )
    return { mu: perCase(f.mean(), n, `${what} mean`), sd: perCase(f.stddev(), n, `${what} sd`) }
  }
  return { mu: perCase(f.mean, n, `${what} mean`), sd: perCase(f.sd, n, `${what} sd`) }
}

/**
 * The log score of any predictive distribution, −(1/n) Σ log p(yᵢ) (a density for continuous predictives, a mass for
 * discrete ones), in nats: the proper scoring rule behind log loss and the Gaussian log score (Gneiting and Raftery
 * 2007, §4.1), read from the model's `predictive` through `logProb`. The predictive's batch is the n cases; for a
 * categorical predictive, y holds class indices.
 */
export const logScore = defineMetric(
  {
    key: 'logScore',
    stability: 'stable',
    name: 'Log score',
    inputs: 'distribution',
    direction: 'lower',
    range: [-Infinity, Infinity],
    notes: ['continuous-ranked-probability-score-and-interval-scores', 'log-loss-and-brier-score'],
    capability: 'predictive',
  },
  (yTrue: Data, predictive: Distribution): number => {
    if (!isDistribution(predictive))
      throw new DomainError('metrics', 'metrics: logScore: needs a predictive distribution')
    const y = values(yTrue)
    nonEmpty(y.length, 'logScore')
    const lp = perCase(predictive.logProb(vector(y)), y.length, 'logScore')
    let s = 0
    for (let i = 0; i < y.length; i++) s -= lp[i]
    return s / y.length
  },
)

/**
 * The CRPS of Gaussian forecasts N(μᵢ, σᵢ²), averaged over cases (Gneiting and Raftery 2007, eq. 21): with
 * z = (y − μ)/σ, CRPS = σ(z(2Φ(z) − 1) + 2φ(z) − 1/√π). In the units of y. The forecast is means and sds, or a
 * model's Normal predictive (so `evaluate` serves it).
 */
export const crpsGaussian = defineMetric(
  {
    key: 'crpsGaussian',
    stability: 'stable',
    name: 'CRPS (Gaussian forecast)',
    inputs: 'distribution',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['continuous-ranked-probability-score-and-interval-scores'],
    capability: 'predictive',
  },
  (yTrue: Data, forecast: GaussianForecast): number => {
    const y = values(yTrue)
    nonEmpty(y.length, 'crpsGaussian')
    const { mu, sd } = gaussianMoments(forecast, y.length, 'crpsGaussian')
    let s = 0
    for (let i = 0; i < y.length; i++) {
      const z = (y[i] - mu[i]) / sd[i]
      s += sd[i] * (z * (2 * normalCdf(z) - 1) + 2 * normalPdf(z) - 1 / Math.sqrt(Math.PI))
    }
    return s / y.length
  },
)

/**
 * The CRPS of ensemble or sample forecasts in the kernel form E|X − y| − ½E|X − X'| (Gneiting and Raftery 2007),
 * averaged over cases. `samples` is an n × m matrix (m draws per case) or, for one case, a vector. The plug-in form
 * divides the spread term by m²; `fair: true` divides by m(m − 1), which is unbiased for the underlying distribution.
 * Computed in O(m log m) per case from the sorted draws.
 */
export const crpsEnsemble = defineMetric(
  {
    key: 'crpsEnsemble',
    stability: 'stable',
    name: 'CRPS (ensemble forecast)',
    inputs: 'distribution',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['continuous-ranked-probability-score-and-interval-scores'],
    capability: 'predictive',
  },
  (yTrue: Data | number, samples: Data | Rows, options: { fair?: boolean } = {}): number => {
    const y = typeof yTrue === 'number' ? Float64Array.of(yTrue) : values(yTrue)
    const S = isMatrixLike(samples)
      ? dense(samples as Rows, 'crpsEnsemble')
      : { rows: 1, cols: values(samples as Data).length, data: values(samples as Data) }
    if (S.rows !== y.length)
      throw new ShapeError('metrics', `metrics: crpsEnsemble: ${S.rows} ensembles for ${y.length} observations`)
    const m = S.cols
    let total = 0
    for (let i = 0; i < y.length; i++) {
      const x = Float64Array.from(S.data.subarray(i * m, (i + 1) * m)).sort()
      let absError = 0
      let spread = 0
      // Σ_{j,k} |x_j − x_k| = 2 Σ_k (2k − m + 1) x_(k) for sorted x (0-based k).
      for (let k = 0; k < m; k++) {
        absError += Math.abs(x[k] - y[i])
        spread += (2 * k - m + 1) * x[k]
      }
      spread *= 2
      total += absError / m - (0.5 * spread) / (options.fair ? m * (m - 1) : m * m)
    }
    return total / y.length
  },
)

/**
 * The negative log predictive density of Gaussian forecasts, −(1/n) Σ log N(yᵢ; μᵢ, σᵢ²) (the log score, orientated
 * as a loss). The forecast is means and sds, or a model's Normal predictive (so `evaluate` serves it); `logScore`
 * is the same score for any predictive distribution.
 */
export const gaussianLogScore = defineMetric(
  {
    key: 'gaussianLogScore',
    stability: 'stable',
    name: 'Log score (Gaussian forecast)',
    inputs: 'distribution',
    direction: 'lower',
    range: [-Infinity, Infinity],
    notes: ['continuous-ranked-probability-score-and-interval-scores'],
    capability: 'predictive',
  },
  (yTrue: Data, forecast: GaussianForecast): number => {
    const y = values(yTrue)
    nonEmpty(y.length, 'gaussianLogScore')
    const { mu, sd } = gaussianMoments(forecast, y.length, 'gaussianLogScore')
    let s = 0
    for (let i = 0; i < y.length; i++) s -= normalLogPdf((y[i] - mu[i]) / sd[i]) - Math.log(sd[i])
    return s / y.length
  },
)

/** A central prediction interval per case. */
export type Interval = { lower: PerCase; upper: PerCase }

/**
 * The interval score of central (1 − α) prediction intervals, averaged over cases (Gneiting and Raftery 2007, §6.2):
 * (u − ℓ) + (2/α)(ℓ − y)·1[y < ℓ] + (2/α)(y − u)·1[y > u].
 */
export const intervalScore = defineMetric(
  {
    key: 'intervalScore',
    stability: 'stable',
    name: 'Interval score',
    inputs: 'distribution',
    direction: 'lower',
    range: [0, Infinity],
    notes: ['continuous-ranked-probability-score-and-interval-scores'],
    capability: 'predictive',
  },
  (yTrue: Data, interval: Interval, options: { alpha: number }): number => {
    const y = values(yTrue)
    nonEmpty(y.length, 'intervalScore')
    const lo = perCase(interval.lower, y.length, 'intervalScore lower')
    const hi = perCase(interval.upper, y.length, 'intervalScore upper')
    const a = options.alpha
    let s = 0
    for (let i = 0; i < y.length; i++) {
      s += hi[i] - lo[i]
      if (y[i] < lo[i]) s += (2 / a) * (lo[i] - y[i])
      if (y[i] > hi[i]) s += (2 / a) * (y[i] - hi[i])
    }
    return s / y.length
  },
)

/** Coverage: the fraction of observations inside their interval [ℓ, u] (report beside a score, not as one). */
export const coverage = defineMetric(
  {
    key: 'coverage',
    stability: 'stable',
    name: 'Interval coverage',
    inputs: 'distribution',
    direction: 'higher',
    range: [0, 1],
    notes: ['continuous-ranked-probability-score-and-interval-scores'],
    capability: 'predictive',
  },
  (yTrue: Data, interval: Interval): number => {
    const y = values(yTrue)
    nonEmpty(y.length, 'coverage')
    const lo = perCase(interval.lower, y.length, 'coverage lower')
    const hi = perCase(interval.upper, y.length, 'coverage upper')
    let inside = 0
    for (let i = 0; i < y.length; i++) if (y[i] >= lo[i] && y[i] <= hi[i]) inside++
    return inside / y.length
  },
)

/**
 * Probability integral transform values uᵢ = Fᵢ(yᵢ) (quantile-calibration): uniform on [0, 1] for a calibrated
 * forecaster. Gaussian forecasts give Φ((y − μ)/σ); ensembles (an n × m matrix of draws) give the fraction of draws
 * ≤ y, or with `stream` the randomised PIT that spreads ties uniformly between the fractions below and at y.
 */
export function pitValues(
  yTrue: Data,
  forecast: { mean: PerCase; sd: PerCase } | { samples: Rows },
  options: { stream?: Stream } = {},
): Tensor {
  const y = values(yTrue)
  if ('mean' in forecast) {
    const mu = perCase(forecast.mean, y.length, 'pitValues mean')
    const sd = perCase(forecast.sd, y.length, 'pitValues sd')
    return vector(Float64Array.from(y, (v, i) => normalCdf((v - mu[i]) / sd[i])))
  }
  const S = dense(forecast.samples, 'pitValues samples')
  if (S.rows !== y.length) throw new ShapeError('metrics', 'metrics: pitValues: one row of samples per observation')
  return vector(
    Float64Array.from(y, (v, i) => {
      let below = 0
      let equal = 0
      for (let k = 0; k < S.cols; k++) {
        const x = S.data[i * S.cols + k]
        if (x < v) below++
        else if (x === v) equal++
      }
      const u = options.stream ? uniform(options.stream) : 1
      return (below + u * equal) / S.cols
    }),
  )
}

/**
 * Perplexity of held-out tokens from their log-probabilities (natural log) under the model,
 * exp(−(1/T) Σ log p(xₜ | x₍<t₎)) (perplexity-as-a-metric).
 */
export const perplexity = defineMetric(
  {
    key: 'perplexity',
    stability: 'stable',
    name: 'Perplexity',
    inputs: 'probabilities',
    direction: 'lower',
    range: [1, Infinity],
    notes: ['perplexity-as-a-metric'],
    capability: 'predictive',
  },
  (logProbabilities: Data): number => {
    const l = values(logProbabilities)
    nonEmpty(l.length, 'perplexity')
    return Math.exp(-meanOf(l))
  },
)
