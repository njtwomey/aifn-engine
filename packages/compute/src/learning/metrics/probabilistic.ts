/**
 * Metrics of predicted probabilities and predictive distributions: log loss, the Brier score and Murphy's
 * decomposition, the spherical score, calibration errors (ECE, MCE, RMS, debiased squared, classwise, confidence,
 * sweep) with reliability diagrams and consistency bars, CRPS (Gaussian and ensemble), the log score of any
 * predictive and of a Gaussian forecast, interval scores, coverage and PIT values, and perplexity.
 *
 * Probabilities are a vector of $\pr(\text{positive})$ for a binary problem, or an $n \times K$ matrix whose columns
 * follow the sorted classes (or `labels`). The scores are losses, averaged over cases, and are proper scoring rules
 * (Gneiting and Raftery, 2007); the calibration errors are not, and depend on the binning, so report `bins` and
 * `strategy` with them. Logarithms are natural, so log scores are in nats. Inputs of different lengths throw
 * `ShapeError`, and an empty input `DomainError`.
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

/**
 * Probabilities: a vector of $\pr(\text{positive})$ for a binary problem, or an $n \times K$ matrix whose columns
 * follow `labels`.
 */
export type Probabilities = Data | Rows

/** Options shared by the metrics of class probabilities. */
type ProbabilityOptions = {
  /** Binary: the positive class. */
  positive?: Label
  /** Multiclass: the classes in the order of the probability columns; default the sorted labels. */
  labels?: readonly Label[]
}

/**
 * The probability each case gave to its true class, and the full rows, for binary or multiclass input. A binary vector
 * $p$ becomes rows $(1 - p, p)$. Throws `ShapeError` for mismatched lengths or a column count other than the number of
 * classes, and `DomainError` for an empty input or a label not among the classes.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, or an $n \times K$ matrix of class probabilities.
 * @param options The positive class (binary) or the classes in column order (matrix).
 * @param what The caller's name, for error messages.
 * @returns `n` cases, `K` classes, the row-major $n \times K$ `rows`, each case's true-class `index`, and whether the
 *   input was `binary`.
 */
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
 * Log loss (cross-entropy), $-\frac{1}{n} \sum_i \log q_{i, y_i}$ in nats (log-loss-and-brier-score), as
 * scikit-learn's `log_loss`. Probabilities are not clipped: a probability of 0 for an observed class gives
 * $+\infty$. Pass `eps` to clip to $[\epsilon, 1 - \epsilon]$ as some libraries do. Rows are not renormalised.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, or an $n \times K$ matrix of class probabilities.
 * @param options `positive` (binary), `labels` (the classes in column order), and `eps`, the clipping $\epsilon$.
 * @returns The mean negative log probability of the true class, in nats.
 *
 * @example Class probabilities (the scikit-learn example)
 * print(logLoss(['spam', 'ham', 'ham', 'spam'], [[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]]))
 *
 * @example A certain mistake costs infinitely much, unless clipped
 * print('binary =', logLoss([1, 0], [0.9, 0.2]))
 * print('certain and wrong =', logLoss([1, 0], [0.9, 1]))
 * print('clipped at 1e-15 =', logLoss([1, 0], [0.9, 1], { eps: 1e-15 }))
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
 * The Brier score (Brier, 1950). For a vector of binary probabilities, the mean squared error
 * $\frac{1}{n} \sum_i (p_i - y_i)^2$ (the form libraries report, as scikit-learn's `brier_score_loss`); for an
 * $n \times K$ matrix, Brier's original $\frac{1}{n} \sum_i \sum_k (q_{ik} - \indicator[y_i = k])^2$, twice the binary
 * form for two classes (log-loss-and-brier-score).
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, or an $n \times K$ matrix of class probabilities.
 * @param options `positive` (binary) or `labels` (the classes in column order).
 * @returns The Brier score: in $[0, 1]$ for a vector, $[0, 2]$ for a matrix.
 *
 * @example Binary (the scikit-learn example), and the same forecasts as two columns
 * print('vector =', brierScore([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3]))
 * print('matrix =', brierScore([0, 1, 1, 0], [[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]))
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
 * The spherical score as a loss, $1 - q_y / \lVert \qvec \rVert$, averaged over cases (proper-scoring-rule): strictly
 * proper by the Cauchy–Schwarz inequality.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, or an $n \times K$ matrix of class probabilities.
 * @param options `positive` (binary) or `labels` (the classes in column order).
 * @returns The mean loss, in $[0, 1]$: 0 for certainty in the true class.
 *
 * @example Certain, confident and unsure forecasts of the positive class
 * print('certain =', sphericalScore([1], [1]))
 * print('0.8 =', sphericalScore([1], [0.8]))
 * print('0.5 =', sphericalScore([1], [0.5]))
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
 * A reliability diagram as a binned `Curve` (no thresholds): per bin $m$, `x` the mean prediction $\bar{p}_m$
 * against `y` the observed frequency $\bar{y}_m$ of the event. Empty bins hold NaN.
 */
export type ReliabilityDiagram = Curve<'reliability'> & {
  /**
   * Bin edges ($M + 1$ values). Equal-mass bins report the smallest prediction of each bin and 1 as the last edge
   * (NaN for an empty bin).
   */
  readonly edges: Tensor
  /** Cases in each bin. */
  readonly counts: Tensor
  /** $\bar{y}_m - \bar{p}_m$ per bin. */
  readonly gap: Tensor
  /** The binned ECE, $\sum_m (n_m / n) \lvert \bar{y}_m - \bar{p}_m \rvert$. */
  readonly ece: number
}

/**
 * Bin index of each prediction, and the bin edges. A `bins` that is not a positive integer, or with `uniform` bins a
 * prediction outside $[0, 1]$, throws `DomainError`.
 *
 * @param p The predictions.
 * @param bins The number of bins $M$.
 * @param strategy `uniform`: bins $[k/M, (k + 1)/M)$, the last closed at 1. `quantile`: the case of rank $r$ (ties by
 *   position) goes to bin $\lfloor rM/n \rfloor$.
 * @returns The bin of each prediction, and the $M + 1$ edges.
 */
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

/**
 * Per-bin counts, mean predictions and event frequencies (NaN for an empty bin).
 *
 * @param y The outcomes, 0/1 per case.
 * @param p The predictions, one per case.
 * @param bins The number of bins $M$.
 * @param strategy How the predictions are binned.
 * @returns The `counts`, mean prediction `meanP` and frequency `freq` of each bin, the `edges`, the number of cases
 *   `n` and the `bin` of each case.
 */
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

/** Options of the binned calibration errors of binary probabilities. */
type CalibrationOptions = {
  /** Number of bins $M$ (default 10). */
  bins?: number
  /** `uniform` (default) or `quantile` (equal-mass). */
  strategy?: BinStrategy
  /** The positive class. */
  positive?: Label
}

/**
 * The reliability diagram of binary probabilities (reliability-diagrams-and-consistency-bars; Murphy and Winkler
 * 1977): predictions binned into $M$ bins, and for each bin the mean prediction $\bar{p}_m$ and the fraction of
 * positives $\bar{y}_m$.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, each in $[0, 1]$.
 * @param options `bins`, the number of bins $M$ (default 10); `strategy`, `uniform` (default) or `quantile`; and
 *   `positive`, the positive class.
 * @returns The diagram: mean predictions against frequencies, with the edges, counts and gaps of the bins and the
 *   binned ECE.
 *
 * @example Two bins
 * const yTrue = [0, 0, 1, 0, 1, 1, 1, 1]
 * const probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * const d = reliabilityDiagram(yTrue, probs, { bins: 2 })
 * print('mean prediction =', d.x, ' frequency =', d.y)
 * print('counts =', d.counts, ' ECE =', d.ece)
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

/**
 * Binned calibration error of binary probabilities with an $L^r$ norm over bins ($r = 1$: ECE, 2: RMS, $\infty$:
 * MCE). Empty bins are skipped.
 *
 * @param y The outcomes, 0/1 per case.
 * @param p The predictions, one per case.
 * @param o The binning: `bins` (default 10) and `strategy` (default `uniform`).
 * @param norm The norm $r$: 1, 2 or `max`.
 * @returns The calibration error.
 */
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

/**
 * Binary truth and predictions, checked to have the same non-zero length.
 *
 * @param yTrue The true labels.
 * @param probabilities The predicted probabilities of the positive class.
 * @param positive The positive class; left out, chosen by `positiveOf`.
 * @param what The caller's name, for error messages.
 * @returns The 0/1 truth `y` and the predictions `p`.
 */
function binaryInputs(yTrue: Labels, probabilities: Data, positive: Label | undefined, what: string) {
  const { y } = binaryTruth(yTrue, positive)
  const p = values(probabilities)
  sameLength(y, p, what)
  nonEmpty(y.length, what)
  return { y, p }
}

/**
 * The registry metadata shared by the calibration errors: stable, of probabilities, lower is better, range $[0, 1]$.
 *
 * @param key The metric's key, its export name.
 * @param name The metric's display name.
 * @param note The slug of the note that defines it.
 * @returns The metric's spec, for `defineMetric`.
 */
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
 * The expected calibration error of binary probabilities,
 * $\sum_m (\lvert B_m \rvert / n) \lvert \bar{y}(B_m) - \bar{p}(B_m) \rvert$ over $M$ bins $B_m$ (Naeini et al.,
 * 2015). Depends on the binning: report `bins` and `strategy`.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, each in $[0, 1]$.
 * @param options `bins`, the number of bins $M$ (default 10); `strategy`, `uniform` (default) or `quantile`; and
 *   `positive`, the positive class.
 * @returns The ECE, in $[0, 1]$.
 *
 * @example Two bins: calibrated below 0.5, under-confident above
 * const yTrue = [0, 0, 1, 0, 1, 1, 1, 1]
 * const probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * print(expectedCalibrationError(yTrue, probs, { bins: 2 }))
 */
export const expectedCalibrationError = defineMetric(
  calibrationInfo('expectedCalibrationError', 'Expected calibration error'),
  (yTrue: Labels, probabilities: Data, options: CalibrationOptions = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'expectedCalibrationError')
    return binnedCalibrationError(y, p, options, 1)
  },
)

/**
 * The maximum calibration error: the largest bin gap $\max_m \lvert \bar{y}_m - \bar{p}_m \rvert$ over non-empty bins
 * (Naeini et al., 2015).
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, each in $[0, 1]$.
 * @param options `bins`, the number of bins $M$ (default 10); `strategy`, `uniform` (default) or `quantile`; and
 *   `positive`, the positive class.
 * @returns The MCE, in $[0, 1]$.
 *
 * @example The worse of two bins
 * const yTrue = [0, 0, 1, 0, 1, 1, 1, 1]
 * const probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * print(maximumCalibrationError(yTrue, probs, { bins: 2 }))
 */
export const maximumCalibrationError = defineMetric(
  calibrationInfo('maximumCalibrationError', 'Maximum calibration error'),
  (yTrue: Labels, probabilities: Data, options: CalibrationOptions = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'maximumCalibrationError')
    return binnedCalibrationError(y, p, options, 'max')
  },
)

/**
 * The RMS ($L^2$) calibration error, $\sqrt{\sum_m (n_m / n)(\bar{y}_m - \bar{p}_m)^2}$ (Kumar et al., 2019), the
 * plug-in $\mathrm{CE}_2$.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, each in $[0, 1]$.
 * @param options `bins`, the number of bins $M$ (default 10); `strategy`, `uniform` (default) or `quantile`; and
 *   `positive`, the positive class.
 * @returns The RMS calibration error, in $[0, 1]$.
 *
 * @example Two bins
 * const yTrue = [0, 0, 1, 0, 1, 1, 1, 1]
 * const probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * print(rmsCalibrationError(yTrue, probs, { bins: 2 }))
 */
export const rmsCalibrationError = defineMetric(
  calibrationInfo('rmsCalibrationError', 'RMS calibration error', 'estimating-calibration-error'),
  (yTrue: Labels, probabilities: Data, options: CalibrationOptions = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'rmsCalibrationError')
    return binnedCalibrationError(y, p, options, 2)
  },
)

/**
 * The debiased squared calibration error (Kumar et al., 2019; estimating-calibration-error):
 * $\sum_m (n_m / n) [(\bar{y}_m - \bar{p}_m)^2 - \bar{y}_m (1 - \bar{y}_m) / (n_m - 1)]$, an estimate of
 * $\mathrm{CE}_2^2$ with the binomial noise of each bin subtracted. It can be negative; bins with one case contribute
 * nothing to the correction. Default equal-mass bins.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, each in $[0, 1]$.
 * @param options `bins`, the number of bins $M$ (default 10); `strategy`, `quantile` (default) or `uniform`; and
 *   `positive`, the positive class.
 * @returns The debiased estimate of the squared calibration error.
 *
 * @example Small bins: the noise correction can make it negative
 * const yTrue = [0, 0, 1, 0, 1, 1, 1, 1]
 * const probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * print('debiased =', debiasedSquaredCalibrationError(yTrue, probs, { bins: 4 }))
 * print('plug-in =', rmsCalibrationError(yTrue, probs, { bins: 4, strategy: 'quantile' }) ** 2)
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
 * $\mathrm{ECE}_{\mathrm{sweep}}$ (Roelofs et al., 2022): equal-mass bins, with the largest number of bins (up to
 * `maxBins`, default $n$) for which the bin frequencies are still non-decreasing. The bin counts are tried in turn
 * from 1, on predictions sorted once: $O(n \log n + M^2)$ for $M$ bin counts tried.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, each in $[0, 1]$.
 * @param options `maxBins`, the most bins to try (default $n$), and `positive`, the positive class.
 * @returns The ECE with equal-mass bins at the chosen count.
 *
 * @example Five bins are the most that stay monotone
 * const yTrue = [0, 0, 1, 0, 1, 1, 1, 1]
 * const probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * print(sweepCalibrationError(yTrue, probs))
 */
export const sweepCalibrationError = defineMetric(
  calibrationInfo('sweepCalibrationError', 'ECE sweep', 'estimating-calibration-error'),
  (yTrue: Labels, probabilities: Data, options: { positive?: Label; maxBins?: number } = {}): number => {
    const { y, p } = binaryInputs(yTrue, probabilities, options.positive, 'sweepCalibrationError')
    const n = p.length
    // Sort once, in the order `binAssignments` uses for equal-mass bins; then bin k of m holds the ranks r with
    // floor(rm/n) = k, from ceil(kn/m) up to ceil((k + 1)n/m), and its frequency comes from prefix sums of the outcomes.
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b] || a - b)
    const positives = new Float64Array(n + 1)
    for (let r = 0; r < n; r++) positives[r + 1] = positives[r] + y[order[r]]
    let best = 1
    for (let m = 1; m <= (options.maxBins ?? n); m++) {
      let monotone = true
      let previous = NaN
      for (let k = 0, start = 0; k < m && monotone; k++) {
        const end = Math.ceil(((k + 1) * n) / m)
        // An empty bin's frequency is NaN, which (as in the comparison of `binStatistics` frequencies) breaks nothing.
        const freq = end > start ? (positives[end] - positives[start]) / (end - start) : NaN
        if (freq < previous) monotone = false
        previous = freq
        start = end
      }
      if (!monotone) break
      best = m
    }
    return binnedCalibrationError(y, p, { bins: best, strategy: 'quantile' }, 1)
  },
)

/**
 * Confidence (top-label) ECE for multiclass probabilities (Guo et al., 2017): bin the largest probability of each case
 * and compare it with the accuracy of the predicted class in each bin. Of tied largest probabilities the first class
 * is predicted.
 *
 * @param yTrue The true labels.
 * @param probabilities An $n \times K$ matrix of class probabilities (a binary vector is read as two columns).
 * @param options `bins` and `strategy` as for `expectedCalibrationError`, and `labels`, the classes in column order.
 * @returns The confidence ECE, in $[0, 1]$.
 *
 * @example Three classes
 * const yTrue = [0, 1, 2, 2]
 * const probs = [[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]]
 * print(confidenceCalibrationError(yTrue, probs))
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
 * Classwise ECE (Kull et al., 2019): the binary ECE of each class's probability column against "is this class",
 * averaged over the $K$ classes.
 *
 * @param yTrue The true labels.
 * @param probabilities An $n \times K$ matrix of class probabilities.
 * @param options `bins` and `strategy` as for `expectedCalibrationError`, and `labels`, the classes in column order.
 * @returns The classwise ECE, in $[0, 1]$.
 *
 * @example Three classes
 * const yTrue = [0, 1, 2, 2]
 * const probs = [[0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]]
 * print(classwiseCalibrationError(yTrue, probs))
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
 * Consistency bars (Bröcker and Smith, 2007): for each bin of a reliability diagram, the central `level` interval
 * (default 0.9) of the observed frequency that a calibrated model would produce, from `resamples` (default 1000)
 * redraws of every label from $\Bern(p_i)$ with the stream. Empty bins hold NaN.
 *
 * @param s The random stream the labels are redrawn from; it is advanced.
 * @param probabilities The predicted probabilities $p_i$, each in $[0, 1]$.
 * @param options `bins` and `strategy`, the binning of `reliabilityDiagram` (default 10 `uniform` bins);
 *   `resamples`, the number of redraws; and `level`, the coverage of each bar.
 * @returns Per bin, the `lower` and `upper` ends of the bar and the mean prediction `meanPredicted`.
 *
 * @example Bars for two bins
 * const probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
 * const bars = consistencyBars(stream(0), probs, { bins: 2, resamples: 200 })
 * print('mean prediction =', bars.meanPredicted)
 * print('lower =', bars.lower, ' upper =', bars.upper)
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
  /** The Brier score, $\frac{1}{n} \sum_i (p_i - y_i)^2$. */
  brier: number
  /** How far each group's forecast is from its outcome frequency (lower is better). */
  reliability: number
  /** How far the groups' outcome frequencies are from the base rate (higher is better). */
  resolution: number
  /** The variance $\bar{o}(1 - \bar{o})$ of the outcomes, which no forecast changes. */
  uncertainty: number
  /**
   * $\mathrm{brier} - (\mathrm{reliability} - \mathrm{resolution} + \mathrm{uncertainty})$: zero when the forecasts
   * are grouped by their distinct values, and the within-bin variance terms when bins pool different forecasts.
   */
  residual: number
}

/**
 * Murphy's decomposition of the binary Brier score (Murphy, 1973; log-loss-and-brier-score):
 * $\mathrm{Brier} = \mathrm{reliability} - \mathrm{resolution} + \mathrm{uncertainty}$, with reliability
 * $\frac{1}{n} \sum_b n_b (f_b - \bar{o}_b)^2$, resolution $\frac{1}{n} \sum_b n_b (\bar{o}_b - \bar{o})^2$ and
 * uncertainty $\bar{o}(1 - \bar{o})$. Without `bins`, forecasts are grouped by distinct value and the identity is
 * exact; with `bins`, $f_b$ is each bin's mean forecast and `residual` reports what the bins leave out.
 *
 * @param yTrue The true labels.
 * @param probabilities $\pr(\text{positive})$ per case, each in $[0, 1]$.
 * @param options `bins`, the number of equal-width bins on $[0, 1]$ (left out: group by distinct forecast), and
 *   `positive`, the positive class.
 * @returns The Brier score, its three terms and the residual.
 *
 * @example Two distinct forecasts: the identity is exact
 * print(brierDecomposition([0, 1, 1, 0], [0.25, 0.75, 0.75, 0.25]))
 *
 * @example Bins pool different forecasts, and the residual shows it
 * print(brierDecomposition([0, 1, 1, 0, 1], [0.1, 0.7, 0.9, 0.3, 0.8], { bins: 2 }))
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

/**
 * A value per case as a Float64Array of length `n`: a number is repeated, data is checked against `n` (`ShapeError`),
 * and a traced value throws `DomainError`.
 *
 * @param x One number for every case, or one value per case.
 * @param n The number of cases.
 * @param what The caller's name, for error messages.
 * @returns The $n$ values.
 */
function perCase(x: PerCase | Value, n: number, what: string): Float64Array {
  if (typeof x === 'number') return new Float64Array(n).fill(x)
  if (isTraced(x)) throw new DomainError('metrics', `metrics: ${what}: traced values are not accepted`)
  const v = values(x as Data)
  if (v.length !== n) throw new ShapeError('metrics', `metrics: ${what}: ${v.length} values for ${n} cases`)
  return v
}

/**
 * A Gaussian forecast per case: means and standard deviations, or a model's predictive `Normal` distribution (the
 * `predictive` capability, as `evaluate` passes it), batch shape $[n]$.
 */
export type GaussianForecast = { mean: PerCase; sd: PerCase } | Distribution

/**
 * True for a distribution object (its `kind` is `distribution`).
 *
 * @param x Any value.
 * @returns Whether `x` is a distribution.
 */
const isDistribution = (x: unknown): x is Distribution =>
  typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'distribution'

/**
 * The means and standard deviations of a Gaussian forecast for $n$ cases; a predictive must be a `Normal`, or
 * `DomainError` is thrown.
 *
 * @param f The forecast: `mean` and `sd`, or a `Normal` predictive.
 * @param n The number of cases.
 * @param what The caller's name, for error messages.
 * @returns The mean `mu` and standard deviation `sd` of each case.
 */
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
 * The log score of any predictive distribution, $-\frac{1}{n} \sum_i \log p(y_i)$ (a density for continuous
 * predictives, a mass for discrete ones), in nats: the proper scoring rule behind log loss and the Gaussian log score
 * (Gneiting and Raftery, 2007, §4.1), read from the model's `predictive` through `logProb`. The predictive's batch is
 * the $n$ cases; for a categorical predictive, $y$ holds class indices. Anything but a distribution throws
 * `DomainError`.
 *
 * @param yTrue The observations $y_i$, one per case.
 * @param predictive The predictive distribution: any object of kind `distribution` whose `logProb` of the vector of
 *   observations gives one log density (or mass) per case.
 * @returns The mean negative log predictive density, in nats.
 *
 * @example A standard normal predictive, written out by hand
 * const predictive = {
 *   kind: 'distribution',
 *   name: 'Normal',
 *   logProb: (y) => toArray(y).map((v) => -0.5 * v * v - 0.5 * Math.log(2 * Math.PI)),
 * }
 * print('log score =', logScore([0, 1, -1], predictive))
 * print('Gaussian log score =', gaussianLogScore([0, 1, -1], { mean: 0, sd: 1 }))
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
 * The CRPS of Gaussian forecasts $\Gauss(\mu_i, \sigma_i^2)$, averaged over cases (Gneiting and Raftery, 2007, eq.
 * 21): with $z = (y - \mu)/\sigma$, $\mathrm{CRPS} = \sigma(z(2\Phi(z) - 1) + 2\phi(z) - 1/\sqrt{\pi})$. In the units
 * of $y$. The forecast is means and sds, or a model's Normal predictive (so `evaluate` serves it).
 *
 * @param yTrue The observations $y_i$, one per case.
 * @param forecast The Gaussian forecast: `mean` and `sd` per case (or one each for all cases), or a `Normal`
 *   predictive distribution with batch shape $[n]$. Another distribution throws `DomainError`.
 * @returns The mean CRPS, in the units of $y$.
 *
 * @example An observation at the mean, with two spreads
 * print('sd 1:', crpsGaussian([0], { mean: 0, sd: 1 }))
 * print('sd 2:', crpsGaussian([0], { mean: 0, sd: 2 }))
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
 * The CRPS of ensemble or sample forecasts in the kernel form
 * $\expect\lvert X - y \rvert - \frac{1}{2}\expect\lvert X - X' \rvert$ (Gneiting and Raftery, 2007), averaged over
 * cases. `samples` is an $n \times m$ matrix ($m$ draws per case) or, for one case, a vector. The plug-in form divides
 * the spread term by $m^2$; `fair: true` divides by $m(m - 1)$, which is unbiased for the underlying distribution.
 * Computed in $O(m \log m)$ per case from the sorted draws. A row count other than the number of observations throws
 * `ShapeError`; no cases, no draws, or a single draw with `fair` throws `DomainError`.
 *
 * @param yTrue The observations, one per case, or a single number for one case.
 * @param samples The draws: an $n \times m$ matrix with a row per case, or a vector of $m$ draws for one case.
 * @param options `fair`, true for the unbiased spread term (needs $m \ge 2$).
 * @returns The mean CRPS, in the units of $y$.
 *
 * @example Two draws either side of the observation
 * print('plug-in =', crpsEnsemble(0.5, [0, 1]))
 * print('fair =', crpsEnsemble(0.5, [0, 1], { fair: true }))
 *
 * @example Two cases, three draws each
 * print(crpsEnsemble([0, 2], [[-1, 0, 1], [0, 1, 2]]))
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
    nonEmpty(y.length, 'crpsEnsemble')
    const m = S.cols
    if (m < (options.fair ? 2 : 1))
      throw new DomainError(
        'crpsEnsemble',
        `crpsEnsemble: ${options.fair ? 'the fair score needs at least two draws' : 'needs at least one draw'} per case, got ${m}`,
      )
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
 * The negative log predictive density of Gaussian forecasts, $-\frac{1}{n} \sum_i \log \Gauss(y_i; \mu_i, \sigma_i^2)$
 * (the log score, orientated as a loss). The forecast is means and sds, or a model's Normal predictive (so `evaluate`
 * serves it); `logScore` is the same score for any predictive distribution.
 *
 * @param yTrue The observations $y_i$, one per case.
 * @param forecast The Gaussian forecast: `mean` and `sd` per case (or one each for all cases), or a `Normal`
 *   predictive distribution with batch shape $[n]$. Another distribution throws `DomainError`.
 * @returns The mean negative log density, in nats.
 *
 * @example Standard normal forecasts: half of log 2 pi, plus half the mean square
 * print(gaussianLogScore([0, 1, -1], { mean: 0, sd: 1 }))
 * print(0.5 * Math.log(2 * Math.PI) + (0 + 1 + 1) / 6)
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

/**
 * A central prediction interval per case: its `lower` end $\ell$ and `upper` end $u$, each one value per case or one
 * for all.
 */
export type Interval = { lower: PerCase; upper: PerCase }

/**
 * The interval score of central $(1 - \alpha)$ prediction intervals, averaged over cases (Gneiting and Raftery, 2007,
 * §6.2): $(u - \ell) + \frac{2}{\alpha}(\ell - y)\indicator[y < \ell] + \frac{2}{\alpha}(y - u)\indicator[y > u]$.
 *
 * @param yTrue The observations $y_i$, one per case.
 * @param interval The intervals $[\ell, u]$.
 * @param options `alpha`, the nominal miss rate $\alpha$ of the intervals (required; 0.1 for 90% intervals).
 * @returns The mean interval score, in the units of $y$.
 *
 * @example Width 6, with one miss below and one above
 * print(intervalScore([1, 5, 10], { lower: 2, upper: 8 }, { alpha: 0.2 }))
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

/**
 * Coverage: the fraction of observations inside their interval $[\ell, u]$ (report beside a score, not as one).
 *
 * @param yTrue The observations $y_i$, one per case.
 * @param interval The intervals $[\ell, u]$, ends included.
 * @returns The fraction covered, in $[0, 1]$.
 *
 * @example One of three inside
 * print(coverage([1, 5, 10], { lower: 2, upper: 8 }))
 */
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
 * Probability integral transform values $u_i = F_i(y_i)$ (quantile-calibration): uniform on $[0, 1]$ for a calibrated
 * forecaster. Gaussian forecasts give $\Phi((y - \mu)/\sigma)$; ensembles (an $n \times m$ matrix of draws) give the
 * fraction of draws $\le y$, or with `stream` the randomised PIT that spreads ties uniformly between the fractions
 * below and at $y$.
 *
 * @param yTrue The observations $y_i$, one per case.
 * @param forecast `mean` and `sd` per case (or one each for all), or `samples`, an $n \times m$ matrix of draws with a
 *   row per case.
 * @param options `stream`, the random stream for the randomised PIT of an ensemble; left out, ties count as below.
 * @returns The PIT value of each case.
 *
 * @example Gaussian forecasts and an ensemble
 * print('Gaussian:', pitValues([0, 1.96], { mean: 0, sd: 1 }))
 * print('ensemble:', pitValues([2], { samples: [[1, 2, 2, 3]] }))
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
 * $\exp(-\frac{1}{T} \sum_t \log p(x_t \mid x_{<t}))$ (perplexity-as-a-metric).
 *
 * @param logProbabilities The natural-log probability the model gave each of the $T$ tokens.
 * @returns The perplexity, at least 1.
 *
 * @example A uniform guess among four tokens
 * print(perplexity([Math.log(0.25), Math.log(0.25), Math.log(0.25)]))
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
