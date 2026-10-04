/**
 * Calibration maps fitted on held-out predictions: temperature scaling and Dirichlet calibration of multiclass
 * outputs, beta calibration and histogram binning of binary scores, the isotonic map (pool adjacent violators) as a
 * function of new scores, and the top-label confidence used to draw multiclass reliability diagrams. Platt scaling of
 * real-valued scores is in `./platt`. Every parametric map is fitted by L-BFGS on the cross-entropy, through autodiff.
 */

import type { MatrixLike, Objective, VectorLike } from 'aifn-compute/foundation/contracts'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  add,
  concat,
  dense,
  div,
  exp,
  get,
  matmul,
  mul,
  ones,
  reshape,
  square,
  sum,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { binaryCrossEntropyWithLogits, softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import { reliabilityDiagram } from 'aifn-compute/learning/metrics'
import { sigmoid, softmax } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'
import { isotonicRegression, type IsotonicFit } from './isotonic'

type F64 = dense.F64

/** Probabilities are clipped to [EPS, 1 − EPS] before a log. */
const EPS = 1e-12

function readLabels(labels: VectorLike, n: number, K: number, where: string): Int32Array {
  const y = dense.toF64(labels, where)
  if (y.length !== n) throw new DomainError(where, `${where}: ${n} predictions and ${y.length} labels`)
  for (const v of y)
    if (!Number.isInteger(v) || v < 0 || v >= K)
      throw new DomainError(where, `${where}: label ${v} is not in 0 … ${K - 1}`)
  return Int32Array.from(y)
}

function readScores(scores: VectorLike, where: string): F64 {
  const s = dense.toF64(scores, where)
  for (const v of s) if (!(v >= 0 && v <= 1)) throw new DomainError(where, `${where}: score ${v} is not in [0, 1]`)
  return s
}

const meanCrossEntropy = (logits: Tensor, y: Int32Array): number =>
  softmaxCrossEntropy(logits, y, { reduction: 'mean' }) as number

// ── Temperature scaling ──────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted temperature. */
export interface TemperatureScaling {
  /** T > 0: the logits are divided by T. */
  readonly temperature: number
  /** The mean cross-entropy before (T = 1) and after. */
  readonly logLossBefore: number
  readonly logLossAfter: number
  /** softmax(z/T) for logits [m, K]. */
  apply(logits: MatrixLike): Tensor
}

/**
 * Temperature scaling (Guo et al., 2017): one T > 0 dividing every logit, chosen to minimise the mean cross-entropy
 * on held-out logits [n, K] and labels [n]. The predicted class never changes. The loss is convex in 1/T; it is
 * minimised over log T by L-BFGS.
 */
export function temperatureScaling(logits: MatrixLike, labels: VectorLike): TemperatureScaling {
  const where = 'temperatureScaling'
  const { data, m: n, n: K } = dense.toMatrixF64(logits, where)
  const y = readLabels(labels, n, K, where)
  const Z = dense.mat(data, n, K)
  const objective: Objective = {
    kind: 'objective',
    name: 'temperature log-loss',
    dim: 1,
    value: (u) => softmaxCrossEntropy(div(Z, exp(get(u, 0))), y, { reduction: 'mean' }),
  }
  const fit = minimize(objective, [0], { maxSteps: 200 })
  const temperature = Math.exp(dense.data(fit.x as Tensor)[0])
  const apply = (z: MatrixLike) => {
    const { data: d, m, n: k } = dense.toMatrixF64(z, where)
    return softmax(
      dense.mat(
        d.map((v) => v / temperature),
        m,
        k,
      ),
    )
  }
  return {
    temperature,
    logLossBefore: meanCrossEntropy(Z, y),
    logLossAfter: meanCrossEntropy(
      dense.mat(
        data.map((v) => v / temperature),
        n,
        K,
      ),
      y,
    ),
    apply,
  }
}

// ── Beta calibration ─────────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted beta calibration map. */
export interface BetaCalibration {
  /** μ(s) = σ(a ln s − b ln(1 − s) + c). */
  readonly a: number
  readonly b: number
  readonly c: number
  apply(scores: VectorLike): Tensor
}

/**
 * Beta calibration (Kull, Silva Filho and Flach, 2017): μ(s) = σ(a ln s − b ln(1 − s) + c) for binary scores s ∈
 * [0, 1], a logistic regression on the features ln s and −ln(1 − s). The family contains the identity (a = b = 1,
 * c = 0) and is the exact posterior when each class's scores are beta-distributed. As the paper recommends, a or b is
 * held at 0 (and the fit repeated) if it comes out negative, so the map is monotone.
 */
export function betaCalibration(scores: VectorLike, labels: VectorLike): BetaCalibration {
  const where = 'betaCalibration'
  const s = readScores(scores, where)
  const n = s.length
  const y = Float64Array.from(readLabels(labels, n, 2, where))
  const features = (v: F64) => {
    const out = new Float64Array(v.length * 2)
    v.forEach((p, i) => {
      const q = Math.min(1 - EPS, Math.max(EPS, p))
      out[2 * i] = Math.log(q)
      out[2 * i + 1] = -Math.log(1 - q)
    })
    return out
  }
  const X = features(s)
  // Fit (a, b, c) with the coordinates in `free`, the others held at 0.
  const fit = (free: readonly boolean[]) => {
    const columns = free.map((f, j) => (f ? j : -1)).filter((j) => j >= 0)
    const k = columns.length
    const D = new Float64Array(n * (k + 1))
    for (let i = 0; i < n; i++) {
      columns.forEach((j, c) => (D[i * (k + 1) + c] = X[2 * i + j]))
      D[i * (k + 1) + k] = 1
    }
    const design = dense.mat(D, n, k + 1)
    const objective: Objective = {
      kind: 'objective',
      name: 'beta calibration log-loss',
      dim: k + 1,
      value: (w) => binaryCrossEntropyWithLogits(matmul(design, w), y, { reduction: 'mean' }),
    }
    const w = dense.data(
      minimize(
        objective,
        Float64Array.from({ length: k + 1 }, (_, j) => (j < k ? 1 : 0)),
        { maxSteps: 300 },
      ).x as Tensor,
    )
    const coef = [0, 0]
    columns.forEach((j, c) => (coef[j] = w[c]))
    return { a: coef[0], b: coef[1], c: w[k] }
  }
  let p = fit([true, true])
  if (p.a < 0 && p.b < 0) p = fit([false, false])
  else if (p.a < 0) p = fit([false, true])
  else if (p.b < 0) p = fit([true, false])
  const { a, b, c } = p
  return {
    a,
    b,
    c,
    apply: (v) => {
      const f = features(dense.toF64(v, where))
      return dense.vec(
        Float64Array.from({ length: f.length / 2 }, (_, i) => sigmoid(a * f[2 * i] + b * f[2 * i + 1] + c) as number),
      )
    },
  }
}

// ── Dirichlet calibration ────────────────────────────────────────────────────────────────────────────────────────────

/** A fitted Dirichlet calibration map. */
export interface DirichletCalibration {
  /** The matrix W [K, K] and offsets b [K] of softmax(W ln q + b). */
  readonly weights: Tensor
  readonly bias: Tensor
  readonly logLossBefore: number
  readonly logLossAfter: number
  /** The map for probability rows [m, K]. */
  apply(probabilities: MatrixLike): Tensor
}

/**
 * Dirichlet calibration (Kull et al., 2019): q ↦ softmax(W ln q + b) for probability rows q [n, K], a multinomial
 * logistic regression on the log-probabilities, fitted from the identity (W = I, b = 0) by L-BFGS. The ODIR penalty
 * λ/(K(K − 1)) Σ_{i≠j} W²ᵢⱼ + μ/K Σ b²ⱼ shrinks the off-diagonal entries and the offsets (defaults λ = μ = 10⁻³). With
 * W = I/T and b = 0 it is temperature scaling on the log-probabilities.
 */
export function dirichletCalibration(
  probabilities: MatrixLike,
  labels: VectorLike,
  options: { lambda?: number; mu?: number; maxSteps?: number } = {},
): DirichletCalibration {
  const where = 'dirichletCalibration'
  const { data, m: n, n: K } = dense.toMatrixF64(probabilities, where)
  const y = readLabels(labels, n, K, where)
  const lambda = options.lambda ?? 1e-3
  const mu = options.mu ?? 1e-3
  const logs = (d: F64, m: number) =>
    concat(
      [
        dense.mat(
          d.map((v) => Math.log(Math.min(1, Math.max(EPS, v)))),
          m,
          K,
        ),
        ones([m, 1]),
      ],
      1,
    )
  const X = logs(data, n)
  // The parameters θ [K + 1, K], flattened: softmax([ln q, 1] θ), so θ's first K rows are Wᵀ and its last row b.
  const offDiagonal = Float64Array.from({ length: (K + 1) * K }, (_, i) => {
    const r = Math.floor(i / K)
    const c = i % K
    return r < K && r !== c ? 1 : 0
  })
  const intercept = Float64Array.from({ length: (K + 1) * K }, (_, i) => (Math.floor(i / K) === K ? 1 : 0))
  const offPenalty = K > 1 ? lambda / (K * (K - 1)) : 0
  const objective: Objective = {
    kind: 'objective',
    name: 'Dirichlet calibration log-loss',
    dim: (K + 1) * K,
    value: (w) => {
      const ce = softmaxCrossEntropy(matmul(X, reshape(w, [K + 1, K])), y, { reduction: 'mean' })
      const sq = square(w)
      return add(
        ce,
        add(mul(offPenalty, sum(mul(dense.vec(offDiagonal), sq))), mul(mu / K, sum(mul(dense.vec(intercept), sq)))),
      )
    },
  }
  const start = Float64Array.from({ length: (K + 1) * K }, (_, i) => (Math.floor(i / K) === i % K ? 1 : 0))
  const fit = minimize(objective, start, { maxSteps: options.maxSteps ?? 500 })
  const theta = dense.mat(dense.data(fit.x as Tensor), K + 1, K)
  const weights = dense.mat(dense.transpose(dense.data(theta).subarray(0, K * K), K, K), K, K)
  const bias = dense.vec(Float64Array.from(dense.data(theta).subarray(K * K)))
  const apply = (q: MatrixLike) => {
    const { data: d, m } = dense.toMatrixF64(q, where)
    return softmax(matmul(logs(d, m), theta) as Tensor)
  }
  return {
    weights,
    bias,
    // softmax(ln q) = q, so the log-probabilities are the logits of the uncalibrated map.
    logLossBefore: meanCrossEntropy(
      dense.mat(
        data.map((v) => Math.log(Math.min(1, Math.max(EPS, v)))),
        n,
        K,
      ),
      y,
    ),
    logLossAfter: meanCrossEntropy(matmul(X, theta) as Tensor, y),
    apply,
  }
}

// ── Histogram binning and isotonic maps ──────────────────────────────────────────────────────────────────────────────

/** A fitted histogram-binning map. */
export interface HistogramBinning {
  /** The bin edges [M + 1] (0 … 1 for uniform bins; score quantiles for equal-mass bins). */
  readonly edges: Tensor
  /** The calibrated value of each bin: its fraction of positives (the bin's centre when it is empty) [M]. */
  readonly values: Tensor
  readonly counts: Tensor
  apply(scores: VectorLike): Tensor
}

/**
 * Histogram binning (Zadrozny and Elkan, 2001): binary scores are cut into M bins (equal width or equal mass) and each
 * bin's scores are replaced by the fraction of positives among its training cases. The bins are those of the
 * reliability diagram (`reliabilityDiagram` in `aifn-compute/learning/metrics`); an empty bin keeps its centre.
 */
export function histogramBinning(
  scores: VectorLike,
  labels: VectorLike,
  options: { bins?: number; strategy?: 'uniform' | 'quantile' } = {},
): HistogramBinning {
  const where = 'histogramBinning'
  const s = readScores(scores, where)
  const y = readLabels(labels, s.length, 2, where)
  const bins = options.bins ?? 10
  const d = reliabilityDiagram(y, s, { bins, strategy: options.strategy ?? 'uniform', positive: 1 })
  const edges = Float64Array.from(dense.data(d.edges))
  // Equal-mass bins with fewer distinct scores than bins leave NaN edges: carry the last edge forward.
  for (let k = 1; k < edges.length; k++) if (!Number.isFinite(edges[k])) edges[k] = edges[k - 1]
  edges[0] = 0
  edges[edges.length - 1] = 1
  const freq = dense.data(d.y)
  const values = Float64Array.from(freq, (v, k) => (Number.isFinite(v) ? v : (edges[k] + edges[k + 1]) / 2))
  // Uniform bins are found as the reliability diagram finds them (⌊sM⌋, the last closed at 1), so a training score
  // lands in the bin whose frequency it contributed to even where k/M rounds differently from s·M.
  const uniform = (options.strategy ?? 'uniform') === 'uniform'
  const binOf = (v: number) => {
    if (uniform) return Math.min(bins - 1, Math.max(0, Math.floor(v * bins)))
    let k = 0
    while (k < bins - 1 && v >= edges[k + 1]) k++
    return k
  }
  return {
    edges: dense.vec(edges),
    values: dense.vec(values),
    counts: d.counts,
    apply: (v) => dense.vec(dense.toF64(v, where).map((x) => values[binOf(x)])),
  }
}

/** A fitted isotonic calibration map. */
export interface IsotonicCalibration extends IsotonicFit {
  /** The step function: each score takes the value of the last block that starts at or below it. */
  apply(scores: VectorLike): Tensor
}

/**
 * Isotonic calibration (Zadrozny and Elkan, 2002): the monotone non-decreasing map of scores to probabilities that
 * fits the labels best in squared error, found by pool adjacent violators (`isotonicRegression`), and applied to new
 * scores as a step function.
 */
export function isotonicCalibration(scores: VectorLike, labels: VectorLike): IsotonicCalibration {
  const where = 'isotonicCalibration'
  const s = dense.toF64(scores, where)
  const y = Float64Array.from(readLabels(labels, s.length, 2, where))
  const fit = isotonicRegression(y, { x: s })
  const t = dense.data(fit.thresholds)
  const v = dense.data(fit.values)
  return {
    ...fit,
    apply: (q) =>
      dense.vec(
        dense.toF64(q, where).map((x) => {
          let k = 0
          while (k < t.length - 1 && x >= t[k + 1]) k++
          return v[k]
        }),
      ),
  }
}

/**
 * The top-label view of multiclass probabilities [n, K] and labels [n]: each case's confidence maxₖ pₖ and whether its
 * top class is correct (1) or not (0), the inputs of a confidence reliability diagram and of top-label ECE.
 */
export function topLabelConfidence(
  probabilities: MatrixLike,
  labels: VectorLike,
): { confidence: Tensor; correct: Tensor; predicted: Tensor } {
  const where = 'topLabelConfidence'
  const { data, m: n, n: K } = dense.toMatrixF64(probabilities, where)
  const y = readLabels(labels, n, K, where)
  const confidence = new Float64Array(n)
  const correct = new Float64Array(n)
  const predicted = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let k = 0
    for (let j = 1; j < K; j++) if (data[i * K + j] > data[i * K + k]) k = j
    confidence[i] = data[i * K + k]
    predicted[i] = k
    correct[i] = k === y[i] ? 1 : 0
  }
  return { confidence: dense.vec(confidence), correct: dense.vec(correct), predicted: dense.vec(predicted) }
}
